import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="Eye Wellness Pro AI", layout="wide")

# JS: Phát âm thanh và Thông báo đẩy hệ thống
def trigger_alert_js(title, message):
    js_code = f"""
    <script>
    if (Notification.permission !== "granted") {{ Notification.requestPermission(); }}
    var audio = new Audio("https://www.soundjay.com/buttons/beep-01a.mp3");
    audio.play();
    if (Notification.permission === "granted") {{
        new Notification("{title}", {{ body: "{message}" }});
    }}
    </script>
    """
    components.html(js_code, height=0, width=0)

st.title("🛡️ AI Eye Guard - Bảo vệ mắt đa tầng")

col_vid, col_info = st.columns([2, 1])

with col_info:
    st.subheader("📊 Giám sát thực tế")
    status_msg = st.empty()
    dist_msg = st.empty()
    st.divider()
    st.info("Hệ thống sẽ bắn thông báo ra góc màn hình nếu bạn sang Tab khác mà quên chớp mắt hoặc ngồi quá gần.")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink_time = time.time()
        self.is_too_close = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        curr_time = time.time()

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            # 1. Đo chớp mắt
            le_v = np.abs(res[159].y - res[145].y)
            le_h = np.abs(res[33].x - res[263].x)
            if (le_v / (le_h + 1e-6)) < 0.2: self.last_blink_time = curr_time

            # 2. Đo khoảng cách (Nhìn gần/xa)
            # Khoảng cách giữa 2 mống mắt (pixel)
            iris_dist = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.is_too_close = iris_dist > 110 # Ngưỡng ngồi quá sát

            if self.is_too_close:
                cv2.putText(img, "BACK AWAY!", (w//4, h//2), 2, 1.5, (0,0,255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-guard",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# Logic quét thông báo đẩy
if ctx.video_processor:
    while True:
        if (time.time() - ctx.video_processor.last_blink_time) > 10:
            trigger_alert_js("Cảnh báo mắt", "Bạn quên chớp mắt quá lâu rồi!")
            status_msg.error("⚠️ QUÊN CHỚP MẮT")
            time.sleep(5)
        if ctx.video_processor.is_too_close:
            trigger_alert_js("Cảnh báo khoảng cách", "Bạn ngồi quá sát màn hình!")
            dist_msg.warning("🚫 NGỒI QUÁ GẦN")
            time.sleep(5)
        time.sleep(1)
