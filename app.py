import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="Eye Wellness Pro AI", layout="wide")

# --- JAVASCRIPT: PHÁT ÂM THANH & THÔNG BÁO ĐẨY (GÓC MÀN HÌNH) ---
def trigger_alert_js(title, message):
    js_code = f"""
    <script>
    // Xin quyền thông báo nếu chưa có
    if (Notification.permission !== "granted") {{
        Notification.requestPermission();
    }}
    
    // Phát âm thanh
    var audio = new Audio("https://www.soundjay.com/buttons/beep-01a.mp3");
    audio.play();

    // Bắn thông báo góc màn hình (hiện cả khi đang ở Tab khác)
    if (Notification.permission === "granted") {{
        new Notification("{title}", {{ body: "{message}", icon: "https://cdn-icons-png.flaticon.com/512/564/564022.png" }});
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
    st.write("**Cơ chế bảo vệ:**")
    st.info("1. Quên chớp mắt (>10s)\n2. Ngồi quá gần (<50cm)\n3. Làm việc quá lâu (>20p)")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink_time = time.time()
        self.start_time = time.time()
        self.is_too_close = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        curr_time = time.time()
        
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            
            # --- LOGIC 1: CHỚP MẮT ---
            le_v = np.abs(res[159].y - res[145].y)
            le_h = np.abs(res[33].x - res[263].x)
            ear = le_v / (le_h + 1e-6)
            if ear < 0.2: self.last_blink_time = curr_time

            # --- LOGIC 2: KHOẢNG CÁCH (NHÌN GẦN/XA) ---
            # Tính khoảng cách giữa 2 đồng tử (Iris distance) theo pixel
            iris_dist = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.is_too_close = iris_dist > 115 # Chỉ số 115 tương đương khoảng 50cm tùy camera

            # Vẽ cảnh báo nhìn gần lên màn hình
            if self.is_too_close:
                cv2.putText(img, "TOO CLOSE! BACK AWAY", (w//10, h//2), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-guard",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )

# --- VÒNG LẶP QUÉT TRẠNG THÁI ĐỂ BẮN THÔNG BÁO ---
if ctx.video_processor:
    while True:
        curr = time.time()
        # Kiểm tra quên chớp mắt
        if (curr - ctx.video_processor.last_blink_time) > 10:
            status_msg.error("⚠️ BẠN QUÊN CHỚP MẮT!")
            trigger_alert_js("Cảnh báo mắt", "Bạn quên chớp mắt quá lâu rồi, hãy chớp mắt đi!")
            time.sleep(5) # Tránh bắn thông báo quá dày

        # Kiểm tra ngồi quá gần
        if ctx.video_processor.is_too_close:
            dist_msg.warning("🚫 NGỒI QUÁ GẦN MÀN HÌNH!")
            trigger_alert_js("Cảnh báo khoảng cách", "Bạn đang ngồi quá sát màn hình, hãy lùi xa ra!")
            time.sleep(5)
            
        time.sleep(1)
