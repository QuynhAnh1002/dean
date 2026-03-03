import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Eye Guard Pro", layout="wide")

# JavaScript: Chỉ dùng để phát âm thanh BEEP và Thông báo hệ thống
def trigger_alert_js(title, message):
    js_code = f"""
    <script>
    if (Notification.permission !== "granted") {{ Notification.requestPermission(); }}
    new Audio("https://www.soundjay.com/buttons/beep-01a.mp3").play();
    if (Notification.permission === "granted") {{
        new Notification("{title}", {{ body: "{message}" }});
    }}
    </script>
    """
    components.html(js_code, height=0, width=0)

st.title("🛡️ AI Eye Guard - Hệ thống bảo vệ mắt")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Giám sát thông minh")
    status_ui = st.empty() 
    blink_ui = st.empty()  
    timer_ui = st.empty()  
    st.divider()
    st.warning("⚠️ Chế độ: Chỉ sử dụng Camera (Đã tắt Microphone).")
    st.info("Hệ thống sẽ nhắc nhở nếu bạn:\n1. Quên chớp mắt > 10s.\n2. Ngồi quá gần màn hình.\n3. Làm việc liên tục > 20 phút.")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink = time.time()
        self.start_time = time.time()
        self.too_close = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        curr = time.time()

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            
            # 1. Đo chớp mắt (EAR)
            ear = np.abs(res[159].y - res[145].y) / (np.abs(res[33].x - res[263].x) + 1e-6)
            if ear < 0.2: 
                self.last_blink = curr

            # 2. Đo khoảng cách (Nhìn gần/xa)
            # Tính khoảng cách mống mắt (Iris)
            dist = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.too_close = dist > 115 

            # Vẽ cảnh báo trực tiếp lên khung hình Video
            if self.too_close:
                cv2.putText(img, "BACK AWAY!", (w//4, h//2), 2, 1.5, (0,0,255), 3)
            
            if (curr - self.last_blink) > 10:
                cv2.putText(img, "PLEASE BLINK!", (50, 80), 2, 1.2, (0,165,255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    # Cấu hình chỉ lấy Video, không lấy Audio (Tắt Mic)
    ctx = webrtc_streamer(
        key="eye-pro", 
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False} 
    )

# --- CẬP NHẬT GIAO DIỆN ---
if ctx.video_processor:
    while True:
        curr = time.time()
        
        # Cập nhật trạng thái khoảng cách
        if ctx.video_processor.too_close:
            status_ui.error("🚫 CẢNH BÁO: NGỒI QUÁ GẦN")
            trigger_alert_js("Lùi xa ra!", "Bạn đang ngồi quá sát màn hình.")
        else:
            status_ui.success("✅ Khoảng cách an toàn")

        # Cập nhật nhắc chớp mắt 10s
        time_diff = int(curr - ctx.video_processor.last_blink)
        blink_ui.metric("Chưa chớp mắt trong", f"{time_diff} giây")
        if time_diff > 10:
            trigger_alert_js("Hãy chớp mắt!", "Bạn đã không chớp mắt hơn 10 giây.")

        # Đếm ngược 20 phút
        elapsed = int(curr - ctx.video_processor.start_time)
        rem = max(0, (20 * 60) - elapsed)
        mm, ss = divmod(rem, 60)
        timer_ui.metric("Nghỉ ngơi sau", f"{mm:02d}:{ss:02d}")
        
        if rem <= 0:
            trigger_alert_js("Đã làm việc 20 phút!", "Hãy nhìn xa 20 feet trong 20 giây.")
            ctx.video_processor.start_time = time.time()
        
        time.sleep(1)
