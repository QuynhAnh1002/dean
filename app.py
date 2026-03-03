import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Eye Guard Pro", layout="wide")

# JavaScript: Thông báo đẩy và âm thanh
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

st.title("🛡️ AI Eye Guard - Dashboard Bảo Vệ Mắt")

# Tạo 2 cột: Trái cho Camera, Phải cho Thông số
col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Thông số thời gian thực")
    status_ui = st.empty()     # Hiển thị: Đang nhìn gần hay xa
    blink_ui = st.empty()      # Hiển thị: Lần chớp mắt cuối cách đây bao lâu
    timer_ui = st.empty()      # Hiển thị: Đồng hồ đếm ngược 20 phút
    st.divider()
    st.markdown("### 🛠️ Hướng dẫn")
    st.info("- Nhìn chằm chằm > 10s: Nhắc chớp mắt.\n- Ngồi quá gần: Nhắc lùi xa.\n- Sau 20 phút: Nhắc nghỉ ngơi.")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink = time.time()
        self.start_time = time.time()
        self.too_close = False
        self.ear = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        curr = time.time()

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            
            # 1. Logic Chớp mắt
            self.ear = np.abs(res[159].y - res[145].y) / (np.abs(res[33].x - res[263].x) + 1e-6)
            if self.ear < 0.2: 
                self.last_blink = curr

            # 2. Logic Khoảng cách
            dist = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.too_close = dist > 115 

            # Vẽ trực tiếp lên màn hình cam
            if self.too_close:
                cv2.putText(img, "BACK AWAY!", (w//4, h//2), 2, 1.5, (0,0,255), 3)
            if (curr - self.last_blink) > 10:
                cv2.putText(img, "BLINK NOW!", (50, 80), 2, 1.2, (0,165,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-pro", 
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )

# --- VÒNG LẶP CẬP NHẬT UI ---
if ctx.video_processor:
    while True:
        curr = time.time()
        
        # 1. Cập nhật trạng thái khoảng cách
        if ctx.video_processor.too_close:
            status_ui.error("🚫 TRẠNG THÁI: NGỒI QUÁ GẦN")
            trigger_alert_js("Cảnh báo khoảng cách", "Vui lòng lùi xa màn hình!")
        else:
            status_ui.success("✅ TRẠNG THÁI: KHOẢNG CÁCH AN TOÀN")

        # 2. Cập nhật thời gian chớp mắt
        time_since_blink = int(curr - ctx.video_processor.last_blink)
        blink_ui.metric("Thời gian chưa chớp mắt", f"{time_since_blink} giây")
        if time_since_blink > 10:
            trigger_alert_js("Mỏi mắt rồi!", "Hãy chớp mắt để làm ẩm võng mạc.")

        # 3. Cập nhật đồng hồ 20 phút (Quy tắc 20-20-20)
        elapsed = int(curr - ctx.video_processor.start_time)
        remaining = max(0, (20 * 60) - elapsed)
        mins, secs = divmod(remaining, 60)
        timer_ui.metric("Thời gian đến kỳ nghỉ", f"{mins:02d}:{secs:02d}")
        
        if remaining <= 0:
            trigger_alert_js("Hết 20 phút!", "Hãy nhìn xa 20 feet trong 20 giây!")
            ctx.video_processor.start_time = time.time() # Reset đồng hồ
        
        time.sleep(1)
