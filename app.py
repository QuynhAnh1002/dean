import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="Eye Wellness Pro AI", layout="wide")

# JavaScript để bắn thông báo ra góc màn hình
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

st.title("🛡️ AI Eye Guard - Bảo vệ mắt đa tầng")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink = time.time()
        self.too_close = False

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            # 1. EAR (Chớp mắt)
            ear = np.abs(res[159].y - res[145].y) / (np.abs(res[33].x - res[263].x) + 1e-6)
            if ear < 0.2: self.last_blink = time.time()

            # 2. Khoảng cách (Iris Distance) - Nhìn gần/xa
            dist = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.too_close = dist > 115 

            if self.too_close:
                cv2.putText(img, "BACK AWAY!", (w//4, h//2), 2, 1.5, (0,0,255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="eye-pro", 
    video_processor_factory=EyeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

if ctx.video_processor:
    if (time.time() - ctx.video_processor.last_blink) > 10:
        trigger_alert_js("Cảnh báo mắt", "Bạn quên chớp mắt quá lâu rồi!")
    if ctx.video_processor.too_close:
        trigger_alert_js("Cảnh báo khoảng cách", "Bạn đang ngồi quá sát màn hình!")
