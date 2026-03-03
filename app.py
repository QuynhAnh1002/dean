import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Eye Guard - Visual Tracking", layout="wide")

# JavaScript: Thông báo hệ thống
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

st.title("🛡️ AI Eye Guard - Theo dõi chuyển động mắt")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Thông số thực tế")
    blink_metric = st.empty()
    dist_status = st.empty()
    st.divider()
    st.info("Khung xanh trên mắt: Đang mở. Khung đỏ: Đang nhắm/Quên chớp.")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink = time.time()
        self.start_time = time.time()
        self.too_close = False
        self.ear = 1.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        curr = time.time()
        
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) 
                                   for p in results.multi_face_landmarks[0].landmark])
            
            # 1. Lấy danh sách điểm cho mắt trái (ID: 33, 160, 158, 133, 153, 144)
            left_eye = mesh_points[[33, 160, 158, 133, 153, 144]]
            right_eye = mesh_points[[362, 385, 387, 263, 373, 380]]

            # 2. Tính EAR
            v_dist = np.linalg.norm(mesh_points[159] - mesh_points[145])
            h_dist = np.linalg.norm(mesh_points[33] - mesh_points[263])
            self.ear = v_dist / (h_dist + 1e-6)

            # 3. Vẽ khung theo dõi chuyển động
            color = (0, 255, 0) # Mặc định là xanh (An toàn)
            if self.ear < 0.23: 
                self.last_blink = curr
                color = (0, 0, 255) # Đổi sang đỏ khi nhắm/chớp

            # Vẽ đường bao quanh mắt
            cv2.polylines(img, [left_eye], True, color, 1, cv2.LINE_AA)
            cv2.polylines(img, [right_eye], True, color, 1, cv2.LINE_AA)
            
            # Vẽ điểm mống mắt (Iris) để theo dõi hướng nhìn
            cv2.circle(img, tuple(mesh_points[468]), 2, (255, 255, 255), -1)
            cv2.circle(img, tuple(mesh_points[473]), 2, (255, 255, 255), -1)

            # 4. Logic Cảnh báo
            diff = int(curr - self.last_blink)
            if diff > 10:
                cv2.putText(img, "BLINK NOW!", (w//3, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 3)
                cv2.rectangle(img, (0,0), (w,h), (0,0,255), 10) # Khung đỏ cảnh báo

            # 5. Khoảng cách
            iris_dist = np.linalg.norm(mesh_points[468] - mesh_points[473])
            self.too_close = iris_dist > 115

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-visual-track",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

if ctx.video_processor:
    last_alert = 0
    while True:
        curr_loop = time.time()
        sec_no_blink = int(curr_loop - ctx.video_processor.last_blink)
        blink_metric.metric("Chưa chớp mắt", f"{sec_no_blink}s")

        if sec_no_blink > 10 and (curr_loop - last_alert) > 5:
            trigger_alert_js("Cảnh báo mỏi mắt!", "Hãy chớp mắt ngay.")
            last_alert = curr_loop

        if ctx.video_processor.too_close:
            dist_status.error("🚫 QUÁ GẦN MÀN HÌNH")
        else:
            dist_status.success("✅ Khoảng cách an toàn")

        time.sleep(0.5)
