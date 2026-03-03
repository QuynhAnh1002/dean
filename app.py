import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Eye Guard - Precision Tracker", layout="wide")

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

st.title("🛡️ AI Eye Guard - Chế độ nhận diện chính xác")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Trạng thái nhãn cầu")
    blink_metric = st.empty()
    st.divider()
    st.info("💡 Chế độ mới: Chỉ khi mí mắt trên và dưới gần như chạm nhau (trùng nhau), khung mới chuyển sang MÀU ĐỎ.")

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
            
            # Tọa độ mí mắt trái và phải để vẽ khung
            left_eye = mesh_points[[33, 160, 158, 133, 153, 144]]
            right_eye = mesh_points[[362, 385, 387, 263, 373, 380]]

            # TÍNH TOÁN EAR (Khoảng cách giữa các điểm mí mắt)
            # 159 (mí trên) và 145 (mí dưới)
            v_dist = np.linalg.norm(mesh_points[159] - mesh_points[145])
            h_dist = np.linalg.norm(mesh_points[33] - mesh_points[263])
            self.ear = v_dist / (h_dist + 1e-6)

            # LOGIC: CHỈ NHẮM MẮT KHI HAI ĐƯỜNG TRÙNG NHAU (Ngưỡng cực thấp)
            # Bình thường EAR khi mở mắt của bạn đang tầm 0.2 - 0.25. 
            # Chúng ta hạ xuống 0.12 để đảm bảo mí mắt phải sát nhau mới báo nhắm.
            if self.ear < 0.04: 
                self.last_blink = curr
                color = (0, 0, 255) # MÀU ĐỎ (Khi mí mắt đã trùng/gập vào nhau)
            else:
                color = (0, 255, 0) # MÀU XANH (Khi mí mắt còn khoảng cách - Mắt mở)

            # Vẽ khung theo dõi chuyển động
            cv2.polylines(img, [left_eye], True, color, 1, cv2.LINE_AA)
            cv2.polylines(img, [right_eye], True, color, 1, cv2.LINE_AA)
            
            # Hiển thị thông báo nếu quên chớp mắt quá 10 giây
            diff = int(curr - self.last_blink)
            if diff > 10:
                cv2.putText(img, "!!! HAY CHOP MAT !!!", (w//5, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
                cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 5)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-precision-v3",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

if ctx.video_processor:
    while True:
        curr_loop = time.time()
        sec_no_blink = int(curr_loop - ctx.video_processor.last_blink)
        blink_metric.metric("Thời gian chưa chớp mắt", f"{sec_no_blink}s")

        if sec_no_blink > 10:
            trigger_alert_js("Cảnh báo mỏi mắt", "Đã 10 giây bạn chưa chớp mắt!")
            time.sleep(5)
            
        time.sleep(1)
