import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av # Thư viện bắt buộc để xử lý frame trên web

st.set_page_config(page_title="Eye Wellness Monitor", page_icon="👁️")

# CSS tạo giao diện tối
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: white; }
    </style>
    """, unsafe_allow_html=True)

st.title("🛡️ Digital Eye Wellness Monitor")
st.write("Giải pháp bảo vệ mắt trực tuyến sử dụng AI.")

# Tham số cấu hình
BLINK_THRESH = 0.2
WORK_TIME_LIMIT = 20 * 60 # 20 phút

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        # Sửa lỗi AttributeError bằng cách khởi tạo tường minh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.last_blink_time = time.time()
        self.start_time = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)
        curr_time = time.time()

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            
            # Tính EAR (Eye Aspect Ratio) dựa trên tọa độ Y mí mắt
            le_v = np.abs(res[159].y - res[145].y)
            le_h = np.abs(res[33].x - res[263].x)
            ear = le_v / (le_h + 1e-6)

            # Cập nhật thời điểm mở mắt cuối cùng
            if ear > BLINK_THRESH:
                self.last_blink_time = curr_time
            
            # 1. Nhắc chớp mắt (Nhấp nháy góc trái)
            if (curr_time - self.last_blink_time) > 10:
                blink_alpha = (np.sin(curr_time * 7) + 1) / 2
                overlay = img.copy()
                cv2.putText(overlay, "!!! HAY CHOP MAT !!!", (30, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 165, 255), 3)
                img = cv2.addWeighted(overlay, blink_alpha, img, 1 - blink_alpha, 0)

            # 2. Nhắc nghỉ ngơi 20 phút (Nhấp nháy đỏ)
            if (curr_time - self.start_time) > WORK_TIME_LIMIT:
                red_alpha = (np.sin(curr_time * 4) + 1) / 4
                red_overlay = np.zeros_like(img)
                red_overlay[:] = (0, 0, 255)
                img = cv2.addWeighted(red_overlay, red_alpha, img, 1 - red_alpha, 0)
                cv2.putText(img, "NGHI MAT 20 GIAY!", (w//5, h//2), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)

        # Trả về frame dưới định dạng av.VideoFrame chuẩn
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Cấu hình WebRTC để kết nối ổn định hơn
webrtc_streamer(
    key="eye-wellness", 
    video_processor_factory=EyeProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={"video": True, "audio": False}
)
