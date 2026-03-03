import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

# Cấu hình giao diện Dashboard
st.set_page_config(page_title="AI Eye Guard - Monitor", layout="wide")

# JavaScript: Thông báo hệ thống và âm thanh cảnh báo
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

st.title("🛡️ AI Eye Guard - Hệ thống giám sát thị lực")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Chỉ số trực tiếp")
    blink_metric = st.empty()
    dist_status = st.empty()
    timer_metric = st.empty()
    st.divider()
    st.info("💡 Mẹo: Nếu mở mắt mà khung vẫn hiện MÀU ĐỎ, hãy đảm bảo ánh sáng đủ tốt hoặc lùi xa cam một chút.")

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
            # Chuyển đổi tọa độ mốc mặt sang pixel
            mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) 
                                   for p in results.multi_face_landmarks[0].landmark])
            
            # Lấy các điểm quanh mắt trái và phải để vẽ khung theo dõi
            left_eye = mesh_points[[33, 160, 158, 133, 153, 144]]
            right_eye = mesh_points[[362, 385, 387, 263, 373, 380]]

            # TÍNH TOÁN EAR (Eye Aspect Ratio)
            # 159 & 145 là mí trên/dưới; 33 & 263 là khóe mắt
            v_dist = np.linalg.norm(mesh_points[159] - mesh_points[145])
            h_dist = np.linalg.norm(mesh_points[33] - mesh_points[263])
            self.ear = v_dist / (h_dist + 1e-6)

            # NGƯỠNG NHẬN DIỆN (Đã hạ xuống 0.15 để tránh báo nhầm khi mắt mở)
            color = (0, 255, 0) # Màu xanh (Mắt đang mở)
            if self.ear < 0.15: 
                self.last_blink = curr
                color = (0, 0, 255) # Màu đỏ (Khi nhắm mắt thực sự)

            # Vẽ khung theo dõi chuyển động mắt
            cv2.polylines(img, [left_eye], True, color, 1, cv2.LINE_AA)
            cv2.polylines(img, [right_eye], True, color, 1, cv2.LINE_AA)
            
            # Vẽ tâm mống mắt (Iris)
            cv2.circle(img, tuple(mesh_points[468]), 2, (255, 255, 255), -1)
            cv2.circle(img, tuple(mesh_points[473]), 2, (255, 255, 255), -1)

            # CẢNH BÁO QUÊN CHỚP MẮT (Sau 10 giây)
            diff = int(curr - self.last_blink)
            if diff > 10:
                cv2.putText(img, "!!! HAY CHOP MAT !!!", (w//5, 100), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)
                cv2.rectangle(img, (0,0), (w,h), (0,0,255), 10) # Khung đỏ báo động

            # ĐO KHOẢNG CÁCH (IRIS DISTANCE)
            iris_dist = np.linalg.norm(mesh_points[468] - mesh_points[473])
            self.too_close = iris_dist > 115 # Ngưỡng ngồi quá gần

            if self.too_close:
                cv2.putText(img, "BACK AWAY!", (w//4, h//2), 2, 1.5, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-guard-final",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}, # Tắt Mic hoàn toàn
        async_processing=True
    )

# --- VÒNG LẶP CẬP NHẬT GIAO DIỆN & THÔNG BÁO ---
if ctx.video_processor:
    last_alert = 0
    while True:
        curr_loop = time.time()
        sec_no_blink = int(curr_loop - ctx.video_processor.last_blink)
        
        # Cập nhật số liệu sang cột Dashboard
        blink_metric.metric("Thời gian chưa chớp mắt", f"{sec_no_blink}s")

        if sec_no_blink > 10 and (curr_loop - last_alert) > 5:
            trigger_alert_js("Cảnh báo sức khỏe", "Bạn đã không chớp mắt quá 10 giây!")
            last_alert = curr_loop

        if ctx.video_processor.too_close:
            dist_status.error("🚫 CẢNH BÁO: NGỒI QUÁ GẦN")
        else:
            dist_status.success("✅ Khoảng cách an toàn")

        # Đồng hồ 20 phút
        elapsed = int(curr_loop - ctx.video_processor.start_time)
        rem = max(0, (20 * 60) - elapsed)
        timer_metric.metric("Kỳ nghỉ tiếp theo", f"{rem//60:02d}:{rem%60:02d}")
        
        if rem <= 0:
            trigger_alert_js("Nghỉ ngơi!", "Đã đến giờ áp dụng quy tắc 20-20-20.")
            ctx.video_processor.start_time = time.time()

        time.sleep(1)
