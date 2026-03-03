import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="Hệ thống Bảo vệ Mắt", layout="wide")

# JavaScript: Gửi thông báo hệ thống và âm thanh
def trigger_alert_js(title, message):
    js_code = f"""
    <script>
    if (Notification.permission !== "granted") {{
        Notification.requestPermission();
    }}
    if (Notification.permission === "granted") {{
        new Notification("{title}", {{ body: "{message}" }});
    }}
    var audio = new Audio("https://www.soundjay.com/buttons/beep-01a.mp3");
    audio.play().catch(e => console.log("Yêu cầu click chuột để phát âm thanh"));
    </script>
    """
    components.html(js_code, height=0, width=0)

st.title("🛡️ AI Eye Guard - Fix Cảnh Báo 10s")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Trạng thái hệ thống")
    blink_counter = st.empty()  # Hiển thị số giây thực tế
    alert_status = st.empty()   # Hiển thị thông báo đỏ trên web
    st.divider()
    st.info("Lưu ý: Nếu số giây vượt quá 10 mà không báo, hãy kiểm tra quyền Thông báo (Notification) của trình duyệt.")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink_time = time.time()
        self.is_too_close = False
        self.ear = 1.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        curr = time.time()

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            
            # Tính chỉ số mở mắt (EAR)
            le_v = np.abs(res[159].y - res[145].y)
            le_h = np.abs(res[33].x - res[263].x)
            self.ear = le_v / (le_h + 1e-6)

            # NGƯỠNG NHẠY: Tăng lên 0.22 để dễ nhận diện chớp mắt hơn
            if self.ear < 0.22: 
                self.last_blink_time = curr

            # Khoảng cách ngồi
            dist = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.is_too_close = dist > 115

            # Vẽ chữ trực tiếp lên Video (Để chắc chắn logic đang chạy)
            diff = int(curr - self.last_blink_time)
            cv2.putText(img, f"No blink: {diff}s", (30, 50), 2, 0.8, (0, 255, 0), 2)
            if diff > 10:
                cv2.putText(img, "BLINK NOW!", (w//4, h//2), 2, 2, (0, 0, 255), 4)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="eye-fix-10s",
    video_processor_factory=EyeProcessor,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# --- VÒNG LẶP KIỂM TRA (Đảm bảo chạy liên tục) ---
if ctx.video_processor:
    while True:
        curr = time.time()
        # Ép lấy dữ liệu từ luồng xử lý camera
        seconds_without_blink = int(curr - ctx.video_processor.last_blink_time)
        
        # Hiển thị số giây lên Dashboard để bạn theo dõi
        blink_counter.metric("Thời gian chưa chớp mắt", f"{seconds_without_blink} giây")

        if seconds_without_blink > 10:
            alert_status.error("⚠️ CẢNH BÁO: BẠN QUÊN CHỚP MẮT!")
            trigger_alert_js("Cảnh báo mắt", f"Đã {seconds_without_blink} giây bạn chưa chớp mắt!")
            time.sleep(4) # Tạm dừng quét để không bắn quá nhiều thông báo
        else:
            alert_status.success("Mắt bạn đang ổn định")
        
        # Cảnh báo khoảng cách
        if ctx.video_processor.is_too_close:
            trigger_alert_js("Ngồi quá gần", "Hãy lùi xa màn hình!")
            time.sleep(4)

        time.sleep(0.5) # Quét cực nhanh (0.5 giây/lần)
