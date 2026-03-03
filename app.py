import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="Eye Wellness Pro AI", layout="wide")

# JavaScript: Gửi thông báo hệ thống và phát âm thanh
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
    audio.play();
    </script>
    """
    components.html(js_code, height=0, width=0)

st.title("🛡️ AI Eye Guard - Đã sửa lỗi Cảnh báo 10s")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Chỉ số trực tiếp")
    # Các ô hiển thị dữ liệu sẽ được cập nhật liên tục
    blink_ui = st.empty()  
    dist_ui = st.empty()
    timer_ui = st.empty()
    st.divider()
    st.info("Hướng dẫn: Sau khi nhấn Start, hãy Click chuột vào trang web 1 lần để kích hoạt âm thanh báo động.")

class EyeProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
        self.last_blink = time.time()
        self.is_too_close = False
        self.ear = 1.0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        curr = time.time()
        
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            
            # Tính EAR (Độ mở mắt)
            # Tăng độ nhạy: EAR < 0.22 là chớp mắt
            v_dist = np.abs(res[159].y - res[145].y)
            h_dist = np.abs(res[33].x - res[263].x)
            self.ear = v_dist / (h_dist + 1e-6)

            if self.ear < 0.22: 
                self.last_blink = curr

            # Đo khoảng cách (Iris)
            dist = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.is_too_close = dist > 115

            # Vẽ trực tiếp lên khung hình để kiểm tra logic
            diff = int(curr - self.last_blink)
            cv2.putText(img, f"No blink: {diff}s", (30, 50), 1, 2, (0, 255, 0), 2)
            if diff > 10:
                cv2.putText(img, "BLINK!", (w//3, h//2), 1, 4, (0, 0, 255), 5)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-guard-fix",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}, # Tắt Mic
        async_processing=True
    )

# --- VÒNG LẶP KIỂM TRA & CẬP NHẬT UI ---
# Đoạn này cực kỳ quan trọng để bắn thông báo JS
if ctx.video_processor:
    # Khởi tạo mốc thời gian nghỉ 20p
    start_session = time.time()
    last_alert_time = 0 

    while True:
        curr_loop = time.time()
        
        # 1. Lấy dữ liệu từ VideoProcessor
        last_blink = ctx.video_processor.last_blink
        diff_blink = int(curr_loop - last_blink)
        
        # 2. Cập nhật con số lên màn hình (Dashboard)
        blink_ui.metric("Thời gian chưa chớp mắt", f"{diff_blink} giây")
        
        # 3. CẢNH BÁO 10 GIÂY (Chống bắn thông báo liên tục gây treo web)
        if diff_blink > 10 and (curr_loop - last_alert_time) > 5:
            trigger_alert_js("Mỏi mắt quá!", f"Bạn đã không chớp mắt {diff_blink} giây rồi!")
            last_alert_time = curr_loop # Đợi 5s sau mới báo tiếp

        # 4. Cảnh báo khoảng cách
        if ctx.video_processor.is_too_close:
            dist_ui.error("🚫 NGỒI QUÁ SÁT MÀN HÌNH")
        else:
            dist_ui.success("✅ Khoảng cách an toàn")

        # 5. Đồng hồ 20 phút
        elapsed = int(curr_loop - start_session)
        rem = max(0, (20 * 60) - elapsed)
        timer_ui.metric("Nghỉ ngơi sau", f"{rem//60:02d}:{rem%60:02d}")
        
        if rem <= 0:
            trigger_alert_js("Đến giờ nghỉ!", "Hãy nhìn xa 20 feet trong 20 giây.")
            start_session = time.time()

        time.sleep(0.5) # Quét mỗi nửa giây
