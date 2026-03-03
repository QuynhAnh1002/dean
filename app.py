import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

# Cấu hình giao diện
st.set_page_config(page_title="AI Eye Guard Pro", layout="wide")

# JavaScript: Phát âm thanh Beep và bắn Thông báo hệ thống (hiện ở góc màn hình)
def trigger_alert_js(title, message):
    js_code = f"""
    <script>
    // Xin quyền thông báo
    if (Notification.permission !== "granted") {{
        Notification.requestPermission();
    }}
    
    // Phát âm thanh báo động
    var audio = new Audio("https://www.soundjay.com/buttons/beep-01a.mp3");
    audio.play();

    // Hiển thị thông báo hệ thống (hiện ra ngay cả khi đang ở tab khác)
    if (Notification.permission === "granted") {{
        new Notification("{title}", {{ 
            body: "{message}", 
            icon: "https://cdn-icons-png.flaticon.com/512/564/564022.png" 
        }});
    }}
    </script>
    """
    components.html(js_code, height=0, width=0)

st.title("🛡️ AI Eye Guard - Bảo vệ mắt đa tầng")
st.write("Hệ thống giám sát chớp mắt, khoảng cách ngồi và thời gian làm việc.")

# Giao diện cột
col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Chỉ số thời gian thực")
    status_ui = st.empty()  # Trạng thái khoảng cách
    blink_ui = st.empty()   # Đếm giây chưa chớp mắt
    timer_ui = st.empty()   # Đếm ngược 20 phút
    st.divider()
    st.info("💡 Mẹo: Hãy nhấn 'Allow' khi trình duyệt hỏi quyền Thông báo để nhận cảnh báo khi đang làm việc ở web khác.")

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
        curr_time = time.time()
        
        results = self.face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        if results.multi_face_landmarks:
            res = results.multi_face_landmarks[0].landmark
            
            # 1. Tính toán EAR (Chỉ số mở mắt)
            # Mí trên: 159, Mí dưới: 145, Khóe mắt: 33, 263
            le_v = np.abs(res[159].y - res[145].y)
            le_h = np.abs(res[33].x - res[263].x)
            self.ear = le_v / (le_h + 1e-6)

            # Nếu nhắm mắt (EAR < 0.2), cập nhật thời gian chớp mắt cuối
            if self.ear < 0.2: 
                self.last_blink = curr_time

            # 2. Đo khoảng cách (Iris Distance)
            # 468 và 473 là tâm 2 mống mắt
            dist_pixel = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.too_close = dist_pixel > 115 # Ngưỡng ~50cm

            # Vẽ cảnh báo lên Video
            if self.too_close:
                cv2.putText(img, "BACK AWAY!", (w//4, h//2), 2, 1.5, (0,0,255), 3)
            if (curr_time - self.last_blink) > 10:
                cv2.putText(img, "BLINK NOW!", (50, 80), 2, 1.2, (0,165,255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    # TẮT MICRO (audio=False)
    ctx = webrtc_streamer(
        key="eye-guard-final",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False}, 
        async_processing=True
    )

# --- VÒNG LẶP CẬP NHẬT UI VÀ BẮN CẢNH BÁO ---
if ctx.video_processor:
    while True:
        curr = time.time()
        
        # 1. Kiểm tra 10 giây không chớp mắt
        seconds_no_blink = int(curr - ctx.video_processor.last_blink)
        blink_ui.metric("Chưa chớp mắt trong", f"{seconds_no_blink} giây")
        
        if seconds_no_blink > 10:
            trigger_alert_js("Cảnh báo mắt", "Bạn đã không chớp mắt hơn 10 giây! Hãy chớp mắt ngay.")
            time.sleep(3) # Tránh bắn thông báo quá liên tục

        # 2. Kiểm tra khoảng cách
        if ctx.video_processor.too_close:
            status_ui.error("🚫 BẠN ĐANG NGỒI QUÁ GẦN!")
            trigger_alert_js("Cảnh báo khoảng cách", "Vui lòng lùi xa màn hình ít nhất 50cm.")
            time.sleep(3)
        else:
            status_ui.success("✅ Khoảng cách ngồi an toàn")

        # 3. Đồng hồ 20 phút (Quy tắc 20-20-20)
        elapsed = int(curr - ctx.video_processor.start_time)
        rem = max(0, (20 * 60) - elapsed)
        mm, ss = divmod(rem, 60)
        timer_ui.metric("Nghỉ ngơi sau", f"{mm:02d}:{ss:02d}")
        
        if rem <= 0:
            trigger_alert_js("Đã đến giờ nghỉ!", "Hãy nhìn xa 20 feet trong 20 giây.")
            ctx.video_processor.start_time = time.time() # Reset
        
        time.sleep(1) # Quét mỗi giây 1 lần để tiết kiệm tài nguyên
