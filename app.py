import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="Hệ thống Bảo vệ Mắt AI", layout="wide")

# JavaScript: Thông báo đẩy + Phát âm thanh (Yêu cầu quyền Notifications)
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
    audio.play().catch(e => console.log("Cần click chuột để phát thanh"));
    </script>
    """
    components.html(js_code, height=0, width=0)

st.title("🛡️ AI Eye Guard - Cảnh Báo Đa Tầng")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Giám sát chỉ số")
    blink_metric = st.empty()  # Hiện số giây chưa chớp mắt
    dist_status = st.empty()   # Hiện trạng thái ngồi gần/xa
    timer_metric = st.empty()  # Đếm ngược 20 phút
    st.divider()
    st.info("💡 Mẹo: Nếu số giây > 10 mà chưa báo, hãy Click vào web 1 lần để trình duyệt cho phép chạy thông báo.")

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
            res = results.multi_face_landmarks[0].landmark
            
            # 1. Logic Chớp mắt (Tăng EAR lên 0.23 để nhạy hơn)
            v_dist = np.abs(res[159].y - res[145].y)
            h_dist = np.abs(res[33].x - res[263].x)
            self.ear = v_dist / (h_dist + 1e-6)

            if self.ear < 0.23: 
                self.last_blink = curr

            # 2. Logic Khoảng cách
            dist_iris = np.sqrt((res[468].x - res[473].x)**2 + (res[468].y - res[473].y)**2) * w
            self.too_close = dist_iris > 115 

            # --- CẢNH BÁO TRỰC TIẾP TRÊN MÀN HÌNH CAMERA ---
            diff_blink = int(curr - self.last_blink)
            
            # Cảnh báo chớp mắt (Chữ nhấp nháy đỏ rực giữa màn hình)
            if diff_blink > 10:
                # Hiệu ứng nhấp nháy theo thời gian
                if int(curr * 4) % 2 == 0: 
                    cv2.putText(img, "!!! BLINK NOW !!!", (w//4, h//2), 
                                cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255), 5)
                # Vẽ khung đỏ bao quanh màn hình
                cv2.rectangle(img, (0,0), (w,h), (0,0,255), 20)

            # Cảnh báo lùi xa
            if self.too_close:
                cv2.putText(img, "TOO CLOSE!", (30, h-50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-guard-final-v2",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- VÒNG LẶP ĐỒNG BỘ GIAO DIỆN & THÔNG BÁO ---
if ctx.video_processor:
    last_alert = 0
    while True:
        curr_loop = time.time()
        
        # Lấy dữ liệu từ Camera ra Dashboard
        sec_no_blink = int(curr_loop - ctx.video_processor.last_blink)
        blink_metric.metric("Chưa chớp mắt trong", f"{sec_no_blink} giây")

        # Bắn thông báo JavaScript (Mỗi 5 giây báo 1 lần để tránh treo)
        if sec_no_blink > 10 and (curr_loop - last_alert) > 5:
            trigger_alert_js("Cảnh báo mỏi mắt!", f"Bạn đã không chớp mắt {sec_no_blink} giây rồi!")
            last_alert = curr_loop

        # Cập nhật trạng thái khoảng cách
        if ctx.video_processor.too_close:
            dist_status.error("🚫 NGỒI QUÁ SÁT MÀN HÌNH")
        else:
            dist_status.success("✅ Khoảng cách an toàn")

        # Đồng hồ 20 phút
        elapsed = int(curr_loop - ctx.video_processor.start_time)
        remaining = max(0, (20 * 60) - elapsed)
        timer_metric.metric("Nghỉ ngơi sau", f"{remaining//60:02d}:{remaining%60:02d}")
        
        if remaining <= 0:
            trigger_alert_js("Hết 20 phút!", "Hãy nhìn xa 20 feet trong 20 giây.")
            ctx.video_processor.start_time = time.time()

        time.sleep(0.5)
