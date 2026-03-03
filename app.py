import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import cv2
import mediapipe as mp
import time
import numpy as np
import av
import streamlit.components.v1 as components

st.set_page_config(page_title="AI Eye Guard - Full Features", layout="wide")

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

st.title("🛡️ AI Eye Guard - Bảo vệ mắt toàn diện")

col_vid, col_stats = st.columns([2, 1])

with col_stats:
    st.subheader("📊 Thông số Dashboard")
    blink_metric = st.empty()
    dist_status = st.empty()
    timer_metric = st.empty()
    st.divider()
    st.info("💡 BẢO VỆ ĐÔI MẮT CỦA BẠN")
    st.warning("Hãy nghỉ ngơi sau 20 phút nhìn màn hình bạn nhé!")

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
            
            # Tọa độ mí mắt để vẽ khung
            left_eye = mesh_points[[33, 160, 158, 133, 153, 144]]
            right_eye = mesh_points[[362, 385, 387, 263, 373, 380]]

            # 1. LOGIC CHỚP MẮT (EAR) - Siết tỉ lệ 0.09
            v_dist = np.linalg.norm(mesh_points[159] - mesh_points[145])
            h_dist = np.linalg.norm(mesh_points[33] - mesh_points[263])
            self.ear = v_dist / (h_dist + 1e-6)

            # Chỉ đỏ khi mí mắt trùng sát nhau (< 0.09)
            if self.ear < 0.09: 
                self.last_blink = curr
                color = (0, 0, 255) # ĐỎ
            else:
                color = (0, 255, 0) # XANH

            # Vẽ khung theo dõi mắt
            cv2.polylines(img, [left_eye], True, color, 1, cv2.LINE_AA)
            cv2.polylines(img, [right_eye], True, color, 1, cv2.LINE_AA)
            
            # 2. LOGIC KHOẢNG CÁCH (Ngồi gần)
            # 468 và 473 là tâm mống mắt
            iris_dist = np.linalg.norm(mesh_points[468] - mesh_points[473])
            self.too_close = iris_dist > 115 # Ngưỡng ngồi quá sát

            # Cảnh báo trực tiếp trên màn hình cam
            if self.too_close:
                cv2.putText(img, "BACK AWAY!", (w//4, h//2 + 50), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.5, (0, 0, 255), 3)

            # Cảnh báo quên chớp mắt 10s trên cam
            diff_blink = int(curr - self.last_blink)
            if diff_blink > 10:
                cv2.putText(img, "!!! HAY CHOP MAT !!!", (w//5, 80), 
                            cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 255), 2)
                cv2.rectangle(img, (0, 0), (w, h), (0, 0, 255), 8)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

with col_vid:
    ctx = webrtc_streamer(
        key="eye-guard-all-in-one",
        video_processor_factory=EyeProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )

# --- VÒNG LẶP ĐỒNG BỘ DASHBOARD & THÔNG BÁO ---
if ctx.video_processor:
    last_alert_time = 0
    while True:
        curr_loop = time.time()
        
        # Cập nhật số giây chưa chớp mắt
        sec_no_blink = int(curr_loop - ctx.video_processor.last_blink)
        blink_metric.metric("Chưa chớp mắt trong", f"{sec_no_blink}s")

        if sec_no_blink > 10 and (curr_loop - last_alert_time) > 5:
            trigger_alert_js("Mỏi mắt!", "Bạn đã không chớp mắt hơn 10 giây!")
            last_alert_time = curr_loop

        # Cập nhật trạng thái ngồi gần
        if ctx.video_processor.too_close:
            dist_status.error("🚫 CẢNH BÁO: NGỒI QUÁ GẦN")
            # Có thể thêm trigger_alert_js ở đây nếu muốn báo cả ngồi gần ra tab khác
        else:
            dist_status.success("✅ Khoảng cách an toàn")

        # Cập nhật đồng hồ 20 phút (Quy tắc 20-20-20)
        elapsed = int(curr_loop - ctx.video_processor.start_time)
        rem = max(0, (20 * 60) - elapsed)
        timer_metric.metric("Nghỉ ngơi sau", f"{rem//60:02d}:{rem%60:02d}")
        
        if rem <= 0:
            trigger_alert_js("Hết 20 phút!", "Hãy nhìn xa 20 feet trong 20 giây.")
            ctx.video_processor.start_time = time.time()

        time.sleep(0.5)
