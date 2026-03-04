🛡️ AI Eye Guard - Digital Wellness Monitor
AI Eye Guard là một ứng dụng web được xây dựng trên nền tảng Streamlit và MediaPipe, thiết kế để bảo vệ sức khỏe thị lực cho những người làm việc lâu với máy tính. Ứng dụng tích hợp trí tuệ nhân tạo để giám sát thói quen chớp mắt, khoảng cách ngồi và thời gian làm việc theo quy tắc khoa học.

I. Tính năng chính (Key Features)
- Giám sát chớp mắt (Blink Monitoring): Tự động phát hiện và nhắc nhở nếu người dùng không chớp mắt trong 10 giây để tránh khô mắt.
- Cảnh báo khoảng cách (Distance Alert): Sử dụng thuật toán đo khoảng cách mống mắt (Iris Distance) để cảnh báo khi người dùng ngồi quá sát màn hình (< 50cm).
- Quy tắc 20-20-20: Đồng hồ đếm ngược 20 phút để nhắc nhở người dùng nghỉ ngơi, nhìn xa 20 feet trong 20 giây.
- Thông báo đa tầng (Multi-layer Alerts): * Visual Alert: Hiệu ứng khung đỏ nhấp nháy trực tiếp trên màn hình camera.
- System Notification: Bắn thông báo đẩy ra góc màn hình (System Tray) ngay cả khi người dùng đang ở Tab trình duyệt khác.

2. Công nghệ sử dụng (Tech Stack)
- Ngôn ngữ: Python 3.11 (Hoạt động ổn định và phù hợp nhất)
- Thư viện AI: MediaPipe (Face Mesh), OpenCV.
- Framework: Streamlit.
- Xử lý Video: Streamlit-webrtc, Av.

3. Cài đặt (Installation)
- Clone repository:
git clone https://github.com/quynhanh1002/dean.git
- Cài đặt các thư viện cần thiết:
pip install -r requirements.txt
- Cấu hình hệ thống (Nếu chạy trên Linux/Streamlit Cloud):
  + Đảm bảo có tệp packages.txt để cài đặt các gói hệ thống:
- libgl1
- libglib2.0-0
3. Hướng dẫn sử dụng (Usage)
- Chạy ứng dụng: streamlit run app.py.
- Cho phép ứng dụng truy cập Camera và Thông báo (Notifications) trên trình duyệt.
- Nhấn nút Start để bắt đầu giám sát.
4. Cấu trúc dự án (Project Structure)
- app.py: Mã nguồn chính xử lý logic AI và giao diện.
- requirements.txt: Danh sách các thư viện Python cần thiết.
- packages.txt: Các thư viện hệ thống cho môi trường Linux.
- website streamlit: eaqenjdnbti93fh64jkczt.streamlit.app 
