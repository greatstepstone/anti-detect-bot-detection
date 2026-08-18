# 🛡️ Anti-detect Browser & Bot Detection using Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Hệ thống ứng dụng Học máy (XGBoost) và Khai thác Mâu thuẫn Thuộc tính Vân tay (FP-Inconsistent Architecture) để phát hiện các trình duyệt chống phát hiện (Anti-detect Browsers) và bot tự động theo thời gian thực.

---

## 📌 Cơ sở Lý thuyết & Tài liệu Tham khảo

Hệ thống được phát triển dựa trên các nghiên cứu khoa học công bố tại các hội nghị bảo mật và đo lường hàng đầu:
* **FP-Inconsistent** (*ACM IMC 2025*): Đo lường và phát hiện sự không nhất quán không gian và thời gian trong lưu lượng bot né tránh.
* **Browser Polygraph** (*ACM IMC 2024*): Ứng dụng dấu vân tay thô (CGFP) để kiểm tra chéo tính tương thích giữa User-Agent và API hệ thống.
* **Him of Many Faces** (*NDSS 2023*): Nghiên cứu 36 tỷ request và phân loại các chiến lược tạo vân tay đối nghịch (*Keep*, *Block*, *Mimic*, *Randomize*).
* **Good Bot, Bad Bot** (*IEEE S&P 2021*): Khảo sát hành vi bot trên 100 honeysites thực tế.

---

## 🌟 Tính năng Nổi bật

1. **📡 Màn hình Giám sát Trực tiếp (Live Interactive Dashboard)**:
   - Tự động thu thập vân tay phía Client (User-Agent, Platform, Touch Points, Screen, WebGL GPU, CPU Cores, Múi giờ) qua JavaScript ngầm.
   - Hiển thị mã QR động để người tham gia quét mã và gửi dữ liệu kiểm thử trực tiếp từ điện thoại / laptop.
   - Bảng Live Feed cập nhật tức thì kết quả phân loại: `✅ Người dùng thật` hoặc `🚨 Phát hiện Bot / Anti-detect`.
2. **🕹️ Giả lập Kịch bản Mẫu (Manual Preset Simulator)**:
   - 8 kịch bản giả lập đại diện cho các lỗi mâu thuẫn điển hình (Screen Mismatch, Touch Points = 0, Linux Server fake iPhone, Proxy IP lệch múi giờ, WebGL Direct3D GPU trên iOS/Safari).
3. **📊 Trực quan hóa Trọng số Đặc trưng (Feature Importance)**:
   - Hiển thị mức độ đóng góp của từng thuộc tính mâu thuẫn vào quyết định của mô hình XGBoost.

---

## 🚀 Cài đặt & Chạy Cục bộ (Local Run)

```bash
# 1. Clone repository
git clone https://github.com/greatstepstone/anti-detect-bot-detection.git
cd anti-detect-bot-detection

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Chạy ứng dụng Streamlit
streamlit run app.py
```

Ứng dụng sẽ tự động mở tại `http://localhost:8501`.

---

## ☁️ Triển khai lên Streamlit Community Cloud

1. Đăng nhập vào [share.streamlit.io](https://share.streamlit.io) bằng tài khoản GitHub.
2. Chọn repository `greatstepstone/anti-detect-bot-detection`.
3. Nhánh chính (`main`) và file khởi chạy (`app.py`).
4. Bấm **Deploy** để nhận URL trực tiếp có HTTPS.
