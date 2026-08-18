Dưới đây là giải pháp demo hoàn chỉnh chỉ trong **1 file mã nguồn duy nhất (`app.py`)**.

Mã nguồn được thiết kế tự động huấn luyện mô hình XGBoost trong bộ nhớ cache dựa trên ma trận thuộc tính của nghiên cứu FP-Inconsistent, hiển thị giao diện phân tích mâu thuẫn thời gian thực và trực quan hóa độ quan trọng của đặc trưng.

---

### Phần 1: Cài đặt môi trường (Chạy trong 1 phút)

Mở Terminal và chạy lệnh cài các thư viện cần thiết:

```bash
pip install streamlit pandas numpy xgboost scikit-learn matplotlib

```

---

### Phần 2: Mã nguồn Demo trọn gói (`app.py`)

Tạo file `app.py` và dán toàn bộ đoạn mã sau:

```python
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Cấu hình giao diện Streamlit
st.set_page_config(page_title="Anti-detect Browser Detection Demo", layout="wide", page_icon="🛡️")

# ==========================================
# 1. HÀM TẠO DỮ LIỆU & HUẤN LUYỆN MODEL (CACHE)
# ==========================================
@st.cache_resource
def init_and_train_model():
    np.random.seed(42)
    n_samples = 8000

    # Cấu hình chuẩn của người dùng thật
    real_configs = [
        {"ua_device": "iPhone", "ua_browser": "Mobile Safari", "platform": "iPhone", "vendor": "Apple Computer, Inc.", "max_touch": 5, "res": "1170x2532", "cpu": 6, "mem": 4, "country": "US", "tz": "America/New_York"},
        {"ua_device": "Desktop", "ua_browser": "Chrome", "platform": "Win32", "vendor": "Google Inc.", "max_touch": 0, "res": "1920x1080", "cpu": 8, "mem": 16, "country": "VN", "tz": "Asia/Ho_Chi_Minh"},
        {"ua_device": "Desktop", "ua_browser": "Safari", "platform": "MacIntel", "vendor": "Apple Computer, Inc.", "max_touch": 0, "res": "2560x1440", "cpu": 8, "mem": 8, "country": "US", "tz": "America/Los_Angeles"},
        {"ua_device": "Android", "ua_browser": "Chrome Mobile", "platform": "Linux armv8l", "vendor": "Google Inc.", "max_touch": 10, "res": "1080x2400", "cpu": 8, "mem": 6, "country": "FR", "tz": "Europe/Paris"},
    ]

    raw_data, labels = [], []
    for _ in range(n_samples):
        is_bot = np.random.choice([0, 1], p=[0.5, 0.5])
        cfg = real_configs[np.random.randint(0, len(real_configs))].copy()
        
        if is_bot == 1:
            leak_type = np.random.choice(["screen_mismatch", "touch_mismatch", "platform_mismatch", "geo_mismatch", "cpu_anomaly"])
            if leak_type == "screen_mismatch":
                cfg["ua_device"] = "iPhone"
                cfg["res"] = "1920x1080"
            elif leak_type == "touch_mismatch":
                cfg["ua_device"] = "iPhone"
                cfg["max_touch"] = np.random.choice([0, 1])
            elif leak_type == "platform_mismatch":
                cfg["ua_browser"] = "Mobile Safari"
                cfg["platform"] = "Linux x86_64"
            elif leak_type == "geo_mismatch":
                cfg["country"] = "FR"
                cfg["tz"] = "America/Los_Angeles"
            elif leak_type == "cpu_anomaly":
                cfg["ua_device"] = "iPhone"
                cfg["cpu"] = np.random.choice([24, 32, 48])
                
        raw_data.append(cfg)
        labels.append(is_bot)

    df = pd.DataFrame(raw_data)
    
    # Feature Engineering
    X = extract_features(df)
    y = np.array(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=60, max_depth=3, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)
    
    return model

def extract_features(df):
    valid_iphone_res = ["1170x2532", "1284x2778", "1179x2556", "1290x2796", "828x1792"]
    geo_map = {"US": ["America/New_York", "America/Los_Angeles", "America/Chicago"], "VN": ["Asia/Ho_Chi_Minh"], "FR": ["Europe/Paris"]}
    
    features = pd.DataFrame()
    features["inconsistent_screen"] = ((df["ua_device"] == "iPhone") & (~df["res"].isin(valid_iphone_res))).astype(int)
    features["inconsistent_touch"] = ((df["ua_device"] == "iPhone") & (df["max_touch"] != 5)).astype(int)
    features["inconsistent_platform"] = ((df["ua_browser"].str.contains("Safari", na=False)) & (df["platform"].str.contains("Linux|Win32", na=False))).astype(int)
    features["inconsistent_geo"] = [0 if tz in geo_map.get(country, []) else 1 for country, tz in zip(df["country"], df["tz"])]
    features["inconsistent_hardware"] = ((df["ua_device"].isin(["iPhone", "Android"])) & (df["cpu"] > 12)).astype(int)
    features["total_inconsistency_score"] = features.sum(axis=1)
    
    return features

# Huấn luyện mô hình ngay khi khởi động
model = init_and_train_model()

# ==========================================
# 2. GIAO DIỆN ĐIỀU KHIỂN & DEMO TRỰC TIẾP
# ==========================================
st.title("🛡️ Demo Phát Hiện Trình Duyệt Chống Phát Hiện (Anti-detect Browser)")
st.caption("Ứng dụng Học máy & Khai thác Mâu thuẫn Thuộc tính Vân tay (FP-Inconsistent Architecture)")

st.divider()

# Sidebar: Chọn kịch bản mẫu
st.sidebar.header("🕹️ Kịch bản kiểm thử nhanh")
preset = st.sidebar.radio(
    "Chọn một trường hợp:",
    [
        "1. Người dùng thật: iPhone Safari chuẩn",
        "2. Người dùng thật: PC Windows Chrome chuẩn",
        "3. Anti-detect: Fake iPhone nhưng màn hình 1920x1080",
        "4. Anti-detect: Fake iPhone nhưng maxTouchPoints = 0",
        "5. Anti-detect: Safari Mobile giả lập trên Linux",
        "6. Anti-detect: Dùng Proxy IP Pháp nhưng Timezone Mỹ",
        "7. Anti-detect: Giả lập Mobile nhưng lộ CPU 32 Cores",
    ]
)

# Ánh xạ giá trị theo Preset
if preset == "1. Người dùng thật: iPhone Safari chuẩn":
    p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz = "iPhone", "Mobile Safari", "iPhone", 5, "1170x2532", 6, "US", "America/New_York"
elif preset == "2. Người dùng thật: PC Windows Chrome chuẩn":
    p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz = "Desktop", "Chrome", "Win32", 0, "1920x1080", 8, "VN", "Asia/Ho_Chi_Minh"
elif preset == "3. Anti-detect: Fake iPhone nhưng màn hình 1920x1080":
    p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz = "iPhone", "Mobile Safari", "iPhone", 5, "1920x1080", 6, "US", "America/New_York"
elif preset == "4. Anti-detect: Fake iPhone nhưng maxTouchPoints = 0":
    p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz = "iPhone", "Mobile Safari", "iPhone", 0, "1170x2532", 6, "US", "America/New_York"
elif preset == "5. Anti-detect: Safari Mobile giả lập trên Linux":
    p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz = "iPhone", "Mobile Safari", "Linux x86_64", 5, "1170x2532", 6, "US", "America/New_York"
elif preset == "6. Anti-detect: Dùng Proxy IP Pháp nhưng Timezone Mỹ":
    p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz = "Desktop", "Chrome", "Win32", 0, "1920x1080", 8, "FR", "America/Los_Angeles"
else:
    p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz = "iPhone", "Mobile Safari", "iPhone", 5, "1170x2532", 32, "US", "America/New_York"

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("📥 Thuộc tính thu thập từ Client Request")
    
    c1, c2 = st.columns(2)
    with c1:
        ua_dev = st.selectbox("Device Type (UA)", ["iPhone", "Android", "Desktop"], index=["iPhone", "Android", "Desktop"].index(p_dev))
        ua_br = st.selectbox("Browser (UA)", ["Mobile Safari", "Safari", "Chrome", "Chrome Mobile"], index=["Mobile Safari", "Safari", "Chrome", "Chrome Mobile"].index(p_br))
        plat = st.selectbox("Navigator Platform", ["iPhone", "MacIntel", "Win32", "Linux armv8l", "Linux x86_64"], index=["iPhone", "MacIntel", "Win32", "Linux armv8l", "Linux x86_64"].index(p_plat))
        touch = st.number_input("maxTouchPoints API", min_value=0, max_value=20, value=p_touch)
        
    with c2:
        res = st.text_input("Screen Resolution", value=p_res)
        cpu = st.number_input("Hardware Concurrency (CPU Cores)", min_value=1, max_value=64, value=p_cpu)
        country = st.selectbox("IP Geolocation Country", ["US", "VN", "FR"], index=["US", "VN", "FR"].index(p_country))
        tz = st.text_input("Timezone Offset / Identifier", value=p_tz)

    input_df = pd.DataFrame([{
        "ua_device": ua_dev, "ua_browser": ua_br, "platform": plat,
        "max_touch": touch, "res": res, "cpu": cpu, "country": country, "tz": tz
    }])
    
    features_df = extract_features(input_df)

with col_right:
    st.subheader("⚙️ Kết quả Phân tích Mâu thuẫn (Feature Matrix)")
    
    f1, f2 = st.columns(2)
    f1.metric("Screen Mismatch", "❌ Phát hiện" if features_df.loc[0, 'inconsistent_screen'] == 1 else "✅ Hợp lệ")
    f1.metric("Touch Mismatch", "❌ Phát hiện" if features_df.loc[0, 'inconsistent_touch'] == 1 else "✅ Hợp lệ")
    f1.metric("Platform Mismatch", "❌ Phát hiện" if features_df.loc[0, 'inconsistent_platform'] == 1 else "✅ Hợp lệ")
    
    f2.metric("Geo/Timezone Mismatch", "❌ Phát hiện" if features_df.loc[0, 'inconsistent_geo'] == 1 else "✅ Hợp lệ")
    f2.metric("Hardware Anomaly", "❌ Phát hiện" if features_df.loc[0, 'inconsistent_hardware'] == 1 else "✅ Hợp lệ")
    f2.metric("Tổng điểm mâu thuẫn", f"{features_df.loc[0, 'total_inconsistency_score']} / 5")

st.divider()

# ==========================================
# 3. KẾT LUẬN TỪ MÔ HÌNH HỌC MÁY
# ==========================================
pred_prob = model.predict_proba(features_df)[0]
is_bot_pred = model.predict(features_df)[0]

res_col1, res_col2 = st.columns([1.2, 1], gap="large")

with res_col1:
    st.subheader("🎯 Đánh giá từ Mô hình Học máy (XGBoost Classifier)")
    if is_bot_pred == 1:
        st.error(f"🚨 **KẾT LUẬN: PHÁT HIỆN ANTI-DETECT BROWSER / BOT**")
        st.write(f"Độ tin cậy Bot: **{pred_prob[1]*100:.2f}%**")
        st.progress(float(pred_prob[1]))
    else:
        st.success(f"✅ **KẾT LUẬN: NGƯỜI DÙNG HỢP LỆ (LEGITIMATE USER)**")
        st.write(f"Độ tin cậy Real User: **{pred_prob[0]*100:.2f}%**")
        st.progress(float(pred_prob[0]))

with res_col2:
    st.subheader("📊 Trọng số Đặc trưng (Global Feature Importance)")
    feat_importances = pd.Series(model.feature_importances_, index=features_df.columns)
    fig, ax = plt.subplots(figsize=(6, 3))
    feat_importances.sort_values().plot(kind='barh', color='#ff4b4b' if is_bot_pred == 1 else '#0083B8', ax=ax)
    ax.set_xlabel("F-Score")
    st.pyplot(fig)

```

Chạy file demo bằng lệnh:

```bash
streamlit run app.py

```

---

### Phần 3: Kịch bản trình diễn trực tiếp (Live Demo Flow)

Khi thực hiện demo, hãy thực hiện theo thứ tự 4 bước sau để tạo sự thuyết phục:

* **Bước 1: Trình diễn trường hợp Người dùng thật (Baseline)**
* *Thao tác:* Chọn kịch bản **1** hoặc **2** ở thanh bên trái.
* *Hiện tượng:* Tất cả các metric báo `✅ Hợp lệ`, điểm mâu thuẫn là $0/5$, mô hình đưa ra kết luận `✅ NGƯỜI DÙNG HỢP LỆ` với độ tin cậy $>99\%$.
* *Lời giải thích:* Thể hiện mô hình không báo động giả (giữ tỷ lệ TNR cao) đối với các cấu hình phần cứng chuẩn.


* **Bước 2: Trình diễn giả mạo phần cứng / màn hình (Spatial Screen Inconsistency)**
* 
*Thao tác:* Chọn kịch bản **3** (Fake iPhone nhưng màn hình 1920x1080).


* *Hiện tượng:* `Screen Mismatch` chuyển sang màu đỏ `❌ Phát hiện`, mô hình lập tức nhận diện `🚨 ANTI-DETECT BOT`.
* 
*Lời giải thích:* Các công cụ anti-detect đổi User-Agent thành iOS nhưng không đổi kích thước canvas/viewport hoặc dùng sai độ phân giải vật lý của Apple.




* **Bước 3: Trình diễn giả lập hệ điều hành lỗi (OS / Platform Inconsistency)**
* 
*Thao tác:* Chọn kịch bản **5** (Safari Mobile trên Linux x86_64).


* *Hiện tượng:* `Platform Mismatch` bị kích hoạt cảnh báo, hệ thống phát hiện hành vi can thiệp DOM/API.
* 
*Lời giải thích:* Bot chạy trên server Linux hoặc container Docker nhưng gửi User-Agent là Safari của Apple, làm lộ mâu thuẫn giữa `navigator.userAgent` và `navigator.platform`.




* **Bước 4: Trình diễn mâu thuẫn Proxy & Múi giờ (Geo-Timezone Mismatch)**
* 
*Thao tác:* Chọn kịch bản **6** (IP Pháp nhưng Timezone Mỹ).


* *Hiện tượng:* `Geo/Timezone Mismatch` bị kích hoạt báo động.
* 
*Lời giải thích:* Bot sử dụng Proxy/SOCKS IP nước ngoài để vượt Geo-blocking nhưng quên đồng bộ múi giờ qua `getTimezoneOffset()` của trình duyệt.