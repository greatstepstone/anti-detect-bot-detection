import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import datetime
import urllib.parse
import uuid

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Anti-detect Browser Detection AI",
    layout="wide",
    page_icon="🛡️"
)

# ==========================================
# 0. HỖ TRỢ BỘ NHỚ LỊCH SỬ (CACHE PROCESS-WIDE)
# ==========================================
@st.cache_resource
def get_live_history():
    """Lưu trữ danh sách các thiết bị truy cập thực nghiệm (Thread-safe singleton)."""
    return []

def get_client_info():
    """Đọc thông tin header thực tế từ trình duyệt của người truy cập."""
    ua = "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148 Safari/604.1"
    platform_hint = ""
    try:
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            headers = st.context.headers
            if headers:
                ua = headers.get("User-Agent", headers.get("user-agent", ua))
                platform_hint = headers.get("Sec-Ch-Ua-Platform", headers.get("sec-ch-ua-platform", ""))
    except Exception:
        pass
    return ua, platform_hint

# ==========================================
# 1. HÀM TẠO DỮ LIỆU & HUẤN LUYỆN MODEL (CACHE)
# ==========================================
@st.cache_resource
def init_and_train_model():
    np.random.seed(42)
    n_samples = 10000

    real_configs = [
        {
            "ua_device": "iPhone", "ua_browser": "Mobile Safari", "platform": "iPhone",
            "vendor": "Apple Computer, Inc.", "max_touch": 5, "res": "1170x2532",
            "cpu": 6, "mem": 4, "country": "US", "tz": "America/New_York",
            "webgl_renderer": "Apple GPU"
        },
        {
            "ua_device": "Desktop", "ua_browser": "Chrome", "platform": "Win32",
            "vendor": "Google Inc.", "max_touch": 0, "res": "1920x1080",
            "cpu": 8, "mem": 16, "country": "VN", "tz": "Asia/Ho_Chi_Minh",
            "webgl_renderer": "ANGLE (NVIDIA, NVIDIA GeForce RTX 3060 Direct3D11 vs_5_0 ps_5_0)"
        },
        {
            "ua_device": "Desktop", "ua_browser": "Safari", "platform": "MacIntel",
            "vendor": "Apple Computer, Inc.", "max_touch": 0, "res": "2560x1440",
            "cpu": 8, "mem": 8, "country": "US", "tz": "America/Los_Angeles",
            "webgl_renderer": "Apple M1"
        },
        {
            "ua_device": "Android", "ua_browser": "Chrome Mobile", "platform": "Linux armv8l",
            "vendor": "Google Inc.", "max_touch": 10, "res": "1080x2400",
            "cpu": 8, "mem": 6, "country": "FR", "tz": "Europe/Paris",
            "webgl_renderer": "Adreno (TM) 660"
        },
    ]

    raw_data, labels = [], []
    for _ in range(n_samples):
        is_bot = np.random.choice([0, 1], p=[0.5, 0.5])
        cfg = real_configs[np.random.randint(0, len(real_configs))].copy()

        if is_bot == 1:
            leak_type = np.random.choice([
                "screen_mismatch", "touch_mismatch", "platform_mismatch",
                "geo_mismatch", "cpu_anomaly", "webgl_mismatch"
            ])
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
            elif leak_type == "webgl_mismatch":
                cfg["ua_device"] = "iPhone"
                cfg["ua_browser"] = "Mobile Safari"
                cfg["webgl_renderer"] = "ANGLE (Intel, Intel(R) UHD Graphics Direct3D11)"

        raw_data.append(cfg)
        labels.append(is_bot)

    df = pd.DataFrame(raw_data)
    X = extract_features(df)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(n_estimators=70, max_depth=3, random_state=42, eval_metric="logloss")
    model.fit(X_train, y_train)

    return model

def parse_user_agent(ua_str):
    """Phân tích sơ bộ User-Agent từ client."""
    ua_str_lower = ua_str.lower() if isinstance(ua_str, str) else ""
    if "iphone" in ua_str_lower or "ipad" in ua_str_lower:
        dev = "iPhone"
        br = "Mobile Safari" if "safari" in ua_str_lower else "Chrome Mobile"
        plat = "iPhone"
        touch = 5
        res = "1170x2532"
        gpu = "Apple GPU"
    elif "android" in ua_str_lower:
        dev = "Android"
        br = "Chrome Mobile"
        plat = "Linux armv8l"
        touch = 10
        res = "1080x2400"
        gpu = "Adreno (TM) GPU"
    elif "macintosh" in ua_str_lower or "mac os" in ua_str_lower:
        dev = "Desktop"
        br = "Safari" if "safari" in ua_str_lower and "chrome" not in ua_str_lower else "Chrome"
        plat = "MacIntel"
        touch = 0
        res = "2560x1440"
        gpu = "Apple M-Series GPU"
    else:
        dev = "Desktop"
        br = "Chrome" if "chrome" in ua_str_lower else "Edge"
        plat = "Win32"
        touch = 0
        res = "1920x1080"
        gpu = "ANGLE (Direct3D11 / Dedicated GPU)"
    return dev, br, plat, touch, res, gpu

def extract_features(df):
    """Trích xuất ma trận mâu thuẫn không gian (Spatial Inconsistencies)."""
    valid_iphone_res = [
        "1170x2532", "1284x2778", "1179x2556", "1290x2796", "828x1792",
        "390x844", "428x926", "393x852", "430x932", "414x896", "375x812", "375x667"
    ]
    geo_map = {
        "US": ["America/New_York", "America/Los_Angeles", "America/Chicago", "America/Denver"],
        "VN": ["Asia/Ho_Chi_Minh", "Asia/Bangkok", "Asia/Saigon"],
        "FR": ["Europe/Paris"]
    }

    features = pd.DataFrame()

    # 1. Screen Resolution Inconsistency
    features["inconsistent_screen"] = [
        1 if (row.get("ua_device") == "iPhone" and str(row.get("res")).lower() not in [r.lower() for r in valid_iphone_res])
        else 0 for _, row in df.iterrows()
    ]

    # 2. Touch API Inconsistency
    features["inconsistent_touch"] = [
        1 if (row.get("ua_device") in ["iPhone", "Android"] and int(row.get("max_touch", 0)) < 2)
        else 0 for _, row in df.iterrows()
    ]

    # 3. Platform Inconsistency
    features["inconsistent_platform"] = [
        1 if ("safari" in str(row.get("ua_browser", "")).lower() and any(p in str(row.get("platform", "")).lower() for p in ["linux", "win32", "windows"]))
        else 0 for _, row in df.iterrows()
    ]

    # 4. Geo / Timezone Inconsistency
    features["inconsistent_geo"] = [
        0 if str(row.get("tz", "")) in geo_map.get(str(row.get("country", "")), [])
        or str(row.get("country", "")) == "Auto/Client"
        or "Asia/Ho_Chi_Minh" in str(row.get("tz", "")) or "Asia/Bangkok" in str(row.get("tz", ""))
        else 1 for _, row in df.iterrows()
    ]

    # 5. Hardware Anomaly (CPU Cores vs Mobile)
    features["inconsistent_hardware"] = [
        1 if (row.get("ua_device") in ["iPhone", "Android"] and int(row.get("cpu", 4)) > 12)
        else 0 for _, row in df.iterrows()
    ]

    # 6. WebGL GPU Mismatch
    features["inconsistent_webgl"] = [
        1 if (row.get("ua_device") == "iPhone" and any(k in str(row.get("webgl_renderer", "")).lower() for k in ["direct3d", "nvidia", "intel", "amd"]))
        else 0 for _, row in df.iterrows()
    ]

    features["total_inconsistency_score"] = features.sum(axis=1)
    return features

# Khởi tạo mô hình
model = init_and_train_model()
default_public_url = "https://anti-detect-bot-detection-hodmmdswgfmb9brpacxdos.streamlit.app"

# ==========================================
# 2. GIAO DIỆN CHÍNH & TABS
# ==========================================
st.markdown("""
<style>
    .main-title { font-size: 2.1rem; font-weight: 800; color: #1E293B; margin-bottom: 0px; }
    .sub-title { font-size: 0.95rem; color: #64748B; margin-bottom: 15px; }
    .badge-bot { background-color: #FEE2E2; color: #DC2626; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
    .badge-real { background-color: #DCFCE7; color: #16A34A; padding: 4px 10px; border-radius: 6px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🛡️ Hệ thống Phát hiện Anti-detect Browser & Bot (Live Interactive AI)</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Nghiên cứu ứng dụng Học máy & Khai thác Mâu thuẫn Thuộc tính Vân tay (FP-Inconsistent Architecture - ACM IMC 2025 / NDSS 2023)</p>', unsafe_allow_html=True)

tab_live, tab_manual, tab_docs = st.tabs([
    "📡 Live Verification & Dashboard (Màn hình Giám sát Trực tiếp)",
    "🕹️ Manual Preset Simulator (Kịch bản Giả lập Thủ công)",
    "📚 Research Background & Metrics (Cơ sở Khoa học)"
])

# ==========================================
# TAB 1: LIVE VERIFICATION & DASHBOARD
# ==========================================
with tab_live:
    # 1. Đọc thông số thiết bị người dùng đang truy cập
    client_ua, client_hint = get_client_info()
    det_dev, det_br, det_plat, det_touch, det_res, det_gpu = parse_user_agent(client_ua)

    # 2. Bảng Tương tác dành cho Người quét QR bằng Điện thoại/Laptop
    st.markdown("### 📱 Cổng Xác minh Thiết bị Trực tiếp (Live Client Portal)")
    st.caption("Dành cho người tham gia quét mã QR bằng điện thoại hoặc mở trên laptop để thử nghiệm trực tiếp.")

    c_box1, c_box2 = st.columns([1.2, 1], gap="medium")

    with c_box1:
        st.markdown(f"**🔍 Thiết bị của bạn được nhận diện tự động:**")
        st.info(f"📍 **Thiết bị**: `{det_dev} ({det_br})` | **Hệ điều hành**: `{det_plat}` | **Màn hình**: `{det_res}` | **Touch**: `{det_touch}`")

        col_act1, col_act2 = st.columns(2)
        with col_act1:
            if st.button("✅ Gửi Vân tay: TÔI LÀ NGƯỜI THẬT", type="primary", use_container_width=True):
                client_row = {
                    "ua_device": det_dev, "ua_browser": det_br, "platform": det_plat,
                    "max_touch": det_touch, "res": det_res, "cpu": 6 if det_dev == "iPhone" else 8,
                    "country": "VN", "tz": "Asia/Ho_Chi_Minh", "webgl_renderer": det_gpu
                }
                eval_df = pd.DataFrame([client_row])
                feat_matrix = extract_features(eval_df)
                pred_is_bot = int(model.predict(feat_matrix)[0])
                pred_prob = model.predict_proba(feat_matrix)[0]

                get_live_history().insert(0, {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "sid": f"real_{uuid.uuid4().hex[:6]}",
                    "device": f"{det_dev} ({det_br})",
                    "platform": det_plat,
                    "res": det_res,
                    "touch": det_touch,
                    "is_bot": pred_is_bot,
                    "bot_confidence": f"{pred_prob[1]*100:.1f}%" if pred_is_bot == 1 else f"{pred_prob[0]*100:.1f}%",
                    "violations": "Không có (Hợp lệ hoàn toàn)",
                    "raw_gpu": det_gpu
                })
                st.success("🎉 Đã gửi xác minh thành công! Màn hình máy chiếu của lớp đã ghi nhận thiết bị của bạn.")
                st.rerun()

        with col_act2:
            attack_scenario = st.selectbox(
                "Hoặc thử vào vai Bot ngụy trang:",
                [
                    "Fake iPhone nhưng màn hình 1920x1080 (Screen Mismatch)",
                    "Fake iPhone nhưng Touch Points = 0 (Touch API Mismatch)",
                    "Safari Mobile giả lập trên Linux Server (Platform Mismatch)",
                    "Dùng Proxy IP Pháp nhưng Timezone Mỹ (Geo Mismatch)",
                    "Giả lập Mobile nhưng lộ CPU 32 Cores (Hardware Anomaly)",
                    "Fake iPhone Safari nhưng lộ GPU Windows Direct3D (WebGL Mismatch)"
                ]
            )
            if st.button("🤖 Gửi Vân tay: TÔI LÀ BOT NGỤY TRANG", use_container_width=True):
                # Thiết lập cấu hình bot mâu thuẫn
                if "1920x1080" in attack_scenario:
                    b_row = {"ua_device": "iPhone", "ua_browser": "Mobile Safari", "platform": "iPhone", "max_touch": 5, "res": "1920x1080", "cpu": 6, "country": "US", "tz": "America/New_York", "webgl_renderer": "Apple GPU", "err": "Màn hình 1920x1080 không tồn tại trên iPhone"}
                elif "Touch Points = 0" in attack_scenario:
                    b_row = {"ua_device": "iPhone", "ua_browser": "Mobile Safari", "platform": "iPhone", "max_touch": 0, "res": "1170x2532", "cpu": 6, "country": "US", "tz": "America/New_York", "webgl_renderer": "Apple GPU", "err": "iOS nhưng maxTouchPoints = 0"}
                elif "Linux Server" in attack_scenario:
                    b_row = {"ua_device": "iPhone", "ua_browser": "Mobile Safari", "platform": "Linux x86_64", "max_touch": 5, "res": "1170x2532", "cpu": 6, "country": "US", "tz": "America/New_York", "webgl_renderer": "Apple GPU", "err": "Safari Mobile lộ Platform Linux"}
                elif "Proxy IP Pháp" in attack_scenario:
                    b_row = {"ua_device": "Desktop", "ua_browser": "Chrome", "platform": "Win32", "max_touch": 0, "res": "1920x1080", "cpu": 8, "country": "FR", "tz": "America/Los_Angeles", "webgl_renderer": "ANGLE (NVIDIA)", "err": "IP Pháp nhưng Timezone Los Angeles"}
                elif "CPU 32 Cores" in attack_scenario:
                    b_row = {"ua_device": "iPhone", "ua_browser": "Mobile Safari", "platform": "iPhone", "max_touch": 5, "res": "1170x2532", "cpu": 32, "country": "US", "tz": "America/New_York", "webgl_renderer": "Apple GPU", "err": "Mobile có 32 CPU Cores"}
                else:
                    b_row = {"ua_device": "iPhone", "ua_browser": "Mobile Safari", "platform": "iPhone", "max_touch": 5, "res": "1170x2532", "cpu": 6, "country": "US", "tz": "America/New_York", "webgl_renderer": "ANGLE (Intel Direct3D11)", "err": "Safari iOS lộ GPU Direct3D của Windows"}

                eval_df = pd.DataFrame([b_row])
                feat_matrix = extract_features(eval_df)
                pred_is_bot = int(model.predict(feat_matrix)[0])
                pred_prob = model.predict_proba(feat_matrix)[0]

                get_live_history().insert(0, {
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "sid": f"bot_{uuid.uuid4().hex[:6]}",
                    "device": f"{b_row['ua_device']} ({b_row['ua_browser']})",
                    "platform": b_row['platform'],
                    "res": b_row['res'],
                    "touch": b_row['max_touch'],
                    "is_bot": pred_is_bot,
                    "bot_confidence": f"{pred_prob[1]*100:.1f}%" if pred_is_bot == 1 else f"{pred_prob[0]*100:.1f}%",
                    "violations": b_row['err'],
                    "raw_gpu": b_row['webgl_renderer']
                })
                st.error("🚨 Đã gửi gói tin tấn công thử nghiệm! Màn hình máy chiếu của lớp đã báo động phát hiện Anti-detect Bot.")
                st.rerun()

    with c_box2:
        st.markdown("**📲 Quét mã để kết nối điện thoại vào Demo:**")
        qr_img_url = f"https://api.qrserver.com/v1/create-qr-code/?size=180x180&data={urllib.parse.quote(default_public_url)}"
        st.image(qr_img_url, width=170)
        st.caption(f"Hoặc truy cập: `{default_public_url}`")

    st.divider()

    # 3. Bảng Giám sát & Metrics Thời gian thực
    st.markdown("### 📊 Màn hình Giám sát Máy chiếu (Live Class Dashboard)")
    history = get_live_history()
    total_req = len(history)
    real_users = sum(1 for h in history if h.get("is_bot") == 0)
    bot_users = sum(1 for h in history if h.get("is_bot") == 1)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Tổng lượt kiểm thử", total_req)
    m2.metric("Người thật (Real User)", real_users)
    m3.metric("Phát hiện Bot / Anti-detect", bot_users)
    tnr_pct = f"{(real_users / max(1, total_req))*100:.1f}%" if total_req > 0 else "100%"
    m4.metric("Tỷ lệ Real User", tnr_pct)

    c_tb1, c_tb2 = st.columns([3, 1])
    with c_tb1:
        st.subheader("📋 Bảng Lịch sử Truy cập Trực tiếp (Live Activity Log)")
    with c_tb2:
        c_r1, c_r2 = st.columns(2)
        with c_r1:
            if st.button("🔄 Làm mới", use_container_width=True):
                st.rerun()
        with c_r2:
            if st.button("🗑️ Xóa Log", use_container_width=True):
                get_live_history().clear()
                st.rerun()

    if history:
        display_data = []
        for h in history:
            badge = "🚨 BOT / ANTI-DETECT" if h.get("is_bot") == 1 else "✅ NGƯỜI THẬT"
            display_data.append({
                "Thời gian": h.get("time"),
                "Mã phiên": h.get("sid", "N/A"),
                "Thiết bị nhận diện": h.get("device"),
                "Hệ điều hành": h.get("platform"),
                "Màn hình": h.get("res"),
                "Touch Points": h.get("touch"),
                "Kết luận AI": badge,
                "Độ tin cậy": h.get("bot_confidence"),
                "Lỗi mâu thuẫn phát hiện": h.get("violations")
            })
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:
        st.info("💡 Chưa có lượt truy cập nào. Hãy bấm một trong 2 nút xanh/đỏ ở phía trên để tạo lượt gửi dữ liệu thử nghiệm đầu tiên!")

# ==========================================
# TAB 2: MANUAL PRESET SIMULATOR
# ==========================================
with tab_manual:
    st.subheader("🕹️ Kịch bản Kiểm thử Thủ công (Offline Presets)")
    st.caption("Cho phép người thuyết trình chủ động minh họa các mâu thuẫn đặc thù khi cần phân tích sâu.")

    preset_choice = st.selectbox(
        "Chọn một kịch bản mẫu để phân tích:",
        [
            "1. Người dùng thật: iPhone Safari chuẩn",
            "2. Người dùng thật: PC Windows Chrome chuẩn",
            "3. Anti-detect: Fake iPhone nhưng màn hình 1920x1080 (Screen Mismatch)",
            "4. Anti-detect: Fake iPhone nhưng maxTouchPoints = 0 (Touch API Mismatch)",
            "5. Anti-detect: Safari Mobile giả lập trên Linux Server (Platform Mismatch)",
            "6. Anti-detect: Dùng Proxy IP Pháp nhưng Timezone Mỹ (Geo Mismatch)",
            "7. Anti-detect: Giả lập Mobile nhưng lộ CPU 32 Cores (Hardware Anomaly)",
            "8. Anti-detect: Fake Mac Safari nhưng lộ WebGL Direct3D của Windows (WebGL Mismatch)"
        ]
    )

    if "1. Người dùng thật: iPhone" in preset_choice:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "iPhone", "Mobile Safari", "iPhone", 5, "1170x2532", 6, "US", "America/New_York", "Apple GPU"
    elif "2. Người dùng thật: PC Windows" in preset_choice:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "Desktop", "Chrome", "Win32", 0, "1920x1080", 8, "VN", "Asia/Ho_Chi_Minh", "ANGLE (NVIDIA RTX 3060 Direct3D11)"
    elif "3. Anti-detect: Fake iPhone nhưng màn hình 1920x1080" in preset_choice:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "iPhone", "Mobile Safari", "iPhone", 5, "1920x1080", 6, "US", "America/New_York", "Apple GPU"
    elif "4. Anti-detect: Fake iPhone nhưng maxTouchPoints = 0" in preset_choice:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "iPhone", "Mobile Safari", "iPhone", 0, "1170x2532", 6, "US", "America/New_York", "Apple GPU"
    elif "5. Anti-detect: Safari Mobile giả lập trên Linux" in preset_choice:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "iPhone", "Mobile Safari", "Linux x86_64", 5, "1170x2532", 6, "US", "America/New_York", "Apple GPU"
    elif "6. Anti-detect: Dùng Proxy IP Pháp" in preset_choice:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "Desktop", "Chrome", "Win32", 0, "1920x1080", 8, "FR", "America/Los_Angeles", "ANGLE (NVIDIA RTX 3060 Direct3D11)"
    elif "7. Anti-detect: Giả lập Mobile nhưng lộ CPU 32 Cores" in preset_choice:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "iPhone", "Mobile Safari", "iPhone", 5, "1170x2532", 32, "US", "America/New_York", "Apple GPU"
    else:
        p_dev, p_br, p_plat, p_touch, p_res, p_cpu, p_country, p_tz, p_gpu = "iPhone", "Mobile Safari", "iPhone", 5, "1170x2532", 6, "US", "America/New_York", "ANGLE (Intel Direct3D11)"

    c_m_left, c_m_right = st.columns([1, 1], gap="large")

    with c_m_left:
        st.markdown("##### 📥 Thuộc tính Thu thập từ Client")
        c1, c2 = st.columns(2)
        with c1:
            ua_dev = st.selectbox("Device Type (UA)", ["iPhone", "Android", "Desktop"], index=["iPhone", "Android", "Desktop"].index(p_dev))
            ua_br = st.selectbox("Browser (UA)", ["Mobile Safari", "Safari", "Chrome", "Chrome Mobile"], index=["Mobile Safari", "Safari", "Chrome", "Chrome Mobile"].index(p_br))
            plat = st.selectbox("Platform", ["iPhone", "MacIntel", "Win32", "Linux armv8l", "Linux x86_64"], index=["iPhone", "MacIntel", "Win32", "Linux armv8l", "Linux x86_64"].index(p_plat))
            touch = st.number_input("maxTouchPoints", min_value=0, max_value=20, value=p_touch)
        with c2:
            res = st.text_input("Screen Resolution", value=p_res)
            cpu = st.number_input("CPU Cores", min_value=1, max_value=64, value=p_cpu)
            country = st.selectbox("IP Country", ["US", "VN", "FR"], index=["US", "VN", "FR"].index(p_country))
            tz = st.text_input("Timezone Identifier", value=p_tz)
        gpu = st.text_input("WebGL Renderer", value=p_gpu)

        sim_df = pd.DataFrame([{
            "ua_device": ua_dev, "ua_browser": ua_br, "platform": plat,
            "max_touch": touch, "res": res, "cpu": cpu, "country": country, "tz": tz,
            "webgl_renderer": gpu
        }])
        sim_features = extract_features(sim_df)

    with c_m_right:
        st.markdown("##### ⚙️ Ma trận Mâu thuẫn (Feature Inconsistency Matrix)")
        f1, f2 = st.columns(2)
        f1.metric("Screen Mismatch", "❌ Phát hiện" if sim_features.loc[0, 'inconsistent_screen'] == 1 else "✅ Hợp lệ")
        f1.metric("Touch Mismatch", "❌ Phát hiện" if sim_features.loc[0, 'inconsistent_touch'] == 1 else "✅ Hợp lệ")
        f1.metric("Platform Mismatch", "❌ Phát hiện" if sim_features.loc[0, 'inconsistent_platform'] == 1 else "✅ Hợp lệ")

        f2.metric("Geo/Timezone Mismatch", "❌ Phát hiện" if sim_features.loc[0, 'inconsistent_geo'] == 1 else "✅ Hợp lệ")
        f2.metric("Hardware Anomaly", "❌ Phát hiện" if sim_features.loc[0, 'inconsistent_hardware'] == 1 else "✅ Hợp lệ")
        f2.metric("WebGL GPU Mismatch", "❌ Phát hiện" if sim_features.loc[0, 'inconsistent_webgl'] == 1 else "✅ Hợp lệ")

        st.metric("Tổng Điểm Mâu Thuẫn", f"{sim_features.loc[0, 'total_inconsistency_score']} / 6")

    st.divider()
    sim_pred_prob = model.predict_proba(sim_features)[0]
    sim_is_bot = model.predict(sim_features)[0]

    res_l, res_r = st.columns([1.2, 1], gap="large")
    with res_l:
        st.markdown("##### 🎯 Kết luận từ Mô hình Học máy (XGBoost Classifier)")
        if sim_is_bot == 1:
            st.error("🚨 **KẾT LUẬN: PHÁT HIỆN TRÌNH DUYỆT ANTI-DETECT / BOT**")
            st.write(f"Độ tin cậy Bot: **{sim_pred_prob[1]*100:.2f}%**")
            st.progress(float(sim_pred_prob[1]))
        else:
            st.success("✅ **KẾT LUẬN: NGƯỜI DÙNG HỢP LỆ (LEGITIMATE USER)**")
            st.write(f"Độ tin cậy Real User: **{sim_pred_prob[0]*100:.2f}%**")
            st.progress(float(sim_pred_prob[0]))

    with res_r:
        st.markdown("##### 📊 Trọng số Đặc trưng (Feature Importance)")
        feat_importances = pd.Series(model.feature_importances_, index=sim_features.columns)
        fig, ax = plt.subplots(figsize=(6, 3))
        feat_importances.sort_values().plot(kind='barh', color='#ff4b4b' if sim_is_bot == 1 else '#0083B8', ax=ax)
        ax.set_xlabel("F-Score")
        st.pyplot(fig)

# ==========================================
# TAB 3: RESEARCH BACKGROUND & METRICS
# ==========================================
with tab_docs:
    st.subheader("📚 Cơ sở Khoa học & Các Công bố Quốc tế Tham chiếu")

    col_d1, col_d2 = st.columns(2, gap="large")

    with col_d1:
        st.markdown("""
        #### 1. FP-Inconsistent (ACM IMC 2025)
        * **Ý tưởng**: Khai thác sự mâu thuẫn logic (Spatial & Temporal Inconsistencies) giữa các thuộc tính phần cứng và phần mềm của thiết bị.
        * **Đóng góp**: Chứng minh các anti-detect browser thương mại bị lộ mâu thuẫn ở độ phân giải, touch API và múi giờ.
        * **Hiệu quả**: Giảm 45%–48% tỷ lệ bot lọt lưới trong khi giữ tỷ lệ nhận diện người thật (TNR) đạt **96.84%**.

        #### 2. Browser Polygraph (ACM IMC 2024)
        * **Ý tưởng**: Dấu vân tay thô (Coarse-Grained Fingerprints - CGFP) kiểm tra chéo sự tương thích giữa chuỗi `User-Agent` và hành vi API thực tế.
        * **Ưu điểm**: Tốc độ thực thi chỉ vài mili-giây, không xâm phạm quyền riêng tư và triển khai được ở quy mô lớn.
        """)

    with col_d2:
        st.markdown("""
        #### 3. Him of Many Faces (NDSS 2023)
        * **Quy mô**: Nghiên cứu 36 tỷ request trên 14 website thương mại hàng đầu của F5, Inc.
        * **Quan sát**: Chỉ có 1.6% dấu vân tay đối nghịch chia sẻ với người dùng thật.
        * **Chiến lược đối nghịch**: Định nghĩa 4 chiến lược của bot gồm *Keep*, *Block*, *Mimic*, và *Randomize*.

        #### 4. Good Bot, Bad Bot (IEEE S&P 2021)
        * **Khảo sát**: 100 honeysites với 26.4 triệu yêu cầu.
        * **Phát hiện**: Trên 86.2% bot xấu cố tình khai báo User-Agent giả của Chrome/Firefox dù chạy trên thư viện HTTP thô sơ.
        """)
