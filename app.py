import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import base64
import socket
import datetime
import urllib.parse
import time

# Cấu hình giao diện Streamlit
st.set_page_config(
    page_title="Anti-detect Browser Detection AI",
    layout="wide",
    page_icon="🛡️"
)

# ==========================================
# 0. HỖ TRỢ MẠNG VÀ BỘ NHỚ LỊCH SỬ (CACHE)
# ==========================================
def get_local_ip():
    """Lấy địa chỉ IP mạng nội bộ (LAN) làm fallback."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"

@st.cache_resource
def get_live_history():
    """Lưu trữ lịch sử các thiết bị truy cập thực nghiệm (Thread-safe singleton)."""
    return []

# ==========================================
# 1. HÀM TẠO DỮ LIỆU & HUẤN LUYỆN MODEL (CACHE)
# ==========================================
@st.cache_resource
def init_and_train_model():
    np.random.seed(42)
    n_samples = 10000

    # Cấu hình chuẩn của người dùng thật (Benign profiles)
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
    if "iphone" in ua_str_lower:
        dev = "iPhone"
        br = "Mobile Safari" if "safari" in ua_str_lower else "Chrome Mobile"
    elif "android" in ua_str_lower:
        dev = "Android"
        br = "Chrome Mobile"
    elif "macintosh" in ua_str_lower or "mac os" in ua_str_lower:
        dev = "Desktop"
        br = "Safari" if "safari" in ua_str_lower and "chrome" not in ua_str_lower else "Chrome"
    else:
        dev = "Desktop"
        br = "Chrome" if "chrome" in ua_str_lower else "Edge"
    return dev, br

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

    # 3. Platform Inconsistency (OS vs Navigator Platform)
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
local_ip = get_local_ip()
default_public_url = "https://anti-detect-bot-detection-hcdmmdswgfmb9brpacxdos.streamlit.app"

# ==========================================
# 2. XỬ LÝ DỮ LIỆU TỰ ĐỘNG TỪ CLIENT (JS PAYLOAD)
# ==========================================
query_params = st.query_params
raw_fp = query_params.get("fp", None)
auto_detected_data = None

if raw_fp:
    try:
        decoded_bytes = base64.b64decode(raw_fp)
        json_str = urllib.parse.unquote(decoded_bytes.decode('utf-8'))
        auto_detected_data = json.loads(json_str)

        # Lưu URL gốc của trình duyệt
        if "origin" in auto_detected_data and auto_detected_data["origin"]:
            st.session_state["public_url"] = auto_detected_data["origin"]

        # Trích xuất và dự đoán
        parsed_dev, parsed_br = parse_user_agent(auto_detected_data.get("ua", ""))
        client_row = {
            "ua_device": parsed_dev,
            "ua_browser": parsed_br,
            "platform": auto_detected_data.get("platform", "Unknown"),
            "max_touch": auto_detected_data.get("max_touch", 0),
            "res": auto_detected_data.get("res", "Unknown"),
            "cpu": auto_detected_data.get("cpu", 4),
            "country": "VN" if "Asia/Ho_Chi_Minh" in auto_detected_data.get("tz", "") or "Asia/Bangkok" in auto_detected_data.get("tz", "") else "Auto/Client",
            "tz": auto_detected_data.get("tz", "Unknown"),
            "webgl_renderer": auto_detected_data.get("webgl_renderer", "Unknown")
        }

        eval_df = pd.DataFrame([client_row])
        feat_matrix = extract_features(eval_df)
        pred_is_bot = int(model.predict(feat_matrix)[0])
        pred_prob = model.predict_proba(feat_matrix)[0]

        # Quản lý lịch sử Live Feed toàn cục (Thread-safe)
        history = get_live_history()
        session_id = auto_detected_data.get("sid", str(auto_detected_data.get("ts", "")))

        # Kiểm tra xem session_id này đã được log chưa
        existing_idx = next((i for i, h in enumerate(history) if h.get("sid") == session_id), None)

        violations = []
        if feat_matrix.loc[0, "inconsistent_screen"] == 1:
            violations.append("Màn hình không khớp thiết bị")
        if feat_matrix.loc[0, "inconsistent_touch"] == 1:
            violations.append("Touch Points bất thường")
        if feat_matrix.loc[0, "inconsistent_platform"] == 1:
            violations.append("Platform lệch với Browser")
        if feat_matrix.loc[0, "inconsistent_hardware"] == 1:
            violations.append(f"CPU ({client_row['cpu']} cores) bất thường cho Mobile")
        if feat_matrix.loc[0, "inconsistent_webgl"] == 1:
            violations.append("WebGL Renderer lộ GPU Desktop")

        record = {
            "time": datetime.datetime.now().strftime("%H:%M:%S"),
            "sid": session_id,
            "device": f"{client_row['ua_device']} ({client_row['ua_browser']})",
            "platform": client_row['platform'],
            "res": client_row['res'],
            "touch": client_row['max_touch'],
            "is_bot": pred_is_bot,
            "bot_confidence": f"{pred_prob[1]*100:.1f}%" if pred_is_bot == 1 else f"{pred_prob[0]*100:.1f}%",
            "violations": ", ".join(violations) if violations else "Không có (Chuẩn)",
            "raw_gpu": client_row['webgl_renderer']
        }

        if existing_idx is not None:
            history[existing_idx] = record
        else:
            history.insert(0, record)

    except Exception as e:
        pass

# URL công khai
current_public_url = st.session_state.get("public_url", default_public_url)

# ==========================================
# 3. GIAO DIỆN CHÍNH & TABS
# ==========================================
st.markdown("""
<style>
    .main-title { font-size: 2.2rem; font-weight: 800; color: #1E293B; margin-bottom: 0px; }
    .sub-title { font-size: 1.05rem; color: #64748B; margin-bottom: 15px; }
    .metric-box { background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 15px; text-align: center; }
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
    # Nhúng Script JavaScript thu thập vân tay
    js_fingerprint_script = f"""
    <script>
    (function() {{
        function getSafeParentUrl() {{
            try {{
                if (window.top && window.top.location && window.top.location.href) {{
                    return new URL(window.top.location.href);
                }}
            }} catch(e) {{}}
            try {{
                if (document.referrer) {{
                    return new URL(document.referrer);
                }}
            }} catch(e) {{}}
            return new URL("{default_public_url}");
        }}

        function collectAndNavigate() {{
            let glVendor = "Unknown", glRenderer = "Unknown";
            try {{
                const canvas = document.createElement("canvas");
                const gl = canvas.getContext("webgl") || canvas.getContext("experimental-webgl");
                if (gl) {{
                    const dbg = gl.getExtension("WEBGL_debug_renderer_info");
                    if (dbg) {{
                        glVendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || "";
                        glRenderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || "";
                    }}
                }}
            }} catch(e) {{}}

            let sid = localStorage.getItem("bot_detect_sid");
            if (!sid) {{
                sid = "dev_" + Math.random().toString(36).substring(2, 9) + "_" + Date.now().toString(36);
                localStorage.setItem("bot_detect_sid", sid);
            }}

            const parentUrl = getSafeParentUrl();
            const publicOrigin = parentUrl.origin + parentUrl.pathname;

            const fp = {{
                ua: navigator.userAgent,
                platform: navigator.platform,
                max_touch: navigator.maxTouchPoints || 0,
                res: window.screen.width + "x" + window.screen.height,
                cpu: navigator.hardwareConcurrency || 4,
                mem: navigator.deviceMemory || 4,
                tz: Intl.DateTimeFormat().resolvedOptions().timeZone || "Unknown",
                webgl_vendor: glVendor,
                webgl_renderer: glRenderer,
                origin: publicOrigin,
                sid: sid,
                ts: Date.now()
            }};

            const rawStr = encodeURIComponent(JSON.stringify(fp));
            const base64Str = btoa(unescape(rawStr));

            if (parentUrl.searchParams.get("fp") !== base64Str) {{
                parentUrl.searchParams.set("fp", base64Str);
                try {{
                    if (window.top) {{
                        window.top.location.replace(parentUrl.href);
                        return;
                    }}
                }} catch(e) {{}}
                window.open(parentUrl.href, "_top");
            }}
        }}

        setTimeout(collectAndNavigate, 500);
    }})();
    </script>
    """
    st.components.v1.html(js_fingerprint_script, height=0)

    # Hiển thị thông báo nếu người dùng đang truy cập trên thiết bị di động
    if auto_detected_data:
        st.success(f"🎉 **Thiết bị của bạn đã được kết nối & phân tích thành công!** (Mã phiên: `{auto_detected_data.get('sid', 'N/A')}`)")

    col_qr, col_stats = st.columns([1, 2], gap="large")

    with col_qr:
        st.subheader("📲 Quét mã để Tham gia Demo")
        
        # QR Widget luôn tự động phát hiện đúng Domain Public của trình duyệt
        qr_widget_html = f"""
        <div style="text-align: center; background: #ffffff; padding: 12px; border-radius: 12px; border: 1px solid #e2e8f0; max-width: 250px; margin-bottom: 10px;">
            <img id="live-dynamic-qr" src="https://api.qrserver.com/v1/create-qr-code/?size=220x220&data={urllib.parse.quote(default_public_url)}" style="width: 200px; height: 200px; border-radius: 8px;" />
            <p id="live-url-text" style="font-size: 0.75rem; color: #475569; margin-top: 8px; word-break: break-all; font-family: monospace; font-weight: bold;">{default_public_url}</p>
        </div>
        <script>
            try {{
                let u = "{default_public_url}";
                if (window.top && window.top.location && window.top.location.href) {{
                    u = window.top.location.origin + window.top.location.pathname;
                }} else if (document.referrer) {{
                    const ref = new URL(document.referrer);
                    u = ref.origin + ref.pathname;
                }}
                document.getElementById("live-dynamic-qr").src = "https://api.qrserver.com/v1/create-qr-code/?size=220x220&data=" + encodeURIComponent(u);
                document.getElementById("live-url-text").innerText = u;
            }} catch(e) {{}}
        </script>
        """
        st.components.v1.html(qr_widget_html, height=270)

        c_btn1, c_btn2 = st.columns(2)
        with c_btn1:
            if st.button("🔄 Làm mới Bảng Live"):
                st.rerun()
        with c_btn2:
            if st.button("🗑️ Xóa Lịch sử Log"):
                get_live_history().clear()
                st.rerun()

    with col_stats:
        st.subheader("📊 Thống kê Hiệu năng Thực tế (Live Metrics)")
        history = get_live_history()
        total_req = len(history)
        real_users = sum(1 for h in history if h.get("is_bot") == 0)
        bot_users = sum(1 for h in history if h.get("is_bot") == 1)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng thiết bị quét", total_req)
        m2.metric("Người thật (Real User)", real_users)
        m3.metric("Phát hiện Bot / Anti-detect", bot_users)
        tnr_pct = f"{(real_users / max(1, total_req))*100:.1f}%" if total_req > 0 else "100%"
        m4.metric("Tỷ lệ Real User", tnr_pct)

        # Hiển thị thông tin phân tích thiết bị hiện tại
        if auto_detected_data:
            st.markdown("---")
            st.markdown("#### 🔍 Phân tích Thiết bị của Bạn Hiện tại:")
            p_dev, p_br = parse_user_agent(auto_detected_data.get("ua", ""))
            c_info1, c_info2 = st.columns([1.2, 1])

            with c_info1:
                st.write(f"- **User-Agent**: `{auto_detected_data.get('ua')[:60]}...`")
                st.write(f"- **Nhận diện**: `{p_dev} - {p_br}` | **Platform**: `{auto_detected_data.get('platform')}`")
                st.write(f"- **Màn hình**: `{auto_detected_data.get('res')}` | **Touch Points**: `{auto_detected_data.get('max_touch')}`")
                st.write(f"- **Múi giờ**: `{auto_detected_data.get('tz')}` | **CPU Cores**: `{auto_detected_data.get('cpu')}`")
                st.write(f"- **WebGL GPU**: `{auto_detected_data.get('webgl_renderer', 'Unknown')[:50]}`")

            with c_info2:
                temp_row = {
                    "ua_device": p_dev, "ua_browser": p_br, "platform": auto_detected_data.get("platform"),
                    "max_touch": auto_detected_data.get("max_touch", 0), "res": auto_detected_data.get("res"),
                    "cpu": auto_detected_data.get("cpu", 4), "country": "VN", "tz": auto_detected_data.get("tz"),
                    "webgl_renderer": auto_detected_data.get("webgl_renderer")
                }
                temp_feats = extract_features(pd.DataFrame([temp_row]))
                temp_bot_pred = model.predict(temp_feats)[0]
                temp_prob = model.predict_proba(temp_feats)[0]

                if temp_bot_pred == 1:
                    st.error("🚨 **KẾT QUẢ: PHÁT HIỆN ANTI-DETECT / BOT**")
                    st.write(f"Độ tin cậy Bot: **{temp_prob[1]*100:.1f}%**")
                else:
                    st.success("✅ **KẾT QUẢ: NGƯỜI DÙNG HỢP LỆ (REAL USER)**")
                    st.write(f"Độ tin cậy Real User: **{temp_prob[0]*100:.1f}%**")

    st.divider()
    
    col_hdr1, col_hdr2 = st.columns([3, 1])
    with col_hdr1:
        st.subheader("📋 Bảng Giám sát Truy cập Thời gian thực (Real-time Live Feed)")
    with col_hdr2:
        auto_refresh_on = st.toggle("⚡ Tự động cập nhật mỗi 3s", value=True)

    if history:
        display_data = []
        for h in history:
            badge = "🚨 BOT / ANTI-DETECT" if h.get("is_bot") == 1 else "✅ NGƯỜI THẬT"
            display_data.append({
                "Thời gian": h.get("time"),
                "Mã phiên": h.get("sid", "N/A")[:10] + "...",
                "Thiết bị nhận diện": h.get("device"),
                "Platform": h.get("platform"),
                "Độ phân giải": h.get("res"),
                "Touch Points": h.get("touch"),
                "Kết luận AI": badge,
                "Độ tin cậy": h.get("bot_confidence"),
                "Lỗi mâu thuẫn phát hiện": h.get("violations")
            })
        st.dataframe(pd.DataFrame(display_data), use_container_width=True, hide_index=True)
    else:
        st.info("💡 Chưa có lượt truy cập nào được ghi lại. Hãy mở camera điện thoại quét mã QR phía trên để bắt đầu gửi vân tay!")

    # Tự động reload trang dashboard mỗi 3 giây nếu bật auto-refresh
    if auto_refresh_on:
        st.components.v1.html(
            """
            <script>
            setTimeout(function() {
                try {
                    window.parent.postMessage({type: 'streamlit:setComponentValue'}, '*');
                } catch(e) {}
            }, 3000);
            </script>
            """,
            height=0
        )

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
