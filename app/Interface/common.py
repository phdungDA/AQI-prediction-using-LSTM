# common.py
# Hàm dùng chung: CSS, đọc dữ liệu, nhãn/màu AQI, load model.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── Đường dẫn tới project (tự suy ra từ vị trí file, không cần sửa khi đổi ổ/máy) ──
# common.py nằm ở <project>/app/Interface/ → project root là 2 cấp trên.
# Có thể ghi đè bằng biến môi trường AQI_PROJECT_ROOT.
PROJECT_ROOT = os.environ.get(
    "AQI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
)
RAW_CSV      = os.path.join(PROJECT_ROOT, "data", "raw", "air_quality_3years.csv")
MODEL_PATH   = os.path.join(PROJECT_ROOT, "models", "lstm_baseline.h5")
SCALER_PATH  = os.path.join(PROJECT_ROOT, "models", "minmax_scaler.pkl")

# Phải khớp src/data/dataloader.py (FEATURE_COLS, WINDOW_SIZE, MAX_GAP_HOURS)
FEATURES   = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
WINDOW     = 24
MAX_GAP    = 6

# AQI của OpenWeather: 1..5
AQI_LABELS = {1: "Tốt", 2: "Khá", 3: "Trung bình", 4: "Kém", 5: "Rất kém"}
AQI_COLORS = {1: "#2e9e5b", 2: "#9bc53d", 3: "#f5b400", 4: "#ee7d2d", 5: "#d64545"}
AQI_ADVICE = {
    1: "Không khí trong lành, thoải mái hoạt động ngoài trời.",
    2: "Chất lượng chấp nhận được, người nhạy cảm nên để ý.",
    3: "Người nhạy cảm nên hạn chế hoạt động mạnh ngoài trời.",
    4: "Nên hạn chế ra ngoài, đeo khẩu trang khi cần.",
    5: "Hạn chế tối đa ra ngoài, dùng máy lọc không khí trong nhà.",
}
POLLUTANT_NAMES = {
    "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "O₃",
    "so2": "SO₂", "co": "CO", "no": "NO", "nh3": "NH₃",
}
# 6 chất hiển thị trên Dashboard, theo đúng thứ tự ảnh mẫu
POLLUTANT_ORDER = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]

# Ngưỡng đánh giá riêng từng chất (µg/m³, trung bình 1h) — xấp xỉ theo
# QCVN 05:2023/BTNMT, chỉ dùng để tô màu/nhãn từng thẻ, KHÔNG phải AQI tổng.
# Cận trên của mức 1..4; vượt mức 4 là mức 5.
POLLUTANT_THRESHOLDS = {
    "pm2_5": [25, 50, 80, 150],
    "pm10":  [50, 100, 150, 200],
    "co":    [10000, 20000, 30000, 40000],
    "no2":   [50, 100, 200, 300],
    "so2":   [50, 125, 350, 500],
    "o3":    [60, 120, 180, 240],
    "no":    [50, 100, 200, 300],
    "nh3":   [200, 400, 800, 1200],
}


def pollutant_level(name: str, value: float) -> int:
    """Mức 1..5 của một chất theo ngưỡng xấp xỉ ở trên."""
    if pd.isna(value):
        return 1
    for i, bound in enumerate(POLLUTANT_THRESHOLDS.get(name, [50, 100, 150, 200]), start=1):
        if value <= bound:
            return i
    return 5


# ── CSS ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        .aqi-card {border-radius: 16px; padding: 24px; color: #fff; text-align: center;}
        .aqi-card .value {font-size: 64px; font-weight: 700; line-height: 1;}
        .aqi-card .label {font-size: 22px; margin-top: 6px;}
        .aqi-card .note  {font-size: 14px; opacity: .9; margin-top: 10px;}
        .badge {display:inline-block; padding:2px 12px; border-radius:12px;
                color:#fff; font-weight:600; font-size:13px;}

        /* ── Dashboard theo ảnh mẫu ── */
        .page-title {font-size: 15px; font-weight: 700; letter-spacing:.4px;
                     color:#374151; text-transform:uppercase; margin-bottom:2px;}
        .page-sub   {color:#6b7280; font-size:14px; margin-bottom: 14px;}

        .hero-card {border-radius:16px; padding:22px 26px;
                    background: linear-gradient(135deg, var(--c1), var(--c2));
                    display:flex; justify-content:space-between; align-items:center;
                    flex-wrap:wrap; gap:16px;}
        .hero-left  {display:flex; align-items:center; gap:22px;}
        .hero-aqi   {text-align:center;}
        .hero-aqi .lbl {font-size:12px; color:#6b7280; font-weight:600;}
        .hero-aqi .val {font-size:40px; font-weight:800; line-height:1; color:#1f2937;}
        .hero-badge {display:inline-block; padding:5px 16px; border-radius:20px;
                     color:#fff; font-weight:700; font-size:14px; margin-bottom:8px;}
        .hero-note  {font-size:13.5px; color:#4b5563; max-width:420px;}
        .hero-right {text-align:right; font-size:13px; color:#6b7280;}
        .hero-right b {color:#374151;}

        .scale-row {display:flex; gap:6px; margin-top:4px;}
        .scale-item {flex:1; text-align:center; font-size:12px; color:#9ca3af;}
        .scale-item .dot {height:6px; border-radius:4px; margin-bottom:4px;}
        .scale-item.active {color:#1f2937; font-weight:700;}

        .section-h {font-size:15px; font-weight:700; color:#374151; margin:22px 0 10px;}

        .pcard {background:#fff; border:1px solid #eef0f2; border-radius:14px;
                padding:16px 18px; height:100%;}
        .pcard .top {display:flex; justify-content:space-between; align-items:center;
                     font-size:13.5px; color:#374151; font-weight:600;}
        .pcard .pbadge {font-size:11.5px; font-weight:700; padding:2px 10px;
                        border-radius:10px; color:#fff;}
        .pcard .pval {font-size:22px; font-weight:800; color:#1f2937; margin:8px 0 4px;}
        .pcard .pfoot {display:flex; justify-content:space-between; align-items:center;
                       font-size:12px; color:#9ca3af; margin-top:4px;}
        .pcard .trend-up   {color:#d64545; font-weight:700;}
        .pcard .trend-down {color:#2e9e5b; font-weight:700;}
        .pcard .trend-flat {color:#9ca3af; font-weight:700;}

        .info-card {background:#fff; border:1px solid #eef0f2; border-radius:14px;
                    padding:16px 18px; height:100%;}
        .info-card .ihead {display:flex; justify-content:space-between; align-items:center;
                           font-size:13.5px; color:#374151; font-weight:600; margin-bottom:10px;}
        .info-card .ititle{font-size:15px; font-weight:700; color:#1f2937;}
        .info-card .inote {font-size:12.5px; color:#6b7280; margin-top:8px; line-height:1.5;}

        .metric-card {background:#fff; border:1px solid #eef0f2; border-radius:14px;
                      padding:16px 18px; text-align:left; height:100%;}
        .metric-card .mlbl {font-size:12.5px; color:#6b7280; margin-bottom:6px;}
        .metric-card .mval {font-size:24px; font-weight:800; color:#1f2937;}
        .metric-card .msub {font-size:12.5px; color:#9ca3af; margin-top:4px;}

        .dist-bar {display:flex; height:10px; border-radius:6px; overflow:hidden; margin:8px 0 10px;}
        .dist-legend {display:flex; gap:18px; flex-wrap:wrap; font-size:12.5px; color:#4b5563;}
        .dist-legend .dot {display:inline-block; width:9px; height:9px; border-radius:50%;
                           margin-right:5px; vertical-align:middle;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def aqi_card(level: int, title: str = "Chỉ số AQI"):
    level = int(level)
    st.markdown(
        f"""
        <div class="aqi-card" style="background:{AQI_COLORS[level]}">
          <div>{title}</div>
          <div class="value">{level}<span style="font-size:24px">/5</span></div>
          <div class="label">{AQI_LABELS[level]}</div>
          <div class="note">{AQI_ADVICE[level]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scale_bar_html(active_level: int) -> str:
    """Thanh 5 mức Tốt..Rất kém, tô đậm mức hiện tại."""
    items = []
    for lvl in range(1, 6):
        cls = "scale-item active" if lvl == active_level else "scale-item"
        tag = " (Hiện tại)" if lvl == active_level else ""
        items.append(
            f'<div class="{cls}"><div class="dot" style="background:{AQI_COLORS[lvl]}"></div>'
            f'{AQI_LABELS[lvl]}<br>{lvl}{tag}</div>'
        )
    return f'<div class="scale-row">{"".join(items)}</div>'


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def sparkline_fig(series: pd.Series, color: str):
    """Biểu đồ mini không trục, dùng trong thẻ từng chất ô nhiễm. color là mã hex."""
    import plotly.graph_objects as go
    fig = go.Figure(go.Scatter(
        y=series.values, mode="lines", line=dict(color=color, width=2.2),
        fill="tozeroy", fillcolor=_hex_to_rgba(color, 0.12),
        hoverinfo="skip",
    ))
    fig.update_layout(
        height=46, margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def trend_arrow(current: float, previous: float):
    """(mũi tên, nhãn, css-class) so với giờ trước."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "─", "ổn định", "trend-flat"
    pct = (current - previous) / abs(previous)
    if pct > 0.02:
        return "▲", "tăng", "trend-up"
    if pct < -0.02:
        return "▼", "giảm", "trend-down"
    return "─", "ổn định", "trend-flat"


# ── Dữ liệu ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Đang đọc dữ liệu...")
def load_data() -> pd.DataFrame:
    """Đọc CSV, đưa về lưới 1 giờ, nội suy gap ngắn cho các chất ô nhiễm.
    Cột AQI không được nội suy. Thời gian theo UTC."""
    df = pd.read_csv(RAW_CSV)
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.floor("60min")
    df = (df.drop_duplicates(subset="datetime", keep="last")
            .set_index("datetime").sort_index())
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", name="datetime")
    df = df.reindex(full)

    # Giống dataloader.handle_missing: nội suy rồi xoá toàn bộ gap dài hơn MAX_GAP,
    # để cửa sổ đưa vào model có cùng phân phối với lúc train.
    for col in FEATURES:
        is_na = df[col].isna()
        filled = df[col].interpolate(method="linear", limit_area="inside")
        filled[_long_gap_mask(is_na, MAX_GAP)] = np.nan
        df[col] = filled
    return df


def _long_gap_mask(is_na: pd.Series, max_gap: int) -> pd.Series:
    """True cho các ô nằm trong đoạn NaN liên tiếp dài hơn max_gap."""
    grp = (is_na != is_na.shift()).cumsum()
    run = is_na.groupby(grp).transform("size")
    return is_na & (run > max_gap)


def get_data() -> pd.DataFrame:
    """Như load_data nhưng hiện lỗi thân thiện thay vì crash."""
    if not os.path.exists(RAW_CSV):
        st.error(f"Không tìm thấy file dữ liệu:\n`{RAW_CSV}`\n\n"
                 "Hãy chạy `python src/ingestion/api_client.py` trong project để tải dữ liệu.")
        st.stop()
    return load_data()


def latest_row(df: pd.DataFrame) -> pd.Series:
    """Bản ghi mới nhất có nhãn AQI hợp lệ."""
    return df.dropna(subset=["aqi"]).iloc[-1]


def last_update_text() -> str:
    if not os.path.exists(RAW_CSV):
        return "Chưa có dữ liệu"
    return latest_row(load_data()).name.strftime("%H:%M %d/%m/%Y") + " (UTC)"


# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Đang tải model...")
def load_model_bundle():
    """Trả về (model, scaler) hoặc (None, lỗi) nếu không load được."""
    try:
        import joblib
        from tensorflow.keras.models import load_model
        model = load_model(MODEL_PATH, compile=False)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:  # thiếu tensorflow / thiếu file / lỗi phiên bản
        return None, str(e)


def predict_next_hour(df: pd.DataFrame):
    """Dự đoán lớp AQI (1..5) của giờ kế tiếp từ 24 giờ gần nhất.
    Trả về (level, probs) hoặc (None, thông báo lỗi)."""
    model, scaler = load_model_bundle()
    if model is None:
        return None, f"Không load được model: {scaler}"

    last_valid = df["aqi"].last_valid_index()
    win = df.loc[:last_valid, FEATURES].tail(WINDOW)   # 24 giờ liên tiếp, không bỏ giờ nào
    if len(win) < WINDOW or win.isna().any().any():
        return None, "24 giờ gần nhất còn thiếu dữ liệu nên không dự đoán được."

    X = scaler.transform(win).astype("float32")[None, ...]
    probs = model.predict(X, verbose=0)[0]
    return int(np.argmax(probs)) + 1, probs
