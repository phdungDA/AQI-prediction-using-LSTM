# common.py
# Hàm dùng chung: CSS, đọc dữ liệu, nhãn/màu AQI, load model.

import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
import streamlit as st

from language import aqi_label, t

# ── Đường dẫn tới project (tự suy ra từ vị trí file, không cần sửa khi đổi ổ/máy) ──
# common.py nằm ở <project>/app/Interface/ → project root là 2 cấp trên.
# Có thể ghi đè bằng biến môi trường AQI_PROJECT_ROOT.
PROJECT_ROOT = os.environ.get(
    "AQI_PROJECT_ROOT", str(Path(__file__).resolve().parents[2])
)
# Nạp .env ở thư mục gốc project (Streamlit không tự đọc .env). Biến môi trường đã có sẵn sẽ không bị ghi đè.
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Địa chỉ FastAPI backend (đọc từ .env hoặc biến môi trường API_BASE_URL; mặc định chạy local)
API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT  = 30  # giây

# API trả giờ UTC; FE cộng thêm 7 tiếng để hiển thị theo giờ Việt Nam (GMT+7)
TZ_OFFSET_HOURS = 7
TZ_LABEL        = "GMT+7"

# AQI của OpenWeather: 1..5. Tên mức và lời khuyên (2 ngôn ngữ) nằm trong language.py
AQI_COLORS = {1: "#2e9e5b", 2: "#9bc53d", 3: "#f5b400", 4: "#ee7d2d", 5: "#d64545"}
POLLUTANT_NAMES = {
    "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂", "o3": "O₃",
    "so2": "SO₂", "co": "CO", "no": "NO", "nh3": "NH₃",
}
# 6 chất hiển thị trên Dashboard, theo đúng thứ tự ảnh mẫu
POLLUTANT_ORDER = ["pm2_5", "pm10", "co", "no2", "so2", "o3"]

# Ngưỡng đánh giá riêng từng chất (µg/m³) — ĐÚNG THEO TÀI LIỆU CHÍNH THỨC
# OpenWeather Air Pollution API: https://openweathermap.org/api/air-pollution
# Dạng [thấp, cao) — không bao gồm cận trên. Đây cũng chính là nguồn dữ liệu
# project đang cào về (src/ingestion/), nên dùng ngưỡng này cho nhất quán.
# NO và NH3 KHÔNG nằm trong công thức tính AQI chính thức của OpenWeather —
# 2 chất này chỉ trả về nồng độ thô, không chuẩn AQI quốc gia nào quy định
# ngưỡng cho NO riêng (chỉ NO2), ngưỡng dưới đây cho no/nh3 là tự đặt tạm.
POLLUTANT_THRESHOLDS = {
    "so2":   [20, 80, 250, 350],
    "no2":   [40, 70, 150, 200],
    "pm10":  [20, 50, 100, 200],
    "pm2_5": [10, 25, 50, 75],
    "o3":    [60, 100, 140, 180],
    "co":    [4400, 9400, 12400, 15400],
    "no":    [50, 100, 200, 300],     # không có qui chuẩn chính thức
    "nh3":   [200, 400, 800, 1200],   # không có qui chuẩn chính thức
}


# Quy đổi của Berkeley Earth: hít PM2.5 22 µg/m³ trong 24 giờ ≈ hút 1 điếu thuốc
CIGARETTE_PM25 = 22


def pollutant_level(name: str, value: float) -> int:
    """Mức 1..5 của một chất theo ngưỡng ở trên; 0 nếu không có dữ liệu."""
    if pd.isna(value):
        return 0
    for i, bound in enumerate(POLLUTANT_THRESHOLDS.get(name, [50, 100, 150, 200]), start=1):
        if value < bound:
            return i
    return 5


# ── CSS ──────────────────────────────────────────────────────────────────────
def inject_css():
    st.markdown(
        """
        <style>
        /* ── Sidebar: cỡ chữ theo cùng thang với các trang ── */
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {font-size:16px;}
        [data-testid="stSidebar"] button [data-testid="stMarkdownContainer"] p {font-size:17px; font-weight:600;}
        [data-testid="stSidebar"] button {padding-top:10px; padding-bottom:10px;}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {font-size:15px; color:#6b7280;}

        /* ── Dashboard theo ảnh mẫu ── */
        .page-title {font-size: 35px; font-weight: 700; letter-spacing:.4px;
                     color:#374151; text-transform:uppercase; margin-bottom:2px;}
        .page-sub   {color:#6b7280; font-size:18px; margin-bottom: 14px;}

        /* Thang chữ dùng chung cho mọi ô:
           nhãn 15px · chữ giải thích 16px · tiêu đề ô 18px · tiêu đề mục 20px · số liệu 26–30px.
           Chữ phụ dùng #6b7280 (không dùng #9ca3af vì quá nhạt, khó đọc). */
        .hero-note  {font-size:18px; color:#4b5563; line-height:1.55; max-width:520px;}

        .section-h {font-size:20px; font-weight:700; color:#374151; margin:26px 0 12px;}

        .pcard {background:#fff; border:1px solid #1f2937; border-radius:14px;
                padding:16px 18px; height:100%;}
        .pcard .top {display:flex; justify-content:space-between; align-items:center;
                     flex-wrap:wrap; gap:6px; font-size:16px; color:#374151; font-weight:700;}
        .pbadge {display:inline-block; font-size:14px; font-weight:700; padding:3px 11px;
                 border-radius:999px; color:#fff; line-height:1.4;}
        .pcard .pval {font-size:26px; font-weight:800; color:#1f2937; margin:8px 0 4px;}
        .unit {font-size:15px; font-weight:600; color:#6b7280;}
        .pfoot {font-size:15px; color:#6b7280; margin-top:4px;}
        .trend-up   {color:#d64545; font-weight:700;}
        .trend-down {color:#2e9e5b; font-weight:700;}
        .trend-flat {color:#6b7280; font-weight:700;}

        /* min-height cố định để các ô cùng hàng cao bằng nhau (cột Streamlit không tự kéo giãn theo nhau) */
        .info-card {background:#fff; border:1px solid #1f2937; border-radius:14px;
                    padding:20px 22px; min-height:190px; box-sizing:border-box;
                    display:flex; flex-direction:column; justify-content:center;}
        .info-card .ihead {display:flex; justify-content:space-between; align-items:center;
                           font-size:18px; color:#374151; font-weight:700; margin-bottom:12px;}
        .info-card .pbadge {font-size:18px; padding:6px 18px;}
        .info-card .ival  {font-size:30px; font-weight:800; color:#1f2937; line-height:1.2;}
        .info-card .inote {font-size:16px; color:#4b5563; margin-top:10px; line-height:1.55;}

        /* Quy trình tính confidence: 4 bước theo thứ tự + thanh xác suất 5 mức */
        .steps {display:grid; grid-template-columns:repeat(auto-fit, minmax(220px, 1fr)); gap:14px;}
        .step  {background:#fff; border:1px solid #1f2937; border-radius:14px; padding:16px 18px;}
        .snum  {width:28px; height:28px; border-radius:50%; background:#374151; color:#fff;
                font-size:15px; font-weight:700; display:flex; align-items:center;
                justify-content:center; margin-bottom:10px;}
        .stitle {font-size:17px; font-weight:700; color:#1f2937; margin-bottom:6px;}
        .stext  {font-size:16px; color:#4b5563; line-height:1.55;}
        .probs {background:#fff; border:1px solid #1f2937; border-radius:14px;
                padding:18px 22px; margin-top:14px; display:flex; flex-direction:column; gap:10px;}
        .prow  {display:grid; grid-template-columns:110px 1fr 64px; align-items:center; gap:12px;
                font-size:16px; color:#4b5563;}
        .prow.top {font-weight:700; color:#1f2937;}
        .pbar  {height:12px; border-radius:6px; background:#eef0f2; overflow:hidden;}
        .pbar i {display:block; height:100%; border-radius:6px;}
        .pct   {text-align:right; font-variant-numeric:tabular-nums;}

        /* Bảng giải thích thang AQI; cuộn ngang trong khung khi màn hình hẹp */
        .scale-wrap  {background:#fff; border:1px solid #1f2937; border-radius:14px; overflow-x:auto;}
        /* Mỗi ô nằm trên 1 dòng (nowrap); màn hình hẹp thì .scale-wrap cuộn ngang.
           Mọi selector bắt đầu bằng .scale-wrap để mạnh hơn style bảng mặc định của Streamlit
           (selector dạng ".class table/th/td": viền xám từng ô, kẻ trên mỗi hàng, margin dưới bảng). */
        .scale-wrap table.scale-table {width:100%; border-collapse:collapse; margin:0;}
        .scale-wrap .scale-table tr {border:none;}
        .scale-wrap .scale-table th, .scale-wrap .scale-table td {
                         border:none; border-right:1px solid #1f2937; border-bottom:1px solid #1f2937;
                         padding:12px 14px; white-space:nowrap; vertical-align:middle; color:#1f2937;}
        .scale-wrap .scale-table th:last-child, .scale-wrap .scale-table td:last-child {border-right:none;}
        .scale-wrap .scale-table tr:last-child td {border-bottom:none;}
        .scale-wrap .scale-table th {background:#f9fafb; text-align:left; font-size:16px; font-weight:800;}
        .scale-wrap .scale-table td {font-size:16px; line-height:1.5;}
        .scale-wrap .scale-table td.cat {font-weight:700;}
        .scale-wrap .scale-table th:first-child {width:80px;}
        .scale-wrap .scale-table .pbadge {min-width:52px; text-align:center; box-sizing:border-box;}

        .metric-card {background:#fff; border:1px solid #1f2937; border-radius:14px;
                      padding:18px 20px; text-align:left; height:100%;}
        .metric-card .mlbl {font-size:15px; font-weight:600; color:#6b7280; margin-bottom:6px;}
        .metric-card .mval {font-size:28px; font-weight:800; color:#1f2937; line-height:1.2;}
        .metric-card .msub {font-size:16px; color:#4b5563; margin-top:6px; line-height:1.5;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def scale_bar_html(active_level: int) -> str:
    """Đồng hồ gauge bán nguyệt SVG, kim tự chỉ theo active_level (1..5)."""
    import math

    cx, cy = 290, 235
    R_out, R_in = 185, 115
    needle_R = 170
    needle_base = 9
    label_R = 215

    # Toạ độ 6 điểm mốc (180°→0°, mỗi bước 36°)
    pts_out, pts_in = [], []
    for i in range(6):
        deg = 180 - i * 36
        rad = math.radians(deg)
        pts_out.append((cx + R_out * math.cos(rad), cy - R_out * math.sin(rad)))
        pts_in .append((cx + R_in  * math.cos(rad), cy - R_in  * math.sin(rad)))

    # 5 cung màu
    arcs = ""
    for i in range(5):
        x1o, y1o = pts_out[i]
        x2o, y2o = pts_out[i + 1]
        x1i, y1i = pts_in[i]
        x2i, y2i = pts_in[i + 1]
        arcs += (
            f'<path d="M{x1o:.1f},{y1o:.1f} A{R_out},{R_out} 0 0,1 {x2o:.1f},{y2o:.1f} '
            f'L{x2i:.1f},{y2i:.1f} A{R_in},{R_in} 0 0,0 {x1i:.1f},{y1i:.1f} Z" '
            f'fill="{AQI_COLORS[i + 1]}"/>'
        )

    # Đường kẻ trắng ngăn cách
    seps = ""
    for x, y in pts_out:
        seps += (
            f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" '
            f'stroke="#f9fafb" stroke-width="3"/>'
        )

    # Label ngoài cung, mid-angle từng mức
    label_svg = ""
    for i in range(5):
        mid_deg = 180 - (i + 0.5) * 36
        rad = math.radians(mid_deg)
        lx = cx + label_R * math.cos(rad)
        ly = cy - label_R * math.sin(rad)
        fw = "800" if (i + 1) == active_level else "700"
        label_svg += (
            f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="middle" '
            f'dominant-baseline="middle" font-size="13" font-weight="{fw}" '
            f'fill="{AQI_COLORS[i + 1]}" font-family="sans-serif">'
            f'{aqi_label(i + 1)}</text>'
        )

    # Kim trỏ vào giữa mức active
    mid_rad = math.radians(180 - (active_level - 0.5) * 36)
    tx = cx + needle_R * math.cos(mid_rad)
    ty = cy - needle_R * math.sin(mid_rad)
    perp = mid_rad + math.pi / 2
    bx1 = cx + needle_base * math.cos(perp)
    by1 = cy - needle_base * math.sin(perp)
    bx2 = cx - needle_base * math.cos(perp)
    by2 = cy + needle_base * math.sin(perp)
    needle = (
        f'<polygon points="{tx:.1f},{ty:.1f} {bx1:.1f},{by1:.1f} {bx2:.1f},{by2:.1f}" '
        f'fill="white" stroke="#9ca3af" stroke-width="1.5"/>'
    )

    hub = (
        f'<circle cx="{cx}" cy="{cy}" r="18" fill="#374151"/>'
        f'<circle cx="{cx}" cy="{cy}" r="11" fill="#e5e7eb"/>'
    )

    level_label = (
        f'<text x="{cx}" y="{cy + 37}" text-anchor="middle" '
        f'font-size="18" font-weight="800" '
        f'fill="{AQI_COLORS[active_level]}" font-family="sans-serif">'
        f'{aqi_label(active_level)}</text>'
    )

    return (
        '<div style="display:flex;justify-content:center;margin:8px 0;">'
        '<svg width="100%" style="max-width:580px;height:auto" viewBox="0 0 580 285" xmlns="http://www.w3.org/2000/svg">'
        + arcs + seps + label_svg + needle + hub + level_label +
        '</svg></div>'
    )


def confidence_ring_html(pct: float, color: str) -> str:
    """Vòng tròn SVG hiển thị % độ tin cậy của dự đoán. Trả về HTML 1 dòng
    (không xuống dòng để không làm vỡ khối HTML trong st.markdown)."""
    import math

    r = 34
    circ = 2 * math.pi * r
    dash = circ * pct / 100
    return (
        '<div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex-shrink:0;">'
        '<div style="position:relative;width:88px;height:88px;">'
        '<svg width="88" height="88" viewBox="0 0 88 88" style="transform:rotate(-90deg)">'
        f'<circle cx="44" cy="44" r="{r}" fill="none" stroke="#eef0f2" stroke-width="9"/>'
        f'<circle cx="44" cy="44" r="{r}" fill="none" stroke="{color}" stroke-width="9" '
        f'stroke-linecap="round" stroke-dasharray="{dash:.1f} {circ:.1f}"/></svg>'
        '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;'
        f'font-size:22px;font-weight:800;color:#1f2937;">{pct:.0f}%</div></div>'
        '<div style="font-size:13px;font-weight:600;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">'
        f'{t("confidence")}</div></div>'
    )


def trend_arrow(current: float, previous: float):
    """(mũi tên, nhãn, css-class) so với giờ trước."""
    if pd.isna(current) or pd.isna(previous) or previous == 0:
        return "─", t("trend_stable"), "trend-flat"
    pct = (current - previous) / abs(previous)
    if pct > 0.02:
        return "▲", t("trend_rising"), "trend-up"
    if pct < -0.02:
        return "▼", t("trend_falling"), "trend-down"
    return "─", t("trend_stable"), "trend-flat"


# ── Dữ liệu ──────────────────────────────────────────────────────────────────
# show_spinner=False: chữ của spinner trong decorator cố định 1 ngôn ngữ, nên get_data() tự hiện spinner
@st.cache_data(ttl=25*60, show_spinner=False)
def load_data_from_api() -> pd.DataFrame:
    """Gọi GET /api/air-quality, trả DataFrame index=datetime (đã cộng 7h → GMT+7), cột aqi + 8 chất.
    Dữ liệu từ API đã được reindex lưới 1h + nội suy gap ngắn ở backend, FE không xử lý lại."""
    resp = requests.get(f"{API_BASE_URL.rstrip('/')}/api/air-quality", timeout=API_TIMEOUT)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    if df.empty:
        raise ValueError("err_empty_db")   # key trong language.py, get_data() dịch khi hiển thị
    # API trả giờ UTC → cộng 7 tiếng để hiển thị theo GMT+7
    df["datetime"] = pd.to_datetime(df["datetime"]) + pd.Timedelta(hours=TZ_OFFSET_HOURS)
    return df.set_index("datetime").sort_index()


def get_data() -> pd.DataFrame:
    """Lấy dữ liệu từ API, hiện lỗi thân thiện thay vì crash."""
    try:
        with st.spinner(t("loading")):
            return load_data_from_api()
    except requests.exceptions.ConnectionError:
        st.error(t("err_connect", url=API_BASE_URL))
        st.stop()
    except ValueError as e:
        st.error(t("err_empty_db") if str(e) == "err_empty_db" else t("err_fetch", e=e))
        st.stop()
    except Exception as e:
        st.error(t("err_fetch", e=e))
        st.stop()


def latest_row(df: pd.DataFrame) -> pd.Series:
    """Bản ghi mới nhất có nhãn AQI hợp lệ."""
    return df.dropna(subset=["aqi"]).iloc[-1]


def last_update_text() -> str:
    try:
        return latest_row(load_data_from_api()).name.strftime("%H:%M %d/%m/%Y") + f" ({TZ_LABEL})"
    except Exception:  # API chưa bật / DB trống
        return t("no_data_yet")


# ── Model ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=5 * 60, show_spinner=False)
def predict_next_hour():
    """Dự đoán lớp AQI (1..5) của giờ kế tiếp bằng cách gọi GET /api/predict/next-hour.
    Trả về (level, probs) hoặc (None, mã lỗi). Backend tự lấy 24h gần nhất từ DB.
    Cache 5 phút (kể cả kết quả lỗi); nút Refresh data sẽ xoá cache.
    Cache dùng chung mọi người xem nên chỉ trả mã lỗi, dịch bằng pred_error_text() lúc hiển thị."""
    try:
        resp = requests.get(f"{API_BASE_URL.rstrip('/')}/api/predict/next-hour", timeout=API_TIMEOUT)
    except requests.exceptions.RequestException:
        return None, "pred_unreachable"

    if resp.status_code == 422:
        return None, "pred_not_enough"
    if resp.status_code == 503:
        return None, "pred_not_ready"
    if resp.status_code != 200:
        return None, f"pred_http:{resp.status_code}"

    data = resp.json()
    return int(data["level"]), np.array(data["probs"])


def pred_error_text(code: str) -> str:
    """Mã lỗi của predict_next_hour() → câu thông báo theo ngôn ngữ hiện tại."""
    if code.startswith("pred_http:"):
        return t("pred_http", code=code.split(":", 1)[1])
    return t(code)
