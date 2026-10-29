# language.py
# Song ngữ Việt / Anh cho toàn bộ giao diện.
# Ngôn ngữ lưu trong st.session_state["lang"] nên mỗi người xem có lựa chọn riêng.
# Thêm chữ mới: thêm 1 key vào TEXT với cả "vi" và "en", rồi gọi t("key", ...).

import pandas as pd
import streamlit as st

DEFAULT_LANG = "vi"   # app dành cho người Hà Nội nên mặc định tiếng Việt

AQI_LABEL_TEXT = {
    "en": {1: "Good", 2: "Fair", 3: "Moderate", 4: "Poor", 5: "Very Poor"},
    "vi": {1: "Tốt", 2: "Khá", 3: "Trung bình", 4: "Kém", 5: "Rất kém"},
}

AQI_ADVICE_TEXT = {
    "en": {
        1: "Clean air, enjoy outdoor activities freely.",
        2: "Acceptable air quality, sensitive people should pay attention.",
        3: "Sensitive people should limit strenuous outdoor activity.",
        4: "Limit going outside, wear a mask when needed.",
        5: "Avoid going outside as much as possible, use an air purifier indoors.",
    },
    "vi": {
        1: "Không khí trong lành, thoải mái hoạt động ngoài trời.",
        2: "Chất lượng không khí chấp nhận được, người nhạy cảm nên chú ý.",
        3: "Người nhạy cảm nên hạn chế vận động mạnh ngoài trời.",
        4: "Hạn chế ra ngoài, đeo khẩu trang khi cần.",
        5: "Tránh ra ngoài nhiều nhất có thể, dùng máy lọc không khí trong nhà.",
    },
}

# Bảng giải thích thang AQI: (ý nghĩa, khuyến nghị) cho từng mức
SCALE_INFO_TEXT = {
    "en": {
        1: ("Clean air, no health risk.", "Enjoy normal outdoor activities."),
        2: ("Acceptable air quality; sensitive groups may feel mild effects.",
            "Sensitive groups should limit prolonged outdoor exertion."),
        3: ("Sensitive groups (children, elderly, respiratory conditions) may be affected.",
            "Sensitive groups should stay indoors; others wear a mask outdoors."),
        4: ("Everyone may start to experience health effects; sensitive groups more severely.",
            "Limit outdoor exposure. Wear an N95 mask. Run air purifiers."),
        5: ("Health warning: the entire population may be seriously affected.",
            "Avoid going outside. N95 mask mandatory. Run air purifiers at max."),
    },
    "vi": {
        1: ("Không khí sạch, không có nguy cơ cho sức khoẻ.", "Hoạt động ngoài trời bình thường."),
        2: ("Chấp nhận được; nhóm nhạy cảm có thể bị ảnh hưởng nhẹ.",
            "Nhóm nhạy cảm nên hạn chế vận động lâu ngoài trời."),
        3: ("Nhóm nhạy cảm (trẻ em, người già, người bệnh hô hấp) có thể bị ảnh hưởng.",
            "Nhóm nhạy cảm nên ở trong nhà; người khác đeo khẩu trang khi ra ngoài."),
        4: ("Mọi người bắt đầu bị ảnh hưởng sức khoẻ; nhóm nhạy cảm bị nặng hơn.",
            "Hạn chế ra ngoài. Đeo khẩu trang N95. Bật máy lọc không khí."),
        5: ("Cảnh báo sức khoẻ: toàn bộ người dân có thể bị ảnh hưởng nghiêm trọng.",
            "Tránh ra ngoài. Bắt buộc đeo N95. Bật máy lọc không khí ở mức tối đa."),
    },
}

VI_WEEKDAYS = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]

TEXT = {
    # ── Sidebar ───────────────────────────────────────────────────────────
    "app_sub":        {"en": "Hanoi air quality", "vi": "Chất lượng không khí"},
    "switch_lang":    {"en": "Tiếng Việt", "vi": "English"},
    "nav_Dashboard":  {"en": "Dashboard", "vi": "Tổng quan"},
    "nav_History":    {"en": "History", "vi": "Lịch sử"},
    "nav_Pollutants": {"en": "Pollutants", "vi": "Chất ô nhiễm"},
    "latest_data":    {"en": "Latest data", "vi": "Dữ liệu mới nhất"},
    "refresh":        {"en": "Refresh data", "vi": "Làm mới dữ liệu"},
    "auto_refresh":   {"en": "Auto update on new data", "vi": "Tự cập nhật khi có dữ liệu mới"},
    "autorefresh_missing": {"en": "Auto refresh requires `pip install streamlit-autorefresh`",
                            "vi": "Cần cài `pip install streamlit-autorefresh` để tự làm mới"},
    "data_source":    {"en": "Data source: OpenWeather", "vi": "Nguồn dữ liệu: OpenWeather"},
    "no_data_yet":    {"en": "No data yet", "vi": "Chưa có dữ liệu"},

    # ── Dữ liệu / lỗi ─────────────────────────────────────────────────────
    "loading":        {"en": "Loading data from API...", "vi": "Đang tải dữ liệu từ API..."},
    "err_connect":    {"en": "Cannot connect to the FastAPI backend at `{url}`.\n\n"
                             "Start it with: `uvicorn app.api:app --port 8000` (run from the project root).",
                       "vi": "Không kết nối được tới backend FastAPI tại `{url}`.\n\n"
                             "Hãy chạy: `uvicorn app.api:app --port 8000` (ở thư mục gốc project)."},
    "err_fetch":      {"en": "Error while fetching data from the API: {e}",
                       "vi": "Lỗi khi lấy dữ liệu từ API: {e}"},
    "err_empty_db":   {"en": "The database has no data yet.", "vi": "Cơ sở dữ liệu chưa có dữ liệu."},
    "pred_unreachable": {"en": "Could not reach the prediction API.",
                         "vi": "Không kết nối được tới API dự đoán."},
    "pred_not_enough":  {"en": "Not enough data in the last 24 hours to make a prediction.",
                         "vi": "24 giờ gần nhất thiếu dữ liệu nên chưa dự đoán được."},
    "pred_not_ready":   {"en": "Prediction service is not ready (model not loaded or no data yet). "
                               "Check /health on the API.",
                         "vi": "Dịch vụ dự đoán chưa sẵn sàng (chưa load được model hoặc chưa có dữ liệu). "
                               "Kiểm tra /health trên API."},
    "pred_http":        {"en": "Prediction API error (HTTP {code}).",
                         "vi": "API dự đoán bị lỗi (HTTP {code})."},

    # ── Thành phần dùng chung ─────────────────────────────────────────────
    "confidence":     {"en": "confidence", "vi": "độ tin cậy"},
    "trend_rising":   {"en": "rising", "vi": "tăng"},
    "trend_falling":  {"en": "falling", "vi": "giảm"},
    "trend_stable":   {"en": "stable", "vi": "ổn định"},
    "no_data":        {"en": "No data", "vi": "Không có dữ liệu"},

    # ── Dashboard ─────────────────────────────────────────────────────────
    "dash_title":     {"en": "CURRENT AIR QUALITY", "vi": "CHẤT LƯỢNG KHÔNG KHÍ HIỆN TẠI"},
    "dash_sub":       {"en": "Hanoi · Updated: {time} (GMT+7)", "vi": "Hà Nội · Cập nhật: {time} (GMT+7)"},
    "recommendation": {"en": "RECOMMENDATION", "vi": "KHUYẾN NGHỊ"},
    "main_pollutant": {"en": "Main pollutant", "vi": "Chất ô nhiễm chính"},
    "indicators_h":   {"en": "Main Air Quality Indicators · OpenWeather standard",
                       "vi": "Chỉ số chất lượng không khí chính · Chuẩn OpenWeather"},
    "no_prev_hour":   {"en": "No data for the previous hour", "vi": "Không có dữ liệu giờ trước"},
    "next_hour":      {"en": "Next hour", "vi": "Giờ tới"},
    "change_worse":   {"en": "Getting worse: {a} → {b}.", "vi": "Xấu đi: {a} → {b}."},
    "change_better":  {"en": "Improving: {a} → {b}.", "vi": "Cải thiện: {a} → {b}."},
    "change_same":    {"en": "Staying at {a}.", "vi": "Giữ ở mức {a}."},
    "not_enough_badge": {"en": "Not enough data", "vi": "Chưa đủ dữ liệu"},
    "cig_head":       {"en": "🚬 In cigarettes", "vi": "🚬 Quy ra thuốc lá"},
    "cig_one":        {"en": "≈ {n} cigarette / day", "vi": "≈ {n} điếu thuốc / ngày"},
    "cig_many":       {"en": "≈ {n} cigarettes / day", "vi": "≈ {n} điếu thuốc / ngày"},
    "cig_note":       {"en": "Breathing the current air for a whole day is like smoking this many cigarettes.",
                       "vi": "Hít thở không khí như hiện tại suốt 1 ngày tương đương hút ngần ấy điếu thuốc."},
    "no_pm_now":      {"en": "No PM2.5 reading for the latest hour.",
                       "vi": "Giờ mới nhất không có số đo PM2.5."},
    # Viết cho người không biết gì về IT/học máy: không dùng thuật ngữ (LSTM, softmax, chuẩn hoá...)
    "conf_h":         {"en": "How is the confidence worked out?", "vi": "Độ tin cậy được tính thế nào?"},
    "step1_t":        {"en": "Look back 24 hours", "vi": "Xem lại 24 giờ qua"},
    "step1_d":        {"en": "We take the hourly readings of fine dust and other polluting gases "
                             "from {a} to {b}.",
                       "vi": "Lấy số đo bụi mịn và các khí ô nhiễm của từng giờ, từ {a} đến {b}."},
    "step2_t":        {"en": "Put on one scale", "vi": "Đưa về cùng thước đo"},
    "step2_d":        {"en": "Each pollutant has its own units and range, so all are converted to the "
                             "same scale to be compared fairly.",
                       "vi": "Mỗi chất có đơn vị và độ lớn khác nhau, nên được quy về cùng một thang "
                             "để so sánh công bằng."},
    "step3_t":        {"en": "Compare with the past", "vi": "So với những lần trước"},
    "step3_d":        {"en": "The model has learned from past years of Hanoi air data. It compares the "
                             "last 24 hours with similar situations before, then splits 100% across the 5 levels.",
                       "vi": "Mô hình đã học từ dữ liệu không khí Hà Nội các năm trước. Nó so 24 giờ vừa qua "
                             "với những lần diễn biến tương tự, rồi chia 100% cho 5 mức."},
    "step4_t":        {"en": "Pick the biggest share", "vi": "Chọn mức nhiều % nhất"},
    "step4_d":        {"en": "The level with the biggest share is the forecast ({lvl}); its share is the "
                             "confidence ({pct}%). Higher means the model is more sure, not that it is "
                             "guaranteed to be right.",
                       "vi": "Mức được chia nhiều nhất là dự đoán ({lvl}), phần trăm của mức đó là độ tin cậy "
                             "({pct}%). Càng cao thì mô hình càng chắc, nhưng không có nghĩa là chắc chắn đúng."},
    "model_output":   {"en": "How the model split 100% for the next hour",
                       "vi": "Mô hình chia 100% cho giờ tới như sau"},
    "scale_h":        {"en": "AQI Scale Reference", "vi": "Bảng giải thích thang AQI"},
    "th_level":       {"en": "Level", "vi": "Mức"},
    "th_category":    {"en": "Category", "vi": "Phân loại"},
    "th_meaning":     {"en": "Meaning", "vi": "Ý nghĩa"},
    "th_rec":         {"en": "Recommendation", "vi": "Khuyến nghị"},

    # ── History ───────────────────────────────────────────────────────────
    "hist_title":     {"en": "HISTORICAL MONITORING REPORT", "vi": "BÁO CÁO LỊCH SỬ QUAN TRẮC"},
    "hist_sub":       {"en": "Hanoi air quality history · Data from {a} to {b} (GMT+7)",
                       "vi": "Lịch sử chất lượng không khí Hà Nội · Dữ liệu từ {a} đến {b} (GMT+7)"},
    "date_range":     {"en": "Date range", "vi": "Khoảng ngày"},
    "pick_both":      {"en": "Select both a start date and an end date.",
                       "vi": "Hãy chọn cả ngày bắt đầu và ngày kết thúc."},
    "no_data_range":  {"en": "No data in this range.", "vi": "Không có dữ liệu trong khoảng này."},
    "summary_h":      {"en": "Summary statistics for the monitoring period",
                       "vi": "Thống kê tổng hợp trong khoảng đã chọn"},
    "most_polluted":  {"en": "Most polluted day", "vi": "Ngày ô nhiễm nhất"},
    "cleanest":       {"en": "Cleanest day", "vi": "Ngày sạch nhất"},
    "not_enough_period": {"en": "Not enough data in this period",
                          "vi": "Không đủ dữ liệu trong khoảng này"},
    "none":           {"en": "None", "vi": "Không có"},
    "no_poor_hours":  {"en": "No hour reached {poor} in this period",
                       "vi": "Không có giờ nào ở mức {poor} trong khoảng này"},
    "polluted_sub":   {"en": "{n} of {h} hours {poor} or worse",
                       "vi": "{n}/{h} giờ ở mức {poor} trở lên"},
    "no_clean_day":   {"en": "No day at {good}/{fair} level",
                       "vi": "Không có ngày nào ở mức {good}/{fair}"},
    "clean_sub":      {"en": "{n} of {h} hours {good} or {fair}",
                       "vi": "{n}/{h} giờ ở mức {good} hoặc {fair}"},
    "ties_one":       {"en": " · +{n} more day tied", "vi": " · +{n} ngày khác bằng điểm"},
    "ties_many":      {"en": " · +{n} more days tied", "vi": " · +{n} ngày khác bằng điểm"},
    "sensor":         {"en": "Sensor reliability", "vi": "Độ đầy đủ dữ liệu"},
    "sensor_val":     {"en": "{m}/{t} hours missing data", "vi": "{m}/{t} giờ thiếu dữ liệu"},
    "sensor_sub":     {"en": "Series completeness: {p}%", "vi": "Mức đầy đủ của chuỗi: {p}%"},
    "dist_h":         {"en": "Air quality distribution over time", "vi": "Phân bố chất lượng không khí"},
    "pie_label":      {"en": "Level {l}: {name}", "vi": "Mức {l}: {name}"},
    "pie_hover":      {"en": "%{label}<br>%{value} hours (%{percent})<extra></extra>",
                       "vi": "%{label}<br>%{value} giờ (%{percent})<extra></extra>"},
    "daily_h":        {"en": "AQI · Daily average (hover for hourly detail)",
                       "vi": "AQI · Trung bình theo ngày (rê chuột để xem từng giờ)"},
    "daily_hover":    {"en": "<b>%{x|%d/%m/%Y}</b> · Avg level: %{y:.2f}<br><br>%{customdata}<extra></extra>",
                       "vi": "<b>%{x|%d/%m/%Y}</b> · Mức trung bình: %{y:.2f}<br><br>%{customdata}<extra></extra>"},
    "trend_select":   {"en": "Parameters shown on the trend chart",
                       "vi": "Các chất hiển thị trên biểu đồ xu hướng"},
    "trend_h":        {"en": "Concentration trend of selected pollutants",
                       "vi": "Xu hướng nồng độ các chất đã chọn"},
    "select_one":     {"en": "Select at least one parameter to display the chart.",
                       "vi": "Hãy chọn ít nhất một chất để hiển thị biểu đồ."},
    "stored_caption": {"en": "Data is stored automatically every 30 minutes.",
                       "vi": "Dữ liệu được tự động lưu mỗi 30 phút."},
    "download_csv":   {"en": "Download CSV", "vi": "Tải CSV"},

    # ── Pollutants ────────────────────────────────────────────────────────
    "poll_title":     {"en": "POLLUTANTS", "vi": "CHẤT Ô NHIỄM"},
    "poll_sub":       {"en": "Track each pollutant over time", "vi": "Theo dõi từng chất ô nhiễm theo thời gian"},
    "time_range":     {"en": "Time range", "vi": "Khoảng thời gian"},
    "range_1":        {"en": "Past 24h", "vi": "24 giờ qua"},
    "range_7":        {"en": "7 days", "vi": "7 ngày"},
    "range_30":       {"en": "30 days", "vi": "30 ngày"},
}


def lang() -> str:
    """Ngôn ngữ hiện tại của người đang xem: "vi" hoặc "en"."""
    return st.session_state.get("lang", DEFAULT_LANG)


def t(key: str, **kw) -> str:
    """Chữ theo ngôn ngữ hiện tại. Chỉ gọi .format khi có tham số,
    để chuỗi chứa ngoặc nhọn của plotly (vd "%{label}") không bị lỗi."""
    s = TEXT[key][lang()]
    return s.format(**kw) if kw else s


def aqi_label(level: int) -> str:
    return AQI_LABEL_TEXT[lang()][int(level)]


def aqi_advice(level: int) -> str:
    return AQI_ADVICE_TEXT[lang()][int(level)]


def scale_info(level: int) -> tuple:
    return SCALE_INFO_TEXT[lang()][int(level)]


def fmt_day(ts: pd.Timestamp) -> str:
    """Thứ + ngày/tháng, vd "Sun 18/10" hoặc "CN 18/10"."""
    if lang() == "vi":
        return f"{VI_WEEKDAYS[ts.weekday()]} {ts:%d/%m}"
    return f"{ts:%a %d/%m}"
