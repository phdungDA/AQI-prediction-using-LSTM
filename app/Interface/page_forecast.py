# page_forecast.py
# Trang dự báo.
#  - Giờ kế tiếp (h+1): model LSTM thật (24 giờ lịch sử -> lớp AQI của giờ tiếp theo).
#  - h+2 .. h+24: model hiện chỉ dự đoán 1 bước, nên đây là ƯỚC LƯỢNG THAM KHẢO
#    theo quy luật trung vị từng giờ trong ngày của 14 ngày gần nhất,
#    KHÔNG phải đầu ra thật của model BiLSTM. Luôn ghi rõ điều này trên giao diện.

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQI_COLORS, AQI_LABELS, get_data, latest_row, predict_next_hour

IMPACT_NOTE = {
    "sáng sớm": "Giao thông còn thưa",
    "giờ cao điểm sáng": "Giao thông đông đúc",
    "trưa - chiều": "Bình thường",
    "giờ cao điểm chiều": "Giao thông đông đúc",
    "tối": "Bắt đầu lắng bụi",
    "đêm": "Có thể nghịch nhiệt, bụi tích tụ",
}


def _part_of_day(hour: int) -> str:
    if 5 <= hour < 7:
        return "sáng sớm"
    if 7 <= hour < 9:
        return "giờ cao điểm sáng"
    if 9 <= hour < 16:
        return "trưa - chiều"
    if 16 <= hour < 19:
        return "giờ cao điểm chiều"
    if 19 <= hour < 22:
        return "tối"
    return "đêm"


def _reference_24h(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    hist = df.loc[end - pd.Timedelta(days=14):end]
    aqi_by_hour = hist["aqi"].dropna().groupby(hist["aqi"].dropna().index.hour).median()
    pm_by_hour = hist["pm2_5"].dropna().groupby(hist["pm2_5"].dropna().index.hour).median()
    aqi_fallback = float(hist["aqi"].median()) if hist["aqi"].notna().any() else 1.0
    pm_fallback = float(hist["pm2_5"].median()) if hist["pm2_5"].notna().any() else 0.0

    times = pd.date_range(end + pd.Timedelta(hours=1), periods=24, freq="h")
    rows = []
    for t in times:
        rows.append({
            "time": t,
            "aqi": int(round(float(aqi_by_hour.get(t.hour, aqi_fallback)))),
            "pm25": float(pm_by_hour.get(t.hour, pm_fallback)),
        })
    return pd.DataFrame(rows)


def render():
    df = get_data()
    cur = latest_row(df)

    st.markdown('<div class="page-title">🔮 DỰ BÁO AQI 24 GIỜ TỚI</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">Mô hình: LSTM (h+1) · Ước lượng tham khảo (h+2 → h+24) · '
        f'Dữ liệu tới {cur.name.strftime("%H:%M %d/%m/%Y")} (UTC)</div>',
        unsafe_allow_html=True,
    )

    # ── Thẻ lớn: dự đoán h+1 từ model thật ───────────────────────────────
    level, probs = predict_next_hour(df.loc[:cur.name])
    next_time = cur.name + pd.Timedelta(hours=1)

    if level is None:
        st.warning(probs)
        return

    conf = float(max(probs)) * 100
    c1, c2 = st.columns([1, 2.2])
    with c1:
        st.markdown(
            f"""
            <div class="hero-aqi">
              <div class="lbl">CẤP AQI</div>
              <div class="val" style="color:{AQI_COLORS[level]}">{level}</div>
            </div>
            <span class="pbadge" style="background:{AQI_COLORS[level]}">● {AQI_LABELS[level]}</span>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div style="font-weight:700; font-size:15px; color:#374151;">
              AQI dự đoán (h+1, {next_time.strftime('%H:%M')})
            </div>
            <div style="font-size:13.5px; color:#4b5563; margin:6px 0;">
              Khuyến nghị: nhóm nhạy cảm (người cao tuổi, trẻ nhỏ, người mắc bệnh hô hấp)
              nên cân nhắc hạn chế vận động thể lực nặng ngoài trời kéo dài.
            </div>
            <span style="font-size:12.5px; color:#2e9e5b; font-weight:600;">
              ● Độ tin cậy h+1: {conf:.1f}%
            </span>
            """,
            unsafe_allow_html=True,
        )

    # ── Biểu đồ bậc thang 24h (h+1 thật + h+2..h+24 tham khảo) ──────────
    st.markdown(
        '<div class="section-h">Diễn biến cấp độ AQI dự báo trong 24 giờ tới (h+1 đến h+24)</div>',
        unsafe_allow_html=True,
    )
    st.caption("⚠️ Chỉ h+1 là đầu ra thật của model LSTM. Các mốc h+2→h+24 là ước lượng tham khảo "
               "theo trung vị từng giờ trong ngày của 14 ngày gần nhất — KHÔNG phải dự báo đa bước của model.")

    ref = _reference_24h(df, cur.name)
    times = [next_time] + list(ref["time"])
    levels = [level] + list(ref["aqi"])

    fig = go.Figure(go.Scatter(
        x=times, y=levels, mode="lines+markers", line_shape="hv",
        line=dict(color="#2e9e5b", width=2),
        marker=dict(size=6, color=[AQI_COLORS[int(v)] for v in levels]),
    ))
    for lvl in range(1, 6):
        fig.add_hrect(y0=lvl - 0.5, y1=lvl + 0.5, fillcolor=AQI_COLORS[lvl], opacity=0.08, line_width=0)
    fig.add_vline(x=next_time, line_dash="dot", line_color="#9ca3af",
                  annotation_text="Hiện tại +1h", annotation_position="top left")
    fig.update_yaxes(range=[0.5, 5.5], tickvals=[1, 2, 3, 4, 5],
                      ticktext=[f"{l} - {AQI_LABELS[l]}" for l in range(1, 6)])
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

    # ── Bảng chi tiết từng giờ ────────────────────────────────────────
    st.markdown('<div class="section-h">Bảng dữ liệu dự báo chi tiết từng giờ (24 mốc)</div>',
                unsafe_allow_html=True)

    rows = [{
        "Mốc dự báo": f"h+1 · {next_time.strftime('%H:%M')}",
        "Cấp độ (1-5)": level,
        "Phân loại": AQI_LABELS[level],
        "PM2.5 dự kiến (µg/m³)": round(float(df["pm2_5"].iloc[-1]), 1) if pd.notna(df["pm2_5"].iloc[-1]) else None,
        "Ghi chú": "Đầu ra thật của model LSTM",
    }]
    for i, r in ref.iterrows():
        pod = _part_of_day(r["time"].hour)
        rows.append({
            "Mốc dự báo": f"h+{i + 2} · {r['time'].strftime('%H:%M')}",
            "Cấp độ (1-5)": int(r["aqi"]),
            "Phân loại": AQI_LABELS[int(r["aqi"])],
            "PM2.5 dự kiến (µg/m³)": round(float(r["pm25"]), 1),
            "Ghi chú": IMPACT_NOTE[pod],
        })
    table = pd.DataFrame(rows)
    st.dataframe(table, width="stretch", hide_index=True)

    worst = int(table["Cấp độ (1-5)"].max())
    st.info(f"Mức cao nhất ước tính trong 24h tới: **{worst} – {AQI_LABELS[worst]}** "
            "(dựa trên quy luật lịch sử, không phải dự báo đa bước của model).")
