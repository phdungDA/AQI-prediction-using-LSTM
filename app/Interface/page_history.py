# page_history.py
# Trang lịch sử: bộ lọc nhanh, thống kê tổng hợp, phân bố AQI, xu hướng.

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQI_COLORS, AQI_LABELS, POLLUTANT_NAMES, get_data

PRESETS = {"24 giờ qua": 1, "7 ngày": 7, "30 ngày": 30}
QCVN_PM25 = 25  # ngưỡng QCVN 05:2023 cho PM2.5 trung bình 24h, µg/m³


def render():
    df = get_data()
    d_min, d_max_ts = df.index.min(), df.index.max()
    d_min_date, d_max_date = d_min.date(), d_max_ts.date()

    st.markdown('<div class="page-title">⏱ BÁO CÁO QUAN TRẮC LỊCH SỬ</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">📍 Lịch sử chất lượng không khí · '
        f'Dữ liệu từ {d_min.strftime("%d/%m/%Y")} đến {d_max_ts.strftime("%d/%m/%Y")} (UTC)</div>',
        unsafe_allow_html=True,
    )

    # ── Bộ lọc ──────────────────────────────────────────────────────────
    if "hist_range" not in st.session_state:
        st.session_state.hist_range = (
            max(d_min_date, (d_max_ts - pd.Timedelta(days=7)).date()), d_max_date,
        )

    c1, c2 = st.columns([3, 2])
    with c1:
        rng = st.date_input(
            "Khoảng ngày", value=st.session_state.hist_range,
            min_value=d_min_date, max_value=d_max_date, key="hist_date_input",
        )
    with c2:
        st.write("")  # căn hàng với date_input
        p1, p2, p3 = st.columns(3)
        for col_box, (label, days) in zip((p1, p2, p3), PRESETS.items()):
            if col_box.button(label, width="stretch", key=f"preset_{days}"):
                st.session_state.hist_range = (
                    max(d_min_date, (d_max_ts - pd.Timedelta(days=days)).date()), d_max_date,
                )
                st.rerun()

    chosen = st.multiselect(
        "Thông số hiển thị trên biểu đồ xu hướng", options=list(POLLUTANT_NAMES),
        default=["pm2_5"], format_func=lambda k: POLLUTANT_NAMES[k],
    )

    if not isinstance(rng, tuple) or len(rng) != 2:
        st.info("Chọn đủ ngày bắt đầu và ngày kết thúc.")
        return
    start, end = rng
    view = df.loc[str(start):str(end)]
    if view.empty:
        st.warning("Không có dữ liệu trong khoảng này.")
        return

    # ── Thẻ thống kê tổng hợp ─────────────────────────────────────────
    aqi_view = view["aqi"].dropna()
    total_hours = len(view)
    missing_hours = int(view["aqi"].isna().sum())
    complete_pct = 100 * (1 - missing_hours / total_hours) if total_hours else 0

    st.markdown('<div class="section-h">Thống kê tổng hợp giai đoạn quan trắc</div>', unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    with m1:
        if len(aqi_view):
            avg_lvl = float(aqi_view.mean())
            avg_txt, avg_sub = f"{avg_lvl:.1f} / 5.0", AQI_LABELS.get(round(avg_lvl), "—")
        else:
            avg_txt, avg_sub = "— / 5.0", "Không có dữ liệu"
        st.markdown(
            f"""<div class="metric-card"><div class="mlbl">Chỉ số trung bình</div>
            <div class="mval">{avg_txt}</div>
            <div class="msub">{avg_sub}</div></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        if len(aqi_view):
            peak_idx = aqi_view.idxmax()
            peak_lvl = int(aqi_view.loc[peak_idx])
            sub = f"Ghi nhận lúc {peak_idx.strftime('%H:%M %d/%m')}"
        else:
            peak_lvl, sub = 0, "Không có dữ liệu"
        peak_color = AQI_COLORS.get(peak_lvl, "#9ca3af")
        peak_label = AQI_LABELS.get(peak_lvl, "—")
        st.markdown(
            f"""<div class="metric-card"><div class="mlbl">Chỉ số cực đại (Đỉnh ô nhiễm)</div>
            <div class="mval">{peak_lvl} <span style="font-size:13px;color:{peak_color}">
            ● {peak_label}</span></div>
            <div class="msub">{sub}</div></div>""",
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""<div class="metric-card"><div class="mlbl">Độ tin cậy cảm biến</div>
            <div class="mval">{missing_hours}/{total_hours} giờ thiếu dữ liệu</div>
            <div class="msub">Tỷ lệ hoàn chỉnh chuỗi: {complete_pct:.1f}%</div></div>""",
            unsafe_allow_html=True,
        )

    # ── Phân bố chất lượng không khí theo thời gian ─────────────────────
    st.markdown('<div class="section-h">Phân bố chất lượng không khí theo thời gian</div>', unsafe_allow_html=True)
    hourly = aqi_view.astype(int)
    dist = hourly.value_counts().reindex(range(1, 6), fill_value=0)
    total = max(int(dist.sum()), 1)
    segs = "".join(
        f'<div style="width:{100*dist[l]/total:.2f}%; background:{AQI_COLORS[l]}"></div>'
        for l in range(1, 6) if dist[l] > 0
    )
    st.markdown(f'<div class="dist-bar">{segs}</div>', unsafe_allow_html=True)
    legend = "".join(
        f'<span><span class="dot" style="background:{AQI_COLORS[l]}"></span>'
        f'Mức {l}: {AQI_LABELS[l]} ({100*dist[l]/total:.0f}%)</span>'
        for l in range(1, 6)
    )
    st.markdown(f'<div class="dist-legend">{legend}</div>', unsafe_allow_html=True)

    # ── Biểu đồ AQI dạng bậc thang ───────────────────────────────────────
    st.markdown('<div class="section-h">AQI · Chỉ số chất lượng không khí theo giờ</div>', unsafe_allow_html=True)
    aqi_line = view["aqi"]
    fig1 = go.Figure(go.Scatter(
        x=aqi_line.index, y=aqi_line.values, mode="lines", line_shape="hv",
        line=dict(color="#374151", width=1.6), connectgaps=False,
    ))
    for lvl in range(1, 6):
        fig1.add_hrect(y0=lvl - 0.5, y1=lvl + 0.5, fillcolor=AQI_COLORS[lvl], opacity=0.10, line_width=0)
    fig1.update_yaxes(range=[0.5, 5.5], tickvals=[1, 2, 3, 4, 5],
                       ticktext=[f"{l} - {AQI_LABELS[l]}" for l in range(1, 6)])
    fig1.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig1, width="stretch")

    # ── PM2.5 với ngưỡng QCVN ────────────────────────────────────────────
    st.markdown('<div class="section-h">PM2.5 (µg/m³) · Nồng độ bụi mịn</div>', unsafe_allow_html=True)
    pm = view["pm2_5"]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=pm.index, y=pm.values, mode="lines", name="PM2.5 thực đo",
                               line=dict(color="#2e9e5b", width=2), fill="tozeroy",
                               fillcolor="rgba(46,158,91,0.12)", connectgaps=False))
    fig2.add_hline(y=QCVN_PM25, line_dash="dot", line_color="#d64545",
                    annotation_text=f"Ngưỡng QCVN: {QCVN_PM25} µg/m³", annotation_position="top left")
    fig2.update_layout(height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, width="stretch")

    # ── Xu hướng nhiều chất ô nhiễm (tuỳ chọn ở trên) ────────────────────
    if chosen:
        st.markdown('<div class="section-h">Xu hướng nồng độ các chất đã chọn</div>', unsafe_allow_html=True)
        fig3 = go.Figure()
        for key in chosen:
            fig3.add_trace(go.Scatter(x=view.index, y=view[key], mode="lines",
                                       name=POLLUTANT_NAMES[key], connectgaps=False))
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                            yaxis_title="µg/m³")
        st.plotly_chart(fig3, width="stretch")

    # ── Tải xuống ───────────────────────────────────────────────────────
    st.caption("Dữ liệu được lưu trữ tự động mỗi 60 phút.")
    st.download_button(
        "⬇️ Tải CSV", data=view.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"aqi_{start}_{end}.csv", mime="text/csv",
    )
