# page_dashboard.py
# Trang Dashboard: tình trạng không khí hiện tại.

import plotly.graph_objects as go
import streamlit as st

from common import (
    AQI_ADVICE, AQI_COLORS, AQI_LABELS, POLLUTANT_NAMES, POLLUTANT_ORDER,
    get_data, latest_row, pollutant_level, predict_next_hour,
    scale_bar_html, sparkline_fig, trend_arrow,
)


def render():
    df = get_data()
    row = latest_row(df)
    level = int(row["aqi"])
    worst_pollutant = max(POLLUTANT_ORDER, key=lambda c: pollutant_level(c, row[c]))

    st.markdown('<div class="page-title">CURRENT AIR QUALITY</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">📍 Hanoi · Updated: '
        f'{row.name.strftime("%H:%M %d/%m/%Y")} (GMT+7)</div>',
        unsafe_allow_html=True,
    )

    # ── Thẻ AQI lớn ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns([1.1, 2.3, 1.1])
    with c1:
        st.markdown(
            f"""
            <div class="hero-aqi">
              <div class="lbl">AIR QUALITY<br>INDEX (AQI)</div>
              <div class="val">{level}<span style="font-size:18px;color:#9ca3af"> / Scale of 5</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <span class="hero-badge" style="background:{AQI_COLORS[level]}">
              ● {AQI_LABELS[level]}
            </span>
            <div class="hero-note">{AQI_ADVICE[level]}</div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f'<div class="hero-right">Main pollutant:<br>'
            f'<b>{POLLUTANT_NAMES[worst_pollutant]}</b></div>',
            unsafe_allow_html=True,
        )
    st.markdown(scale_bar_html(level), unsafe_allow_html=True)

    # ── 6 thẻ chỉ số khí quyển chi tiết ──────────────────────────────────
    st.markdown('<div class="section-h">Detailed atmospheric indicators · QCVN 05:2023/BTNMT standard</div>'
                f'<div class="section-sub">Air indicators of the latest hour ({row.name.strftime("%H:%M %d/%m")}, GMT+7) · trend vs. previous hour</div>',
                unsafe_allow_html=True)

    hist24 = df.tail(24)
    cols = st.columns(len(POLLUTANT_ORDER))
    for col_box, key in zip(cols, POLLUTANT_ORDER):
        plvl = pollutant_level(key, row[key])
        color = AQI_COLORS[plvl]
        prev = df[key].shift(1).loc[row.name]   # giá trị 1 giờ trước giờ mới nhất
        arrow, trend_lbl, trend_cls = trend_arrow(row[key], prev)
        with col_box:
            st.markdown(
                f"""
                <div class="pcard">
                  <div class="top">{POLLUTANT_NAMES[key]}
                    <span class="pbadge" style="background:{color}">● {AQI_LABELS[plvl]}</span>
                  </div>
                  <div class="pval">{row[key]:.1f} <span style="font-size:14px;color:#9ca3af">µg/m³</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            series = hist24[key].interpolate(limit_direction="both")
            st.plotly_chart(sparkline_fig(series, color), width="stretch",
                             config={"displayModeBar": False}, key=f"spark_{key}")
            st.markdown(
                f'<div class="pfoot"><span class="{trend_cls}">{arrow} {trend_lbl}</span></div>',
                unsafe_allow_html=True,
            )

    # ── 2 khối gợi ý: giờ kế tiếp / tệ nhất 24h ──────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    i1, i2 = st.columns(2)

    next_level, probs = predict_next_hour(df)
    with i1:
        if next_level is not None:
            badge = (f'<span class="pbadge" style="background:{AQI_COLORS[next_level]}">'
                      f'● {AQI_LABELS[next_level]} ({next_level})</span>')
            note = ("Expected to stay the same, no major change." if next_level == level
                    else "The model predicts a change in AQI level compared to now.")
        else:
            badge = '<span class="pbadge" style="background:#9ca3af">● Not enough data</span>'
            note = probs  # thông báo lỗi
        st.markdown(
            f"""
            <div class="info-card">
              <div class="ihead">🕐 Next hour</div>
              {badge}
              <div class="inote">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    worst24 = df.tail(24)
    worst_idx = worst24["aqi"].idxmax() if worst24["aqi"].notna().any() else None
    with i2:
        if worst_idx is not None:
            wl = int(worst24.loc[worst_idx, "aqi"])
            badge = (f'<span class="pbadge" style="background:{AQI_COLORS[wl]}">'
                      f'● {AQI_LABELS[wl]} ({wl})</span>')
            note = f"Peak at {worst_idx.strftime('%H:%M %d/%m')} (past 24 hours)."
        else:
            badge = '<span class="pbadge" style="background:#9ca3af">● Not enough data</span>'
            note = "Not enough AQI data in the past 24 hours."
        st.markdown(
            f"""
            <div class="info-card">
              <div class="ihead">⚠️ Worst in past 24h &nbsp; <span style="color:#9ca3af">Pollution peak alert</span></div>
              {badge}
              <div class="inote">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
