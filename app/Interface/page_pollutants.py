# page_pollutants.py
# Trang Chất ô nhiễm: xem chi tiết từng chất trong khoảng thời gian chọn.

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import POLLUTANT_NAMES, POLLUTANT_ORDER, get_data, pollutant_level, AQI_COLORS, AQI_LABELS


def render():
    df = get_data()
    d_min, d_max_ts = df.index.min(), df.index.max()

    st.markdown('<div class="page-title">🧪 POLLUTANTS</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Track each pollutant over time</div>',
                unsafe_allow_html=True)

    days = st.radio("Time range", [1, 7, 30], index=1,
                     format_func=lambda d: {1: "Past 24h", 7: "7 days", 30: "30 days"}[d],
                     horizontal=True)
    start = max(d_min, d_max_ts - pd.Timedelta(days=days))
    view = df.loc[start:d_max_ts]

    cols = st.columns(4)
    for i, key in enumerate(POLLUTANT_ORDER + ["no", "nh3"]):
        cur_val = view[key].dropna().iloc[-1] if view[key].notna().any() else float("nan")
        plvl = pollutant_level(key, cur_val) if pd.notna(cur_val) else 1
        with cols[i % 4]:
            st.markdown(
                f"""<div class="pcard"><div class="top">{POLLUTANT_NAMES[key]}
                <span class="pbadge" style="background:{AQI_COLORS[plvl]}">● {AQI_LABELS[plvl]}</span></div>
                <div class="pval">{cur_val:.1f} <span style="font-size:12px;color:#9ca3af">µg/m³</span></div>
                </div>""" if pd.notna(cur_val) else
                f"""<div class="pcard"><div class="top">{POLLUTANT_NAMES[key]}</div>
                <div class="pval" style="color:#9ca3af;font-size:14px">No data</div></div>""",
                unsafe_allow_html=True,
            )
            fig = go.Figure(go.Scatter(x=view.index, y=view[key], mode="lines",
                                        line=dict(color=AQI_COLORS[plvl], width=1.8),
                                        connectgaps=False, hoverinfo="skip"))
            fig.update_layout(height=120, margin=dict(l=10, r=10, t=6, b=6))
            st.plotly_chart(fig, width="stretch", key=f"poll_{key}")
