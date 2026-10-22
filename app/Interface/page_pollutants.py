# page_pollutants.py
# Trang Chất ô nhiễm: xem chi tiết từng chất trong khoảng thời gian chọn.

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import POLLUTANT_NAMES, POLLUTANT_ORDER, get_data, pollutant_level, AQI_COLORS
from language import aqi_label, lang, t

DAY_OPTIONS = [1, 7, 30]


def render():
    df = get_data()
    d_min, d_max_ts = df.index.min(), df.index.max()

    st.markdown(f'<div class="page-title">{t("poll_title")}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="page-sub">{t("poll_sub")}</div>', unsafe_allow_html=True)

    # Radio lưu lựa chọn dưới dạng chữ đã dịch ("7 ngày"), sang ngôn ngữ khác chữ đó không còn
    # trong danh sách và bị lỗi. Nên mỗi ngôn ngữ dùng 1 radio riêng (key theo ngôn ngữ),
    # còn lựa chọn lưu bằng số ngày trong poll_days_value để giữ nguyên khi đổi ngôn ngữ.
    days = st.radio(t("time_range"), DAY_OPTIONS,
                     index=DAY_OPTIONS.index(st.session_state.get("poll_days_value", 7)),
                     format_func=lambda d: t(f"range_{d}"),
                     horizontal=True, key=f"poll_days_{lang()}")
    st.session_state.poll_days_value = days
    start = max(d_min, d_max_ts - pd.Timedelta(days=days))
    view = df.loc[start:d_max_ts]

    cols = st.columns(4)
    for i, key in enumerate(POLLUTANT_ORDER + ["no", "nh3"]):
        cur_val = view[key].dropna().iloc[-1] if view[key].notna().any() else float("nan")
        plvl = pollutant_level(key, cur_val) if pd.notna(cur_val) else 1
        with cols[i % 4]:
            st.markdown(
                f"""<div class="pcard"><div class="top">{POLLUTANT_NAMES[key]}
                <span class="pbadge" style="background:{AQI_COLORS[plvl]}">● {aqi_label(plvl)}</span></div>
                <div class="pval">{cur_val:.1f} <span class="unit">µg/m³</span></div>
                </div>""" if pd.notna(cur_val) else
                f"""<div class="pcard"><div class="top">{POLLUTANT_NAMES[key]}</div>
                <div class="pval" style="color:#6b7280;font-size:16px">{t("no_data")}</div></div>""",
                unsafe_allow_html=True,
            )
            fig = go.Figure(go.Scatter(x=view.index, y=view[key], mode="lines",
                                        line=dict(color=AQI_COLORS[plvl], width=1.8),
                                        connectgaps=False, hoverinfo="skip"))
            fig.update_layout(height=120, margin=dict(l=10, r=10, t=6, b=6))
            st.plotly_chart(fig, width="stretch", key=f"poll_{key}")
