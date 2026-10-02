# app.py
# Entry point: cấu hình trang, sidebar, điều hướng giữa 3 file giao diện.
# Chạy: streamlit run app.py

import streamlit as st

from common import inject_css, last_update_text
import page_dashboard
import page_history
import page_forecast
import page_pollutants

st.set_page_config(page_title="AQI Dashboard", page_icon="🌫️", layout="wide")
inject_css()

PAGE_ICONS = {"Dashboard": "🏠", "Lịch sử": "🕐", "Dự báo 24h": "📈", "Chất ô nhiễm": "🌫️"}
PAGES = list(PAGE_ICONS)
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("### 🌫️ AQI Dashboard")
    st.caption("Vị trí hiện tại")
    st.markdown("---")

    for p in PAGES:
        if st.button(
            f"{PAGE_ICONS[p]}  {p}", width="stretch",
            type="primary" if st.session_state.page == p else "secondary",
            key=f"nav_{p}",
        ):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")
    st.caption("🕐 Dữ liệu mới nhất")
    st.write(last_update_text())

    if st.button("🔄 Làm mới dữ liệu", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    auto = st.checkbox("Tự làm mới (60 phút)", value=True)
    if auto:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60 * 60 * 1000, key="auto_refresh")
        except ImportError:
            st.caption("⚠️ Cần `pip install streamlit-autorefresh`")

    st.markdown("---")
    st.caption("Nguồn dữ liệu: OpenWeather")

# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "Dashboard":
    page_dashboard.render()
elif st.session_state.page == "Lịch sử":
    page_history.render()
elif st.session_state.page == "Dự báo 24h":
    page_forecast.render()
else:
    page_pollutants.render()
