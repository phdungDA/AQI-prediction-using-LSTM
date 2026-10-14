# app.py
# Entry point: cấu hình trang, sidebar, điều hướng giữa 3 file giao diện.
# Chạy: streamlit run app.py

import streamlit as st

from common import inject_css, last_update_text
import page_dashboard
import page_history
import page_forecast
import page_pollutants

st.set_page_config(page_title="AQI Hà Nội", page_icon="🌫️", layout="wide")
inject_css()

PAGE_ICONS = {"Dashboard": "🏠", "History": "🕐", "Forecast 24h": "📈", "Pollutants": "🌫️"}
PAGES = list(PAGE_ICONS)
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

with st.sidebar:
    st.markdown("### 🌫️ AQI Hà Nội")
    st.caption("Hanoi")
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
    st.caption("🕐 Latest data")
    st.write(last_update_text())

    if st.button("🔄 Refresh data", width="stretch"):
        st.cache_data.clear()
        st.rerun()

    auto = st.checkbox("Auto refresh (60 min)", value=True)
    if auto:
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=60 * 60 * 1000, key="auto_refresh")
        except ImportError:
            st.caption("⚠️ Requires `pip install streamlit-autorefresh`")

    st.markdown("---")
    st.caption("Data source: OpenWeather")

# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "Dashboard":
    page_dashboard.render()
elif st.session_state.page == "History":
    page_history.render()
elif st.session_state.page == "Forecast 24h":
    page_forecast.render()
else:
    page_pollutants.render()
