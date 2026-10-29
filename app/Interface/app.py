# app.py
# Entry point: cấu hình trang, sidebar, điều hướng giữa 3 trang giao diện.
# Chạy: streamlit run app.py

import streamlit as st

from common import inject_css, last_update_text
from language import DEFAULT_LANG, t
import page_dashboard
import page_history
import page_pollutants

st.set_page_config(page_title="AQI Ha Noi", page_icon="🌫️", layout="wide")
inject_css()

PAGES = ["Dashboard", "History", "Pollutants"]
if st.session_state.get("page") not in PAGES:   # lần đầu vào, hoặc trang cũ đã bị bỏ
    st.session_state.page = "Dashboard"
if "lang" not in st.session_state:
    st.session_state.lang = DEFAULT_LANG

CHECK_SECONDS = 60

with st.sidebar:
    # ── Auto refresh ──────────────────────────────────────────────────────
    # Mỗi phút chạy lại trang một lần. Dữ liệu và dự đoán được cache theo mốc giờ mới nhất
    # của DB (common.latest_version), nên chỉ khi DB có giờ mới thì mới thật sự lấy lại.
    # debounce=False: mặc định thư viện đếm lại từ đầu mỗi khi trang chạy lại (bấm nút,
    # đổi trang...), nên người đang thao tác sẽ không bao giờ được tự cập nhật.
    if st.session_state.get("auto_refresh_on", True):
        try:
            from streamlit_autorefresh import st_autorefresh
            st_autorefresh(interval=CHECK_SECONDS * 1000, debounce=False, key="auto_refresh")
        except ImportError:
            st.caption(t("autorefresh_missing"))

    st.markdown(
        f"""
        <div style="display:flex;align-items:center;gap:12px;padding:4px 0 8px;">
          <svg width="56" height="56" viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
            <circle cx="200" cy="200" r="195" fill="white" stroke="#1f2937" stroke-width="3"/>
            <path d="M214.2,37.6 A163,163 0 0,1 350.1,136.3" stroke="#2e9e5b" stroke-width="36" fill="none" stroke-linecap="round"/>
            <path d="M358.8,163.3 A163,163 0 0,1 307.0,323.0" stroke="#9bc53d" stroke-width="36" fill="none" stroke-linecap="round"/>
            <path d="M283.9,339.7 A163,163 0 0,1 116.1,339.7" stroke="#f5b400" stroke-width="36" fill="none" stroke-linecap="round"/>
            <path d="M93.0,323.0 A163,163 0 0,1 41.2,163.3" stroke="#ee7d2d" stroke-width="36" fill="none" stroke-linecap="round"/>
            <path d="M49.9,136.3 A163,163 0 0,1 185.8,37.6" stroke="#d64545" stroke-width="36" fill="none" stroke-linecap="round"/>
            <text x="200" y="218" text-anchor="middle" font-family="sans-serif" font-size="82" font-weight="800" fill="#1f2937">AQI</text>
          </svg>
          <div>
            <div style="font-size:22px;font-weight:800;color:#1f2937;line-height:1.2;">AQI Hà Nội</div>
            <div style="font-size:15px;color:#6b7280;">{t("app_sub")}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Nút đổi ngôn ngữ: nhãn là ngôn ngữ sẽ chuyển sang (đang tiếng Việt thì hiện "English")
    if st.button(t("switch_lang"), width="stretch", key="switch_lang"):
        st.session_state.lang = "en" if st.session_state.lang == "vi" else "vi"
        st.rerun()
    st.markdown("---")

    for p in PAGES:
        if st.button(
            t("nav_" + p), width="stretch",
            type="primary" if st.session_state.page == p else "secondary",
            key=f"nav_{p}",
        ):
            st.session_state.page = p
            st.rerun()

    st.markdown("---")
    st.caption(t("latest_data"))
    st.write(last_update_text())

    if st.button(t("refresh"), width="stretch", key="refresh_data"):
        st.cache_data.clear()
        st.rerun()

    st.checkbox(t("auto_refresh"), value=True, key="auto_refresh_on")

    st.markdown("---")
    st.caption(t("data_source"))

# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "Dashboard":
    page_dashboard.render()
elif st.session_state.page == "History":
    page_history.render()
else:
    page_pollutants.render()
