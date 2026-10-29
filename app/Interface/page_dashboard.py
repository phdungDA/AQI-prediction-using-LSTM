# page_dashboard.py
# Trang Dashboard: tình trạng không khí hiện tại.

import pandas as pd
import streamlit as st

from common import (
    AQI_COLORS, CIGARETTE_PM25, POLLUTANT_NAMES, POLLUTANT_ORDER,
    confidence_ring_html, get_data, latest_row, pollutant_level, pred_error_text,
    predict_next_hour, scale_bar_html, trend_arrow,
)
from language import aqi_advice, aqi_label, scale_info, t


def render():
    df = get_data()
    row = latest_row(df)
    level = int(row["aqi"])
    worst_pollutant = max(POLLUTANT_ORDER, key=lambda c: pollutant_level(c, row[c]))

    st.markdown(f'<div class="page-title">{t("dash_title")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">{t("dash_sub", time=row.name.strftime("%H:%M %d/%m/%Y"))}</div>',
        unsafe_allow_html=True,
    )

    # ── Thẻ AQI lớn ──────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            f"""<div style="padding-top:6px;">
                  <div style="font-size:16px;color:#6b7280;font-weight:700;">{t("recommendation")}</div>
                  <div class="hero-note" style="margin-top:8px;">{aqi_advice(level)}</div>
                </div>""",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""<div style="text-align:right; padding-top:6px;">
                  <div style="font-size:16px;color:#6b7280;font-weight:600;">{t("main_pollutant")}</div>
                  <div style="font-size:30px;font-weight:800;color:#1f2937;margin-top:6px;">
                    {POLLUTANT_NAMES[worst_pollutant]}
                  </div>
                </div>""",
            unsafe_allow_html=True,
        )
    st.markdown(scale_bar_html(level), unsafe_allow_html=True)
    # ── 6 thẻ chỉ số khí quyển chi tiết ──────────────────────────────────
    st.markdown(f'<div class="section-h">{t("indicators_h")}</div>', unsafe_allow_html=True)

    cols = st.columns(len(POLLUTANT_ORDER))
    for col_box, key in zip(cols, POLLUTANT_ORDER):
        val = row[key]
        if pd.isna(val):
            color, badge_txt, val_txt = "#9ca3af", t("no_data"), "—"
        else:
            plvl = pollutant_level(key, val)
            color, badge_txt, val_txt = AQI_COLORS[plvl], aqi_label(plvl), f"{val:.1f}"
        prev = df[key].shift(1).loc[row.name]   # giá trị 1 giờ trước giờ mới nhất
        arrow, trend_lbl, trend_cls = trend_arrow(val, prev)
        if pd.isna(val) or pd.isna(prev):
            trend_txt = t("no_prev_hour")
        else:
            # So với giờ trước: vd "▲ rising +4.2 µg/m³"
            trend_txt = f'{arrow} {trend_lbl} {val - prev:+.1f} µg/m³'
        with col_box:
            st.markdown(
                f"""
                <div class="pcard">
                  <div class="top">{POLLUTANT_NAMES[key]}
                    <span class="pbadge" style="background:{color}">● {badge_txt}</span>
                  </div>
                  <div class="pval">{val_txt} <span class="unit">µg/m³</span></div>
                  <div class="pfoot"><span class="{trend_cls}">{trend_txt}</span></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── 2 khối: dự đoán giờ kế tiếp / số điếu thuốc 24h qua ─────────────
    st.markdown("<br>", unsafe_allow_html=True)
    i1, i2 = st.columns(2)

    next_level, probs = predict_next_hour()
    with i1:
        if next_level is not None:
            badge = (f'<span class="pbadge" style="background:{AQI_COLORS[next_level]}">'
                      f'● {aqi_label(next_level)} ({next_level})</span>')
            if next_level > level:
                change = t("change_worse", a=aqi_label(level), b=aqi_label(next_level))
            elif next_level < level:
                change = t("change_better", a=aqi_label(level), b=aqi_label(next_level))
            else:
                change = t("change_same", a=aqi_label(level))
            note = f"<b>{change}</b>"
            ring = confidence_ring_html(float(max(probs)) * 100, AQI_COLORS[next_level])
        else:
            badge = f'<span class="pbadge" style="background:#9ca3af">● {t("not_enough_badge")}</span>'
            note = pred_error_text(probs)  # probs lúc này là mã lỗi
            ring = ""
        st.markdown(
            f"""
            <div class="info-card">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;">
                <div>
                  <div class="ihead">{t("next_hour")}</div>
                  {badge}
                  <div class="inote">{note}</div>
                </div>{ring}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    pm_now = row["pm2_5"]
    with i2:
        if pd.notna(pm_now):
            # PM2.5 hiện tại ÷ 22 = số điếu/ngày nếu cả ngày hít không khí như lúc này (Berkeley Earth)
            cigs = pm_now / CIGARETTE_PM25
            value = t("cig_one" if round(cigs, 1) == 1 else "cig_many", n=f"{cigs:.1f}")
            note = t("cig_note")
        else:
            value = "—"
            note = t("no_pm_now")
        st.markdown(
            f"""
            <div class="info-card">
              <div class="ihead">{t("cig_head")}</div>
              <div class="ival">{value}</div>
              <div class="inote">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── Quy trình tính confidence (chỉ hiện khi có dự đoán) ─────────────
    if next_level is not None:
        win_start = row.name - pd.Timedelta(hours=23)
        conf = float(max(probs)) * 100
        steps = [
            (t("step1_t"), t("step1_d", a=f"{win_start:%H:%M %d/%m}", b=f"{row.name:%H:%M %d/%m}")),
            (t("step2_t"), t("step2_d")),
            (t("step3_t"), t("step3_d")),
            (t("step4_t"), t("step4_d", lvl=aqi_label(next_level), pct=f"{conf:.1f}")),
        ]
        steps_html = "".join(
            f'<div class="step"><div class="snum">{i}</div><div class="stitle">{title}</div>'
            f'<div class="stext">{desc}</div></div>'
            for i, (title, desc) in enumerate(steps, start=1)
        )
        bars_html = "".join(
            f'<div class="prow{" top" if lvl == next_level else ""}">'
            f'<span>{aqi_label(lvl)}</span>'
            f'<span class="pbar"><i style="width:{p * 100:.1f}%;background:{AQI_COLORS[lvl]}"></i></span>'
            f'<span class="pct">{p * 100:.1f}%</span></div>'
            for lvl, p in zip(range(1, 6), probs)
        )
        st.markdown(f'<div class="section-h">{t("conf_h")}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="steps">{steps_html}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="probs"><div class="stitle">{t("model_output")}</div>{bars_html}</div>',
            unsafe_allow_html=True,
        )

    # ── Giải thích thang đo AQI (cuối trang Dashboard) ──────────────────
    st.markdown(f'<div class="section-h">{t("scale_h")}</div>', unsafe_allow_html=True)
    # HTML viết liền 1 dòng mỗi hàng: tránh dòng thụt lề bị markdown hiểu thành khối code
    rows_html = "".join(
        f'<tr><td><span class="pbadge" style="background:{AQI_COLORS[lvl]}">{lvl}</span></td>'
        f'<td class="cat">{aqi_label(lvl)}</td>'
        f'<td>{scale_info(lvl)[0]}</td><td>{scale_info(lvl)[1]}</td></tr>'
        for lvl in range(1, 6)
    )
    st.markdown(
        '<div class="scale-wrap"><table class="scale-table"><thead><tr>'
        f'<th>{t("th_level")}</th><th>{t("th_category")}</th>'
        f'<th>{t("th_meaning")}</th><th>{t("th_rec")}</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table></div>',
        unsafe_allow_html=True,
    )
