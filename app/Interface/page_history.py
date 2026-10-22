# page_history.py
# Trang lịch sử: bộ lọc ngày, thống kê tổng hợp, phân bố AQI, xu hướng.

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import AQI_COLORS, POLLUTANT_NAMES, get_data
from language import aqi_label, fmt_day, t

MIN_DAY_HOURS = 12  # ngày có ít hơn số giờ dữ liệu AQI này thì không xếp hạng ngày ô nhiễm/sạch nhất


def render():
    df = get_data()
    d_min, d_max_ts = df.index.min(), df.index.max()
    d_min_date, d_max_date = d_min.date(), d_max_ts.date()

    st.markdown(f'<div class="page-title">{t("hist_title")}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="page-sub">{t("hist_sub", a=d_min.strftime("%d/%m/%Y"), b=d_max_ts.strftime("%d/%m/%Y"))}</div>',
        unsafe_allow_html=True,
    )

    # ── Bộ lọc ngày ─────────────────────────────────────────────────────
    if "hist_range" not in st.session_state:
        st.session_state.hist_range = (
            max(d_min_date, (d_max_ts - pd.Timedelta(days=7)).date()), d_max_date,
        )

    rng = st.date_input(
        t("date_range"), value=st.session_state.hist_range,
        min_value=d_min_date, max_value=d_max_date, key="hist_date_input",
    )

    if not isinstance(rng, tuple) or len(rng) != 2:
        st.info(t("pick_both"))
        return
    start, end = rng
    view = df.loc[str(start):str(end)]
    if view.empty:
        st.warning(t("no_data_range"))
        return

    # ── Thẻ thống kê tổng hợp ─────────────────────────────────────────
    aqi_view = view["aqi"].dropna()
    total_hours = len(view)
    missing_hours = int(view["aqi"].isna().sum())
    complete_pct = 100 * (1 - missing_hours / total_hours) if total_hours else 0

    st.markdown(f'<div class="section-h">{t("summary_h")}</div>', unsafe_allow_html=True)
    # Thống kê theo ngày. Chỉ xét ngày có >= MIN_DAY_HOURS giờ dữ liệu AQI,
    # để ngày thiếu dữ liệu (vd hôm nay mới qua vài giờ) không bị xếp hạng sai.
    day = aqi_view.index.normalize()
    daily = pd.DataFrame({
        "hours": aqi_view.groupby(day).size(),
        "poor":  (aqi_view >= 4).groupby(day).sum(),    # số giờ Poor trở lên
        "clean": (aqi_view <= 2).groupby(day).sum(),    # số giờ Good/Fair
        "peak":  aqi_view.groupby(day).max(),
    })
    daily["pm25"] = view["pm2_5"].groupby(view.index.normalize()).mean()
    daily = daily[daily["hours"] >= MIN_DAY_HOURS]

    def _ties(ranked: pd.DataFrame, keys: list) -> str:
        """' · +N more days' nếu có ngày khác hòa hoàn toàn với ngày đứng đầu."""
        n = int((ranked[keys] == ranked[keys].iloc[0]).all(axis=1).sum()) - 1
        return t("ties_one" if n == 1 else "ties_many", n=n) if n > 0 else ""

    def _card(col, label, value, sub, color="#1f2937"):
        with col:
            st.markdown(
                f"""<div class="metric-card"><div class="mlbl">{label}</div>
                <div class="mval" style="color:{color}">{value}</div>
                <div class="msub">{sub}</div></div>""",
                unsafe_allow_html=True,
            )

    m1, m2, m3 = st.columns(3)

    # Ngày ô nhiễm nhất: nhiều giờ Poor trở lên nhất; bằng nhau thì PM2.5 trung bình cao hơn
    polluted = daily[daily["poor"] > 0].sort_values(["poor", "pm25"], ascending=[False, False])
    if daily.empty:
        _card(m1, t("most_polluted"), "—", t("not_enough_period"), "#9ca3af")
    elif polluted.empty:
        _card(m1, t("most_polluted"), t("none"), t("no_poor_hours", poor=aqi_label(4)), AQI_COLORS[1])
    else:
        d, r = polluted.index[0], polluted.iloc[0]
        _card(m1, t("most_polluted"), fmt_day(d),
              t("polluted_sub", n=int(r["poor"]), h=int(r["hours"]), poor=aqi_label(4),
                pm=f"{r['pm25']:.0f}") + _ties(polluted, ["poor", "pm25"]),
              AQI_COLORS[int(r["peak"])])

    # Ngày sạch nhất: không có giờ Poor nào và ít nhất nửa số giờ Good/Fair;
    # chọn ngày có tỉ lệ giờ Good/Fair cao nhất, bằng nhau thì PM2.5 trung bình thấp hơn
    clean = daily[(daily["poor"] == 0) & (daily["clean"] >= daily["hours"] / 2)].copy()
    clean["share"] = clean["clean"] / clean["hours"]
    clean = clean.sort_values(["share", "pm25"], ascending=[False, True])
    if daily.empty:
        _card(m2, t("cleanest"), "—", t("not_enough_period"), "#9ca3af")
    elif clean.empty:
        _card(m2, t("cleanest"), t("none"),
              t("no_clean_day", good=aqi_label(1), fair=aqi_label(2), poor=aqi_label(4)), AQI_COLORS[4])
    else:
        d, r = clean.index[0], clean.iloc[0]
        _card(m2, t("cleanest"), fmt_day(d),
              t("clean_sub", n=int(r["clean"]), h=int(r["hours"]), good=aqi_label(1), fair=aqi_label(2),
                pm=f"{r['pm25']:.0f}") + _ties(clean, ["share", "pm25"]),
              AQI_COLORS[1])

    _card(m3, t("sensor"), t("sensor_val", m=missing_hours, t=total_hours),
          t("sensor_sub", p=f"{complete_pct:.1f}"))

    # ── Phân bố chất lượng không khí theo thời gian ─────────────────────
    st.markdown(f'<div class="section-h">{t("dist_h")}</div>', unsafe_allow_html=True)
    hourly = aqi_view.astype(int)
    dist = hourly.value_counts().reindex(range(1, 6), fill_value=0)
    total = max(int(dist.sum()), 1)

    levels_present = [l for l in range(1, 6) if dist[l] > 0]
    fig_pie = go.Figure(go.Pie(
        labels=[t("pie_label", l=l, name=aqi_label(l)) for l in levels_present],
        values=[int(dist[l]) for l in levels_present],
        marker=dict(colors=[AQI_COLORS[l] for l in levels_present], line=dict(color="#ffffff", width=2)),
        hole=0.45,
        textinfo="percent",
        textfont=dict(size=13, color="#ffffff"),
        hovertemplate=t("pie_hover"),
    ))
    fig_pie.update_layout(
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        annotations=[dict(text=f"{total}h", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="#374151")],
    )
    st.plotly_chart(fig_pie, width="stretch")

     # ── Biểu đồ AQI: 1 cột = trung bình 1 ngày, hover xem chi tiết 24h ──
    st.markdown(f'<div class="section-h">{t("daily_h")}</div>', unsafe_allow_html=True)

    day_key = view.index.normalize()
    daily_avg = view["aqi"].groupby(day_key).mean().dropna()

    def _hour_breakdown(day: pd.Timestamp) -> str:
        day_vals = view.loc[view.index.normalize() == day, "aqi"].sort_index()
        if day_vals.empty:
            return t("no_data")

        groups = []  # mỗi phần tử: (giờ bắt đầu, giờ kết thúc, level)
        start_h = prev_h = None
        prev_lvl = None

        for t, v in day_vals.items():
            h = t.hour
            lvl = None if pd.isna(v) else int(v)
            if lvl == prev_lvl and prev_h is not None and h == prev_h + 1:
                prev_h = h  # nối dài nhóm hiện tại
            else:
                if start_h is not None:
                    groups.append((start_h, prev_h, prev_lvl))
                start_h = prev_h = h
                prev_lvl = lvl

        if start_h is not None:
            groups.append((start_h, prev_h, prev_lvl))

        lines = []
        for s, e, lvl in groups:
            label = t("no_data") if lvl is None else aqi_label(lvl)
            rng = f"{s}h" if s == e else f"{s}h→{e}h"
            lines.append(f"{rng}: {label}")
        return "<br>".join(lines)

    hover_texts = [_hour_breakdown(d) for d in daily_avg.index]
    bar_colors = [AQI_COLORS[int(round(v))] for v in daily_avg.values]

    fig1 = go.Figure(go.Bar(
        x=daily_avg.index, y=daily_avg.values,
        marker=dict(color=bar_colors, line=dict(color="#ffffff", width=1)),
        customdata=hover_texts,
        hovertemplate=t("daily_hover"),
    ))
    fig1.update_yaxes(range=[0, 5.5], tickvals=[1, 2, 3, 4, 5],
                       ticktext=[f"{l} - {aqi_label(l)}" for l in range(1, 6)],
                       gridcolor="rgba(0,0,0,0.08)", zeroline=False)
    fig1.update_xaxes(tickangle=-45, showgrid=False, tickformat="%d/%m")
    fig1.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=60),
        bargap=0.15,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, align="left"),
    )
    st.plotly_chart(fig1, width="stretch")

    # ── Chọn chất ô nhiễm cho biểu đồ xu hướng ───────────────────────────
    chosen = st.multiselect(
        t("trend_select"), options=list(POLLUTANT_NAMES),
        default=["pm2_5"], format_func=lambda k: POLLUTANT_NAMES[k], key="trend_multiselect",
    )

    # ── Xu hướng nhiều chất ô nhiễm ──────────────────────────────────────
    st.markdown(f'<div class="section-h">{t("trend_h")}</div>', unsafe_allow_html=True)

    if chosen:
        fig3 = go.Figure()
        for key in chosen:
            fig3.add_trace(go.Scatter(x=view.index, y=view[key], mode="lines",
                                       name=POLLUTANT_NAMES[key], connectgaps=False))
        fig3.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10),
                            yaxis_title="µg/m³")
        st.plotly_chart(fig3, width="stretch")
    else:
        st.info(t("select_one"))

    # ── Tải xuống ───────────────────────────────────────────────────────
    st.caption(t("stored_caption"))
    st.download_button(
        t("download_csv"), data=view.reset_index().to_csv(index=False).encode("utf-8"),
        file_name=f"aqi_{start}_{end}.csv", mime="text/csv",
    )
