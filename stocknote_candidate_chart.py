"""An in-app daily candlestick chart for the selected monitored candidate."""
import pandas as pd
import streamlit as st

from stocknote_candidate_view import candidate_code


@st.cache_data(ttl=900, max_entries=100, show_spinner=False)
def load_daily_prices(code):
    import yfinance as yf

    if not candidate_code(code):
        return pd.DataFrame()
    frame = yf.download(
        f"{code}.T", period="1y", interval="1d", auto_adjust=False,
        progress=False, threads=False,
    )
    if frame is None or frame.empty:
        return pd.DataFrame()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    columns = ["Open", "High", "Low", "Close"]
    if not set(columns).issubset(frame.columns):
        return pd.DataFrame()
    frame = frame[columns].apply(pd.to_numeric, errors="coerce").dropna()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    return frame.rename_axis("Date").reset_index().tail(130)


def render_daily_chart(candidate):
    import altair as alt

    code = candidate_code(candidate.get("code"))
    name = candidate.get("name") or code
    st.subheader(f"{code} {name} の日足チャート")
    try:
        with st.spinner("日足チャートを読み込んでいます…"):
            frame = load_daily_prices(code)
    except Exception:
        st.info("日足の株価データを取得できませんでした。しばらくしてから銘柄を選び直してください。")
        return
    if frame.empty:
        st.info("この銘柄の日足データは現在取得できません。")
        return
    base = alt.Chart(frame).encode(
        x=alt.X("Date:T", title="日付"),
        color=alt.condition("datum.Close >= datum.Open", alt.value("#ef5350"), alt.value("#42a5f5")),
        tooltip=[alt.Tooltip("Date:T", title="日付", format="%Y/%m/%d"),
                 alt.Tooltip("Open:Q", title="始値", format=",.2f"),
                 alt.Tooltip("High:Q", title="高値", format=",.2f"),
                 alt.Tooltip("Low:Q", title="安値", format=",.2f"),
                 alt.Tooltip("Close:Q", title="終値", format=",.2f")],
    )
    wick = base.mark_rule().encode(y=alt.Y("Low:Q", title="株価（円）", scale=alt.Scale(zero=False)), y2="High:Q")
    body = base.mark_bar(size=5).encode(y=alt.Y("Open:Q", title="株価（円）", scale=alt.Scale(zero=False)), y2="Close:Q")
    st.altair_chart((wick + body).properties(height=340).interactive(), width="stretch", alt=f"{code} {name}の直近約6か月の日足ローソク足チャート")
    st.caption("1本が1営業日です。赤は始値以上で終了、青は始値より下で終了。直近約6か月を表示しています。")
