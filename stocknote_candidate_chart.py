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
    import plotly.graph_objects as go

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
    chart = go.Figure(
        data=[
            go.Candlestick(
                x=frame["Date"],
                open=frame["Open"],
                high=frame["High"],
                low=frame["Low"],
                close=frame["Close"],
                increasing_line_color="#ef5350",
                decreasing_line_color="#42a5f5",
                name=f"{code} {name}",
            )
        ]
    )
    chart.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis_rangeslider_visible=False,
        xaxis_title="日付",
        yaxis_title="株価（円）",
        hovermode="x unified",
        showlegend=False,
    )
    st.plotly_chart(chart, width="stretch", config={"displayModeBar": False})
    st.caption("1本が1営業日です。赤は始値以上で終了、青は始値より下で終了。直近約6か月を表示しています。")
