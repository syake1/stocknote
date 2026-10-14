"""Streamlit dashboard for Japanese ex-dividend price recovery research."""

from __future__ import annotations

from datetime import date, timedelta
import math

import altair as alt
import pandas as pd
import streamlit as st

from analyzer import AnalysisResult, analyze_history, normalize_tickers
from sbi_csv import merge_yahoo_dividends, parse_sbi_csv
from yahoo_data import fetch_yahoo_history


st.set_page_config(
    page_title="日本株 権利落ち回復リサーチ",
    page_icon=":material/candlestick_chart:",
    layout="wide",
)

st.title(":material/candlestick_chart: 日本株 権利落ち回復リサーチ")
st.caption("配当の権利落ち後、何営業日目に反発し、元の株価へ戻ったかを過去データから確認します。")


@st.cache_data(ttl="6h", max_entries=80, show_spinner=False)
def load_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    history = fetch_yahoo_history(symbol, start, end)
    if history is None or history.empty:
        raise ValueError("株価データを取得できませんでした。")
    return history


def generate_sample_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Create deterministic sample data for offline UI evaluation."""
    index = pd.bdate_range(start=start, end=end, inclusive="left")
    if index.empty:
        return pd.DataFrame(columns=["Close", "Dividends"], index=index)

    seed = sum(ord(char) for char in symbol) % 500
    event_positions: list[int] = []
    dividends = pd.Series(0.0, index=index)
    for year in sorted(set(index.year)):
        for month in (3, 9):
            candidates = index[(index.year == year) & (index.month == month)]
            if len(candidates) == 0:
                continue
            event_date = candidates[-2] if len(candidates) >= 2 else candidates[-1]
            event_positions.append(index.get_loc(event_date))
            dividends.loc[event_date] = 30.0 + seed % 20

    prices: list[float] = []
    for position in range(len(index)):
        baseline = 1800 + seed + position * 0.28 + 55 * math.sin(position / 34)
        shock = sum(
            -(35 + seed % 20) * math.exp(-(position - event) / 13)
            for event in event_positions
            if position >= event
        )
        prices.append(round(baseline + shock, 1))

    return pd.DataFrame({"Close": prices, "Dividends": dividends}, index=index)


@st.cache_data(ttl="6h", max_entries=40, show_spinner=False)
def load_and_analyze(
    symbols: tuple[str, ...],
    years: int,
    horizon: int,
    as_of: date,
    source: str,
    schema_version: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    del schema_version
    fetch_start = date(as_of.year - years - 1, 1, 1)
    lookback_start = pd.Timestamp(as_of) - pd.DateOffset(years=years)
    fetch_end = as_of + timedelta(days=1)
    events: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    errors: list[str] = []

    for symbol in symbols:
        try:
            history = (
                generate_sample_history(symbol, fetch_start, fetch_end)
                if source == "サンプルデータ"
                else load_history(symbol, fetch_start, fetch_end)
            )
            result: AnalysisResult = analyze_history(
                history, symbol, lookback_start, horizon
            )
            if result.events.empty:
                errors.append(f"{symbol}: 対象期間に配当イベントがありません")
                continue
            events.append(result.events)
            curves.append(result.curves)
        except Exception as exc:  # Network and vendor errors vary by version.
            errors.append(f"{symbol}: {exc}")

    all_events = pd.concat(events, ignore_index=True) if events else pd.DataFrame()
    all_curves = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return all_events, all_curves, errors


def analyze_sbi_upload(
    content: bytes,
    symbol: str,
    years: int,
    horizon: int,
    as_of: date,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    try:
        sbi_history = parse_sbi_csv(content)
        fetch_start = sbi_history.index.min().date()
        fetch_end = sbi_history.index.max().date() + timedelta(days=1)
        yahoo_history = load_history(symbol, fetch_start, fetch_end)
        history = merge_yahoo_dividends(sbi_history, yahoo_history)
        lookback_start = pd.Timestamp(as_of) - pd.DateOffset(years=years)
        result = analyze_history(history, symbol, lookback_start, horizon)
        if result.events.empty:
            return pd.DataFrame(), pd.DataFrame(), [
                f"{symbol}: CSVの期間内に配当イベントがありません"
            ]
        return result.events, result.curves, []
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), [f"{symbol}: {exc}"]


with st.sidebar:
    st.header("分析条件")
    data_source = st.segmented_control(
        "データ",
        ["Yahoo実データ", "SBI CSV", "サンプルデータ"],
        default="Yahoo実データ",
    )
    with st.form("analysis_form"):
        raw_tickers = st.text_area(
            "銘柄コード（1行に1銘柄）",
            value="7203\n8306\n9432",
            placeholder="7203\n8306\n9432",
            help="カンマは不要です。Enterで改行するか、スペースで区切って入力できます。",
            height=140,
        )
        uploaded_sbi_csv = st.file_uploader(
            "HYPER SBI 2 日足CSV",
            type=["csv"],
            help="時系列画面から『日足』『未修正』で出力したCSVを選びます。",
            disabled=data_source != "SBI CSV",
        )
        years = st.slider("過去の分析期間", 1, 10, 5, format="%d年")
        horizon = st.slider(
            "回復を待つ期間",
            10,
            180,
            90,
            step=10,
            format="%d営業日",
        )
        submitted = st.form_submit_button(
            ":material/analytics: 分析する", type="primary", width="stretch"
        )

    st.divider()
    st.caption(
        "反発: 権利落ち日の翌営業日以降で、前日終値を初めて上回った日。\n\n"
        "回復: 権利付き最終日の終値以上へ初めて戻った日。"
    )

symbols = normalize_tickers(raw_tickers)
if not symbols:
    st.info("銘柄コードを入力してください。", icon=":material/info:")
    st.stop()

if len(symbols) > 12:
    st.warning("一度に分析できるのは12銘柄までです。先頭12銘柄を使います。")
    symbols = symbols[:12]

with st.spinner("株価と配当履歴を読み込んでいます...", show_time=True):
    if data_source == "SBI CSV":
        if uploaded_sbi_csv is None:
            st.info("HYPER SBI 2から出力した日足CSVを選んでください。")
            st.stop()
        if len(symbols) > 1:
            st.warning("SBI CSV分析では、入力した先頭の銘柄コードを使用します。")
        events, curves, errors = analyze_sbi_upload(
            uploaded_sbi_csv.getvalue(), symbols[0], years, horizon, date.today()
        )
    else:
        events, curves, errors = load_and_analyze(
            tuple(symbols),
            years,
            horizon,
            date.today(),
            data_source or "Yahoo実データ",
            3,
        )

if data_source == "サンプルデータ":
    st.info(
        "サンプルデータを表示中です。数値は操作確認用で、実在の相場データではありません。",
        icon=":material/science:",
    )
elif data_source == "SBI CSV":
    st.info(
        "終値はHYPER SBI 2の未修正日足CSV、配当イベントはYahoo Financeを使用しています。",
        icon=":material/database:",
    )

if errors:
    with st.expander(f"取得メモ（{len(errors)}件）"):
        for message in errors:
            st.caption(message)

if events.empty:
    st.error("分析できる配当イベントが見つかりませんでした。銘柄コードまたは期間を変更してください。")
    st.stop()

events = events.sort_values("ex_date", ascending=False).reset_index(drop=True)
events["recovered_bottom_days"] = events["bottom_days"].where(events["recovered"])
recovered = events["recovered"].sum()
recovery_rate = recovered / len(events)
median_rebound = events["rebound_days"].median()
median_recovery = events.loc[events["recovered"], "recovery_days"].median()

summary = (
    events.groupby("ticker", as_index=False)
    .agg(
        events=("ex_date", "count"),
        median_rebound=("rebound_days", "median"),
        median_add_buy=("recovered_bottom_days", "median"),
        median_recovery=("recovery_days", "median"),
        recovered_events=("recovered", "sum"),
        recovery_rate=("recovered", "mean"),
        median_gap=("gap_pct", "median"),
        worst_drawdown=("max_drawdown_pct", "min"),
    )
)
symbol_order = {symbol: position for position, symbol in enumerate(symbols)}
summary["input_order"] = summary["ticker"].map(symbol_order).fillna(len(symbols))
summary = summary.sort_values("input_order").drop(columns="input_order")

st.subheader("まず見る結論")
focus_ticker = st.selectbox(
    "確認する銘柄",
    summary["ticker"].tolist(),
    format_func=lambda ticker: ticker.removesuffix(".T"),
)
focus = summary.loc[summary["ticker"] == focus_ticker].iloc[0]
focus_add_buy = focus["median_add_buy"]
focus_recovery = focus["median_recovery"]

with st.container(border=True):
    conclusion_columns = st.columns(3)
    conclusion_columns[0].metric(
        "追加買い参考日",
        "確認なし"
        if pd.isna(focus_add_buy)
        else (
            "権利落ち当日"
            if focus_add_buy == 0
            else f"権利落ち後 {focus_add_buy:.0f}営業日目"
        ),
    )
    conclusion_columns[1].metric(
        "元値回復の目安",
        "期間内に未回復"
        if pd.isna(focus_recovery)
        else f"{focus_recovery:.0f}営業日",
    )
    conclusion_columns[2].metric("元値まで戻った割合", f"{focus['recovery_rate']:.0%}")
    if pd.isna(focus_add_buy):
        st.warning("分析期間内に、回復したケースを確認できませんでした。")
    else:
        st.info(
            f"過去{int(focus['events'])}回のうち元値まで回復した"
            f"{int(focus['recovered_events'])}回について、回復までに一番安かった日を集計。"
            f"その中央が{focus_add_buy:.0f}営業日目です。0日目は権利落ち当日です。"
        )

st.caption("追加買い目安は過去の底値日の中央値です。将来の底値や回復を保証するものではありません。")

metric_columns = st.columns(4)
with metric_columns[0]:
    st.metric("権利落ち回数", f"{len(events)}回", border=True)
with metric_columns[1]:
    st.metric(
        "反発開始の中央値",
        "未確認" if pd.isna(median_rebound) else f"{median_rebound:.0f}日目",
        border=True,
    )
with metric_columns[2]:
    st.metric(
        "株価回復の中央値",
        "未回復" if pd.isna(median_recovery) else f"{median_recovery:.0f}日",
        border=True,
    )
with metric_columns[3]:
    st.metric("期間内の回復率", f"{recovery_rate:.0%}", border=True)

summary_display = summary.assign(
    ticker=summary["ticker"].str.removesuffix(".T"),
)
summary_display["buy_day"] = summary["median_add_buy"].map(
    lambda value: (
        "確認なし"
        if pd.isna(value)
        else "権利落ち当日"
        if value == 0
        else f"権利落ち後 {value:.0f}営業日目"
    )
)
summary_display["recovery_day"] = summary["median_recovery"].map(
    lambda value: "期間内に未回復" if pd.isna(value) else f"{value:.0f}営業日"
)

st.subheader("銘柄ごとの追加買い参考日")
for start in range(0, len(summary_display), 3):
    card_columns = st.columns(3)
    for column, row in zip(
        card_columns, summary_display.iloc[start : start + 3].itertuples()
    ):
        with column:
            st.metric(
                f"銘柄 {row.ticker}",
                row.buy_day.replace("権利落ち後 ", ""),
                border=True,
            )
            st.caption(
                f"元値回復: {row.recovery_day}　回復率: {row.recovery_rate:.0%}"
            )

with st.expander("銘柄別の詳しい集計"):
    st.dataframe(
        summary_display[
            ["ticker", "buy_day", "recovery_day", "recovery_rate", "events"]
        ],
        hide_index=True,
        column_config={
            "ticker": st.column_config.TextColumn("銘柄", pinned=True),
            "buy_day": st.column_config.TextColumn("追加買い参考日"),
            "recovery_day": st.column_config.TextColumn("元値回復の目安"),
            "recovery_rate": st.column_config.NumberColumn(
                "元値まで戻った割合", format="percent"
            ),
            "events": st.column_config.NumberColumn("過去回数", format="%d回"),
        },
    )

with st.container(border=True):
        st.subheader(f"{focus_ticker.removesuffix('.T')} 追加買い参考グラフ")
        signal_columns = st.columns(2)
        signal_columns[0].markdown(
            ":orange[**オレンジ線：追加買い参考日**]  "
            + ("確認なし" if pd.isna(focus_add_buy) else f"**{focus_add_buy:.0f}営業日目**")
        )
        signal_columns[1].markdown(
            ":green[**緑線：元値回復**]  "
            + ("期間内に未回復" if pd.isna(focus_recovery) else f"**{focus_recovery:.0f}営業日**")
        )
        st.caption(
            "オレンジ線は、過去に回復した回で『回復までに一番安かった日』を"
            "集計した参考位置です。未来の底値を示す線ではありません。"
        )
        chart_range = st.segmented_control(
            "表示期間",
            ["10日", "20日", "30日", "全期間"],
            default="30日",
            key="average_chart_range",
        )
        chart_days = (
            horizon if chart_range == "全期間" else int((chart_range or "30日").removesuffix("日"))
        )
        focus_curves = curves.loc[
            (curves["ticker"] == focus_ticker) & (curves["day"] <= chart_days)
        ].copy()
        focus_curves["after_ex_return_pct"] = focus_curves.groupby("ex_date")[
            "close"
        ].transform(lambda prices: prices / prices.iloc[0] - 1)
        average_curve = focus_curves.groupby("day", as_index=False)[
            "after_ex_return_pct"
        ].mean()
        chart = (
            alt.Chart(average_curve)
            .mark_line(strokeWidth=2, color="#60A5FA")
            .encode(
                x=alt.X("day:Q", title="権利落ちからの営業日"),
                y=alt.Y(
                    "after_ex_return_pct:Q",
                    title="権利落ち日終値からの騰落率",
                    axis=alt.Axis(format="%"),
                ),
                tooltip=[
                    alt.Tooltip("day:Q", title="営業日", format=".0f"),
                    alt.Tooltip(
                        "after_ex_return_pct:Q", title="その後の騰落率", format=".2%"
                    ),
                ],
            )
            .properties(height=320)
            .interactive()
        )
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#94A3B8").encode(y="y:Q")
        buy_marker = pd.DataFrame(
            {"day": [] if pd.isna(focus_add_buy) else [focus_add_buy]}
        )
        buy_marker = buy_marker.loc[buy_marker["day"] <= chart_days]
        recovery_marker = pd.DataFrame(
            {"day": [] if pd.isna(focus_recovery) else [focus_recovery]}
        )
        recovery_marker = recovery_marker.loc[recovery_marker["day"] <= chart_days]
        buy_rule = (
            alt.Chart(buy_marker)
            .mark_rule(color="#FF9800", strokeWidth=7, strokeDash=[8, 4])
            .encode(
                x=alt.X("day:Q"),
                tooltip=[alt.Tooltip("day:Q", title="買い検討", format=".0f")],
            )
        )
        recovery_rule = (
            alt.Chart(recovery_marker)
            .mark_rule(color="#22C55E", strokeWidth=2)
            .encode(
                x=alt.X("day:Q"),
                tooltip=[alt.Tooltip("day:Q", title="元値回復", format=".0f")],
            )
        )
        st.altair_chart(chart + zero + buy_rule + recovery_rule)
        st.caption(
            f"権利落ち直後の{chart_days}営業日を拡大表示。"
            "0日目は権利落ち当日です。"
        )

st.subheader("権利落ち回ごとの結果")
display_events = events.copy()
display_events["status"] = display_events["recovered"].map(
    {True: "回復", False: "未回復"}
)
st.dataframe(
    display_events[
        [
            "ticker",
            "ex_date",
            "dividend",
            "rebound_days",
            "rebound_date",
            "bottom_days",
            "bottom_date",
            "recovery_days",
            "recovery_date",
            "adjusted_recovery_days",
            "max_drawdown_pct",
            "status",
        ]
    ],
    hide_index=True,
    column_config={
        "ticker": st.column_config.TextColumn("銘柄", pinned=True),
        "ex_date": st.column_config.DateColumn("権利落ち日", format="YYYY/MM/DD"),
        "dividend": st.column_config.NumberColumn("1株配当", format="¥%.2f"),
        "rebound_days": st.column_config.NumberColumn("反発まで", format="%.0f営業日"),
        "rebound_date": st.column_config.DateColumn("反発確認日", format="YYYY/MM/DD"),
        "bottom_days": st.column_config.NumberColumn("最安値まで", format="%.0f営業日"),
        "bottom_date": st.column_config.DateColumn("最安値日", format="YYYY/MM/DD"),
        "recovery_days": st.column_config.NumberColumn("回復まで", format="%.0f営業日"),
        "recovery_date": st.column_config.DateColumn("元値回復日", format="YYYY/MM/DD"),
        "adjusted_recovery_days": st.column_config.NumberColumn(
            "配当調整後回復", format="%.0f日"
        ),
        "max_drawdown_pct": st.column_config.NumberColumn("最大下落", format="percent"),
        "status": st.column_config.TextColumn("判定"),
    },
)

event_options = {
    f"{row.ticker} | {row.ex_date:%Y/%m/%d} | 配当 ¥{row.dividend:,.2f}": (
        row.ticker,
        row.ex_date,
    )
    for row in events.itertuples()
}
selected_label = st.selectbox("詳しく見る権利落ち回", list(event_options))
selected_ticker, selected_date = event_options[selected_label]
detail = curves[
    (curves["ticker"] == selected_ticker) & (curves["ex_date"] == selected_date)
].copy()

with st.container(border=True):
    st.subheader("選択回の株価推移")
    line = (
        alt.Chart(detail)
        .mark_line(strokeWidth=2, color="#60A5FA")
        .encode(
            x=alt.X("day:Q", title="権利落ちからの営業日"),
            y=alt.Y("close:Q", title="終値（円）", scale=alt.Scale(zero=False)),
            tooltip=[
                alt.Tooltip("date:T", title="日付", format="%Y/%m/%d"),
                alt.Tooltip("day:Q", title="営業日", format=".0f"),
                alt.Tooltip("close:Q", title="終値", format=",.1f"),
            ],
        )
        .properties(height=330)
    )
    targets = pd.DataFrame(
        {
            "水準": ["権利付き終値", "配当調整後"],
            "価格": [detail["pre_close"].iloc[0], detail["adjusted_target"].iloc[0]],
        }
    )
    rules = (
        alt.Chart(targets)
        .mark_rule(strokeDash=[6, 4])
        .encode(
            y="価格:Q",
            color=alt.Color(
                "水準:N",
                scale=alt.Scale(
                    domain=["権利付き終値", "配当調整後"],
                    range=["#F87171", "#34D399"],
                ),
                title="回復判定水準",
            ),
            tooltip=["水準:N", alt.Tooltip("価格:Q", format=",.1f")],
        )
    )
    st.altair_chart(line + rules)

with st.expander("指標の読み方と注意点"):
    st.markdown(
        """
- 日数は土日祝日を除く取引日数です。権利落ち日を0日目として数えます。
- 「配当調整後回復」は、権利付き終値から1株配当を引いた水準への回復です。
- Yahoo実データでは株価と配当履歴をYahoo Financeから取得します。SBI CSVでは終値をSBI、配当履歴をYahoo Financeから取得します。
- SBI CSVは「日足」「未修正」で出力してください。株式分割をまたぐ期間は結果を個別に確認してください。
- これは過去データの調査ツールであり、将来の値動きや利益を保証する売買推奨ではありません。
        """
    )
