import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf


def _num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def fmt_num(v, suffix=""):
    return "—" if v is None else f"{float(v):,.2f}{suffix}"


def fmt_pct(v):
    return "—" if v is None else f"{float(v) * 100:.1f}%"


@st.cache_data(ttl=1800, show_spinner=False)
def company_profile(code):
    tk = yf.Ticker(f"{code}.T")
    try:
        info = tk.info or {}
    except Exception:
        info = {}
    return {
        "name": info.get("longName") or info.get("shortName") or code,
        "sector": info.get("sector") or info.get("industry") or "—",
        "summary": info.get("longBusinessSummary") or "",
        "market_cap": _num(info.get("marketCap")),
        "target": _num(info.get("targetMeanPrice")),
        "employees": info.get("fullTimeEmployees"),
    }


@st.cache_data(ttl=1800, show_spinner=False)
def performance_trend(code):
    try:
        inc = yf.Ticker(f"{code}.T").financials
    except Exception:
        return pd.DataFrame()
    if inc is None or inc.empty:
        return pd.DataFrame()

    def pick(labels):
        for label in labels:
            if label in inc.index:
                return pd.to_numeric(inc.loc[label], errors="coerce")
        return None

    revenue = pick(["Total Revenue", "TotalRevenue", "Operating Revenue"])
    op = pick(["Operating Income", "OperatingIncome"])
    net = pick(["Net Income", "NetIncome", "Net Income Common Stockholders"])
    cols = list(inc.columns[:4])
    if not cols:
        return pd.DataFrame()

    rows = {}
    if revenue is not None:
        rows["売上高"] = [revenue.get(c, np.nan) / 1e8 for c in cols]
    if op is not None:
        rows["営業利益"] = [op.get(c, np.nan) / 1e8 for c in cols]
    if net is not None:
        rows["純利益"] = [net.get(c, np.nan) / 1e8 for c in cols]
    if not rows:
        return pd.DataFrame()
    labels = [c.strftime("%Y/%m") if hasattr(c, "strftime") else str(c) for c in cols]
    df = pd.DataFrame(rows, index=labels).T
    return df[df.columns[::-1]]


def chance_gauge(score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(score),
        number={"suffix": "/100"},
        title={"text": "逆張りチャンス"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"thickness": 0.35}}
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def financial_comment(f):
    notes = []
    per = f.get("per"); pbr = f.get("pbr"); roe = f.get("roe")
    eq = f.get("equity_ratio"); opm = f.get("opm"); growth = f.get("growth"); div = f.get("div")
    if per is not None:
        notes.append("PERは割安水準" if 0 < per <= 15 else "PERはやや高め" if per >= 30 else "PERは標準圏")
    if pbr is not None:
        notes.append("PBRは低め" if 0 < pbr <= 1.2 else "PBRは高め" if pbr >= 3 else "PBRは標準圏")
    if roe is not None:
        notes.append("ROE良好" if roe >= 0.10 else "ROE低め" if roe < 0.05 else "ROE標準")
    if eq is not None:
        notes.append("自己資本比率良好" if eq >= 0.50 else "自己資本比率に注意" if eq < 0.20 else "自己資本比率は標準")
    if opm is not None:
        notes.append("営業利益率良好" if opm >= 0.10 else "営業利益率低め" if opm < 0.03 else "営業利益率は標準")
    if growth is not None:
        notes.append("増収" if growth > 0 else "減収")
    if div is not None and div >= 0.03:
        notes.append("配当利回り3%以上")
    return "・".join(notes) if notes else "取得できた財務指標が少ないため、評価は参考扱いです。"


FINANCIAL_GUIDE = {
    "PER": "株価が利益の何倍まで買われているか。低いほど割安の目安ですが、業種比較が必要です。",
    "PBR": "株価が純資産の何倍か。1倍前後以下は割安の目安ですが、収益力も確認します。",
    "ROE": "自己資本を使って利益を生む力。一般に10%以上は良好の目安です。",
    "自己資本比率": "総資産に占める自己資本の割合。高いほど財務余力がある目安です。",
    "営業利益率": "本業の売上に対する利益率。高く安定しているほど本業が強い目安です。",
    "売上成長率": "前年と比べた売上の増減。プラスなら増収です。",
    "配当利回り": "株価に対する年間配当の割合。3%以上を加点しています。",
}


def financial_rows(f):
    """Build the displayed values and plain-language judgements from scored data."""
    specs = [
        ("PER", "per", lambda v: fmt_num(v, "倍"), lambda v: "割安" if 0 < v <= 15 else "高め" if v >= 35 else "標準"),
        ("PBR", "pbr", lambda v: fmt_num(v, "倍"), lambda v: "低め" if 0 < v <= 1.2 else "高め" if v >= 4 else "標準"),
        ("ROE", "roe", fmt_pct, lambda v: "良好" if v >= .10 else "注意" if v < 0 else "標準"),
        ("自己資本比率", "equity_ratio", fmt_pct, lambda v: "良好" if v >= .50 else "低め" if v < .20 else "標準"),
        ("営業利益率", "opm", fmt_pct, lambda v: "良好" if v >= .10 else "赤字" if v < 0 else "標準"),
        ("売上成長率", "growth", fmt_pct, lambda v: "増収" if v >= .05 else "減収" if v < 0 else "横ばい"),
        ("配当利回り", "div", fmt_pct, lambda v: "3%以上" if v >= .03 else "3%未満"),
    ]
    rows = []
    for label, key, formatter, judge in specs:
        value = f.get(key)
        rows.append({
            "指標": label,
            "数値": formatter(value) if value is not None else "データなし",
            "判定": judge(float(value)) if value is not None else "判定しない",
            "意味": FINANCIAL_GUIDE[label],
            "取得元": f.get("sources", {}).get(key, "取得元なし"),
        })
    return rows


def render_extended_detail(code, row, f, final_score):
    st.markdown("---")
    st.markdown("## 🧾 詳細分析")

    c1, c2 = st.columns([1, 2])
    with c1:
        st.plotly_chart(chance_gauge(row.get("買いスコア", 0)), use_container_width=True, config={"displayModeBar": False})
    with c2:
        rsi = row.get("RSI14")
        bb = row.get("BB下限")
        px = row.get("現在値")
        reasons = []
        if rsi is not None:
            if rsi <= 30: reasons.append("RSIは売られすぎ圏")
            elif rsi <= 40: reasons.append("RSIは低位で反発候補")
            else: reasons.append("RSIは売られすぎ圏ではない")
        if px is not None and bb is not None:
            if px <= bb * 1.02: reasons.append("株価はBB−2σ付近")
            elif px < row.get("MA25", px): reasons.append("株価は25日線より下")
        if row.get("包み陽線"): reasons.append("包み陽線を確認")
        if row.get("MACD") is not None and row.get("MACDシグナル") is not None:
            reasons.append("MACDはシグナル以上" if row["MACD"] >= row["MACDシグナル"] else "MACDはまだ弱い")
        st.markdown("### 🎯 逆張り判定理由")
        st.write(" / ".join(reasons) if reasons else "判定材料が不足しています。")
        st.metric("総合評価", f"{final_score:.1f}/100")

    st.markdown("### 💰 ファンダメンタル評価")
    st.info(financial_comment(f))
    rows = financial_rows(f)
    metric_cols = st.columns(7)
    for col, item in zip(metric_cols, rows):
        col.metric(item["指標"], item["数値"])
        col.caption(item["判定"])
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    if not f.get("available"):
        st.warning("今回は財務指標を取得できなかったため、ファンダメンタル点は中立50点として扱います。推測値は表示しません。")
    else:
        st.caption(f"取得できた財務指標: {f.get('available', 0)}/7。データなしの項目は総合点に加減していません。")

    profile = company_profile(code)
    st.markdown("### 🏢 事業概要")
    p1, p2, p3 = st.columns(3)
    p1.metric("会社名", profile["name"] if profile["name"] != code else row.get("銘柄名", code))
    p2.metric("業種", profile["sector"])
    if profile["market_cap"] is not None:
        p3.metric("時価総額", f"{profile['market_cap']/1e8:,.0f}億円")
    else:
        p3.metric("時価総額", "—")
    if profile["summary"]:
        with st.expander("事業内容・企業概要", expanded=False):
            st.write(profile["summary"])
    else:
        st.caption("企業概要は取得できませんでした。")

    perf = performance_trend(code)
    st.markdown("### 📊 業績推移")
    if perf.empty:
        st.caption("業績推移データは取得できませんでした。")
    else:
        left, right = st.columns([1, 2])
        with left:
            st.dataframe(perf.round(0), use_container_width=True)
        with right:
            fig = go.Figure()
            for idx in perf.index:
                fig.add_trace(go.Bar(x=perf.columns, y=perf.loc[idx], name=idx))
            fig.update_layout(barmode="group", height=320, margin=dict(l=10, r=10, t=20, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("### 🔗 参考情報")
    st.markdown(f"[Yahoo!ファイナンス](https://finance.yahoo.co.jp/quote/{code}.T/profile) / [株探](https://kabutan.jp/stock/?code={code}) / [みんかぶ](https://minkabu.jp/stock/{code})")
