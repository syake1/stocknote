import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="銘柄分析ノート PRO", layout="wide")
st.title("📋 銘柄分析ノート PRO")
st.caption("CSVは高速一括分析、選択した銘柄だけ詳細分析します。")


def normalize_code(value):
    text = str(value).strip().upper().replace(".T", "")
    text = re.sub(r"\.0$", "", text)
    m = re.search(r"([0-9A-Z]{4})", text)
    return m.group(1) if m else ""


def read_uploaded_csv(uploaded):
    raw = uploaded.getvalue()
    last = None
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
            break
        except Exception as exc:
            last = exc
    else:
        raise ValueError(f"CSVを読み込めませんでした: {last}")

    df = df.dropna(how="all").copy()
    if df.empty:
        raise ValueError("CSVに銘柄がありません。")

    code_col = None
    name_col = None
    for col in df.columns:
        key = str(col).strip().lower()
        if code_col is None and key in {"code", "コード", "銘柄コード", "証券コード"}:
            code_col = col
        if name_col is None and key in {"name", "名称", "銘柄名", "会社名"}:
            name_col = col

    if code_col is None:
        best_col, best_count = None, 0
        for col in df.columns:
            count = df[col].astype(str).map(normalize_code).ne("").sum()
            if count > best_count:
                best_col, best_count = col, count
        code_col = best_col

    if code_col is None:
        raise ValueError("銘柄コード列を見つけられませんでした。")

    if name_col is None:
        for col in df.columns:
            if col != code_col:
                name_col = col
                break

    df[code_col] = df[code_col].map(normalize_code)
    df = df[df[code_col] != ""].drop_duplicates(code_col).reset_index(drop=True)
    if df.empty:
        raise ValueError("有効な銘柄コードがありません。")
    return df, code_col, name_col


def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def technical_row(code, name, hist):
    if hist is None or hist.empty or "Close" not in hist:
        return {"code": code, "name": name or code, "error": "株価データなし"}
    hist = hist.dropna(subset=["Close"]).copy()
    if len(hist) < 30:
        return {"code": code, "name": name or code, "error": "履歴不足"}

    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(close) < 30:
        return {"code": code, "name": name or code, "error": "終値データ不足"}

    rsi = calculate_rsi(close)
    mid = close.rolling(25).mean()
    std = close.rolling(25).std()
    lower = mid - 2 * std
    upper = mid + 2 * std
    latest = float(close.iloc[-1])
    latest_rsi = float(rsi.iloc[-1])
    bb_mid = float(mid.iloc[-1])
    bb_lower = float(lower.iloc[-1])
    bb_upper = float(upper.iloc[-1])

    width = max(bb_mid - bb_lower, 1e-9)
    bb_pos = (latest - bb_mid) / width
    rsi_score = float(np.clip((70 - latest_rsi) / 40 * 10, 0, 10))
    bb_score = float(np.clip((1.25 - bb_pos) / 2.25 * 10, 0, 10))
    score = float(np.clip(rsi_score * 0.65 + bb_score * 0.35, 0, 10))
    if latest <= bb_lower * 1.02:
        score = max(score, 9.5)

    return {
        "code": code,
        "name": name or code,
        "momentum": score,
        "current_price": latest,
        "sim_buy": min(latest, bb_lower),
        "sim_sell": max(latest * 1.02, bb_mid),
        "sim_target": max(latest * 1.05, bb_upper),
        "rsi": latest_rsi,
        "bb_lower": bb_lower,
        "bb_mid": bb_mid,
        "error": None,
    }


def frame_from_batch(downloaded, ticker):
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy()
    for level in range(downloaded.columns.nlevels):
        values = downloaded.columns.get_level_values(level)
        if ticker in values:
            try:
                frame = downloaded.xs(ticker, axis=1, level=level, drop_level=True).copy()
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(-1)
                return frame
            except Exception:
                pass
    return pd.DataFrame()


def download_one(code):
    ticker = f"{code}.T"
    for _ in range(2):
        try:
            hist = yf.download(ticker, period="1y", auto_adjust=False, progress=False, threads=False, timeout=20)
            if hist is not None and not hist.empty:
                if isinstance(hist.columns, pd.MultiIndex):
                    hist.columns = hist.columns.get_level_values(0)
                return hist
        except Exception:
            pass
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def fast_batch_analyze(items):
    items = [(normalize_code(c), str(n or "")) for c, n in items]
    items = [(c, n) for c, n in items if c]
    results = []
    batch_size = 20

    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        tickers = [f"{c}.T" for c, _ in batch]
        downloaded = pd.DataFrame()
        try:
            downloaded = yf.download(
                tickers,
                period="1y",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=25,
            )
        except Exception:
            downloaded = pd.DataFrame()

        for code, name in batch:
            ticker = f"{code}.T"
            frame = frame_from_batch(downloaded, ticker)
            if frame.empty or "Close" not in frame:
                frame = download_one(code)
            try:
                results.append(technical_row(code, name, frame))
            except Exception as exc:
                results.append({"code": code, "name": name or code, "error": f"分析エラー: {exc}"})
    return results


@st.cache_data(ttl=900, show_spinner=False)
def detailed_analysis(code):
    code = normalize_code(code)
    hist = yf.download(code + ".T", period="2y", auto_adjust=False, progress=False, threads=False, timeout=20)
    if hist is None or hist.empty:
        return {"error": "株価データが取得できませんでした。"}
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    hist = hist.dropna(subset=["Open", "High", "Low", "Close"]).copy()
    if len(hist) < 60:
        return {"error": "株価データが十分にありません。"}

    base = technical_row(code, code, hist)
    if base.get("error"):
        return base
    try:
        info = yf.Ticker(code + ".T").info or {}
    except Exception:
        info = {}

    hist["MA25"] = hist["Close"].rolling(25).mean()
    hist["MA75"] = hist["Close"].rolling(75).mean()
    mid = hist["Close"].rolling(25).mean()
    std = hist["Close"].rolling(25).std()
    hist["BB_upper"] = mid + 2 * std
    hist["BB_lower"] = mid - 2 * std

    base.update({
        "name": info.get("shortName") or code,
        "sector": info.get("sector") or "—",
        "summary": info.get("longBusinessSummary") or "情報が取得できませんでした。",
        "per": info.get("trailingPE"),
        "pbr": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "dividend_yield": info.get("dividendYield"),
        "hist": hist,
    })
    return base


def chart(hist):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=hist.index, open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"], name="株価"))
    for label, col in (("MA25", "MA25"), ("MA75", "MA75"), ("BB +2σ", "BB_upper"), ("BB -2σ", "BB_lower")):
        if col in hist:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[col], name=label))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False)
    return fig


for key, value in {
    "watchlist_df": None,
    "watchlist_code_col": None,
    "watchlist_name_col": None,
    "batch_results": None,
    "analyze_code": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.markdown("### 📂 CSV買い銘柄リスト")
uploaded = st.file_uploader("CSVファイルをアップロード", type=["csv"])

if uploaded is not None:
    try:
        df, code_col, name_col = read_uploaded_csv(uploaded)
        st.session_state.watchlist_df = df
        st.session_state.watchlist_code_col = code_col
        st.session_state.watchlist_name_col = name_col
        st.success(f"✅ {len(df)}銘柄を読み込みました")
    except Exception as exc:
        st.error(str(exc))

if st.session_state.watchlist_df is not None and not st.session_state.watchlist_df.empty:
    df = st.session_state.watchlist_df
    code_col = st.session_state.watchlist_code_col
    name_col = st.session_state.watchlist_name_col
    items = []
    for _, row in df.iterrows():
        code = normalize_code(row[code_col])
        name = ""
        if name_col is not None and name_col in row.index and pd.notna(row[name_col]):
            name = str(row[name_col]).strip()
        if code:
            items.append((code, name))

    st.caption(f"分析対象: {len(items)}銘柄")
    if st.button("📊 一括分析してランキング表示", type="primary", use_container_width=True):
        with st.spinner(f"{len(items)}銘柄を分析しています。しばらくお待ちください…"):
            fast_batch_analyze.clear()
            st.session_state.batch_results = fast_batch_analyze(tuple(items))

if st.session_state.batch_results is not None:
    valid = [r for r in st.session_state.batch_results if not r.get("error") and r.get("momentum") is not None]
    invalid = [r for r in st.session_state.batch_results if r.get("error") or r.get("momentum") is None]
    valid.sort(key=lambda x: x["momentum"], reverse=True)

    st.markdown("### 🏆 逆張りチャンス ランキング")
    st.caption(f"分析成功 {len(valid)}銘柄 / 失敗 {len(invalid)}銘柄")

    if valid:
        table = pd.DataFrame([
            {"順位": i + 1, "コード": r["code"], "銘柄名": r["name"], "逆張りスコア": round(r["momentum"], 1), "RSI14": round(r["rsi"], 1), "現在値": round(r["current_price"], 1), "BB下限": round(r["bb_lower"], 1)}
            for i, r in enumerate(valid)
        ])
        st.dataframe(table, hide_index=True, use_container_width=True)

        options = valid[:50]
        selected = st.selectbox(
            "詳細分析する銘柄",
            [r["code"] for r in options],
            format_func=lambda c: next(f"{r['code']} {r['name']}｜スコア {r['momentum']:.1f}｜RSI {r['rsi']:.1f}" for r in options if r["code"] == c),
        )
        if st.button("🔍 選択銘柄を詳細分析", use_container_width=True):
            st.session_state.analyze_code = selected
    else:
        st.warning("ランキングを作れる銘柄がありませんでした。")

    if invalid:
        with st.expander(f"⚠️ 分析できなかった銘柄（{len(invalid)}件）"):
            for r in invalid[:100]:
                st.caption(f"{r.get('code','')} {r.get('name','')}: {r.get('error','不明なエラー')}")

st.markdown("---")
st.markdown("### 🔍 銘柄コードから個別分析")
manual = st.text_input("銘柄コード", placeholder="例：7203", max_chars=8)
if st.button("分析を実行"):
    code = normalize_code(manual)
    if code:
        st.session_state.analyze_code = code
    else:
        st.warning("有効な銘柄コードを入力してください。")

if st.session_state.analyze_code:
    detail = detailed_analysis(st.session_state.analyze_code)
    if detail.get("error"):
        st.error(detail["error"])
    else:
        st.subheader(f"🏢 {detail['name']} ({detail['code']})")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現在価格", f"¥{detail['current_price']:,.0f}")
        c2.metric("逆張りスコア", f"{detail['momentum']:.1f}/10")
        c3.metric("RSI14", f"{detail['rsi']:.1f}")
        c4.metric("BB下限", f"¥{detail['bb_lower']:,.0f}")
        st.plotly_chart(chart(detail["hist"].tail(180)), use_container_width=True)
        st.caption(f"更新日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')}")
