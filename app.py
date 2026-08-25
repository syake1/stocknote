import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="銘柄分析ノート PRO (逆張り特化)", layout="wide", initial_sidebar_state="collapsed")
st.title("📋 銘柄分析ノート PRO")
st.caption("CSV一括分析は高速テクニカル検索、個別銘柄は詳細分析を行います。")


def normalize_code(value):
    code = str(value).strip().replace(".0", "")
    m = re.search(r"(\d{4})", code)
    return m.group(1) if m else ""


def read_uploaded_csv(uploaded):
    raw = uploaded.getvalue()
    last_error = None
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise ValueError(f"CSVを読み込めませんでした: {last_error}")

    df = df.dropna(how="all").copy()
    if df.empty:
        return df, None, None

    code_col = None
    name_col = None
    for col in df.columns:
        key = str(col).strip().lower()
        if code_col is None and key in {"code", "コード", "銘柄コード", "証券コード"}:
            code_col = col
        if name_col is None and key in {"name", "名称", "銘柄名", "会社名"}:
            name_col = col
    if code_col is None:
        code_col = df.columns[0]
    if name_col is None and len(df.columns) > 1:
        name_col = df.columns[1]

    df[code_col] = df[code_col].map(normalize_code)
    df = df[df[code_col] != ""].drop_duplicates(code_col).reset_index(drop=True)
    return df, code_col, name_col


def calculate_rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50)


def technical_row(code, name, hist):
    required = {"Open", "High", "Low", "Close", "Volume"}
    if hist is None or hist.empty or not required.issubset(hist.columns):
        return {"code": code, "name": name, "error": "株価データなし"}
    hist = hist.dropna(subset=["Close"]).copy()
    if len(hist) < 30:
        return {"code": code, "name": name, "error": "履歴不足"}

    close = hist["Close"].astype(float)
    rsi = calculate_rsi(close)
    mid = close.rolling(25).mean()
    std = close.rolling(25).std()
    lower = mid - 2 * std
    upper = mid + 2 * std
    latest = float(close.iloc[-1])
    latest_rsi = float(rsi.iloc[-1])
    bb_lower = float(lower.iloc[-1]) if pd.notna(lower.iloc[-1]) else latest
    bb_mid = float(mid.iloc[-1]) if pd.notna(mid.iloc[-1]) else latest
    bb_upper = float(upper.iloc[-1]) if pd.notna(upper.iloc[-1]) else latest

    if bb_mid != bb_lower:
        bb_pos = (latest - bb_mid) / max(bb_mid - bb_lower, 1e-9)
    else:
        bb_pos = 0.0

    # 逆張り向け: RSIが低いほど、BB下限に近いほど高得点。
    rsi_score = np.clip((70 - latest_rsi) / 40 * 10, 0, 10)
    bb_score = np.clip((1.25 - bb_pos) / 2.25 * 10, 0, 10)
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


def extract_ticker_frame(downloaded, ticker):
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if not isinstance(downloaded.columns, pd.MultiIndex):
        return downloaded.copy()
    for level in range(downloaded.columns.nlevels):
        vals = downloaded.columns.get_level_values(level)
        if ticker in vals:
            try:
                return downloaded.xs(ticker, axis=1, level=level, drop_level=True).copy()
            except Exception:
                pass
    return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def fast_batch_analyze(items):
    """CSV一括分析。企業スクレイピングをせず株価をまとめて取得する。"""
    items = [(normalize_code(c), str(n or "")) for c, n in items]
    items = [(c, n) for c, n in items if c]
    if not items:
        return []

    results = []
    batch_size = 80
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        tickers = [f"{c}.T" for c, _ in batch]
        try:
            downloaded = yf.download(
                tickers,
                period="1y",
                group_by="column",
                auto_adjust=False,
                progress=False,
                threads=True,
                timeout=30,
            )
        except Exception as exc:
            for code, name in batch:
                results.append({"code": code, "name": name, "error": f"一括取得失敗: {exc}"})
            continue

        for code, name in batch:
            ticker = f"{code}.T"
            frame = extract_ticker_frame(downloaded, ticker)
            try:
                results.append(technical_row(code, name, frame))
            except Exception as exc:
                results.append({"code": code, "name": name, "error": f"分析エラー: {exc}"})
    return results


@st.cache_data(ttl=900, show_spinner=False)
def load_detail_history(code):
    try:
        hist = yf.download(f"{code}.T", period="2y", auto_adjust=False, progress=False, timeout=20)
    except Exception:
        return pd.DataFrame()
    if isinstance(hist.columns, pd.MultiIndex):
        hist.columns = hist.columns.get_level_values(0)
    return hist.dropna(subset=["Open", "High", "Low", "Close"]).copy()


@st.cache_data(ttl=1800, show_spinner=False)
def load_info(code):
    try:
        return yf.Ticker(f"{code}.T").info or {}
    except Exception:
        return {}


def detailed_analysis(code):
    code = normalize_code(code)
    hist = load_detail_history(code)
    if hist.empty or len(hist) < 60:
        return {"error": "株価データが十分に取得できませんでした。"}
    info = load_info(code)
    base = technical_row(code, info.get("shortName") or code, hist)
    if base.get("error"):
        return base
    hist["MA25"] = hist["Close"].rolling(25).mean()
    hist["MA75"] = hist["Close"].rolling(75).mean()
    hist["MA200"] = hist["Close"].rolling(200).mean()
    hist["RSI"] = calculate_rsi(hist["Close"])
    mid = hist["Close"].rolling(25).mean()
    std = hist["Close"].rolling(25).std()
    hist["BB_upper"] = mid + 2 * std
    hist["BB_lower"] = mid - 2 * std
    base.update({
        "hist": hist,
        "sector": info.get("sector") or "—",
        "summary": info.get("longBusinessSummary") or "情報が取得できませんでした。",
        "per": info.get("trailingPE"),
        "pbr": info.get("priceToBook"),
        "roe": info.get("returnOnEquity"),
        "market_cap": info.get("marketCap"),
        "dividend_yield": info.get("dividendYield"),
    })
    return base


def chart(hist):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist.index,
        open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
        name="株価",
        increasing=dict(line=dict(color="#ef4444"), fillcolor="#ef4444"),
        decreasing=dict(line=dict(color="#3b82f6"), fillcolor="#3b82f6"),
    ))
    for label, col in (("MA25", "MA25"), ("MA75", "MA75"), ("BB +2σ", "BB_upper"), ("BB -2σ", "BB_lower")):
        if col in hist:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[col], name=label, line=dict(width=1.2)))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=30, b=20))
    return fig


def fmt_num(value, digits=1):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}"
    except Exception:
        return "—"


if "watchlist_df" not in st.session_state:
    st.session_state.watchlist_df = None
if "watchlist_code_col" not in st.session_state:
    st.session_state.watchlist_code_col = None
if "watchlist_name_col" not in st.session_state:
    st.session_state.watchlist_name_col = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None
if "analyze_code" not in st.session_state:
    st.session_state.analyze_code = None

st.markdown("### 📂 CSV買い銘柄リスト")
st.caption("SBIなどのCSVを読み込み、一括では株価・RSI・ボリンジャーバンドだけを高速分析します。")
uploaded = st.file_uploader("CSVファイルをアップロード", type=["csv"], label_visibility="collapsed")

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

    if st.button("📊 一括分析してランキング表示", type="primary", use_container_width=True):
        with st.spinner(f"{len(items)}銘柄を一括取得・分析しています…"):
            fast_batch_analyze.clear()
            st.session_state.batch_results = fast_batch_analyze(tuple(items))

if st.session_state.batch_results is not None:
    valid = [r for r in st.session_state.batch_results if not r.get("error") and r.get("momentum") is not None]
    invalid = [r for r in st.session_state.batch_results if r.get("error") or r.get("momentum") is None]
    valid.sort(key=lambda r: r["momentum"], reverse=True)

    st.markdown("### 🏆 逆張りチャンス ランキング")
    st.caption(f"分析成功 {len(valid)}銘柄 / 失敗 {len(invalid)}銘柄")

    if not valid:
        st.warning("ランキングを作れる銘柄がありませんでした。下の『分析できなかった銘柄』を確認してください。")
    else:
        table = pd.DataFrame([
            {
                "順位": i + 1,
                "コード": r["code"],
                "銘柄名": r["name"],
                "逆張りスコア": round(r["momentum"], 1),
                "RSI14": round(r["rsi"], 1),
                "現在値": round(r["current_price"], 1),
                "BB下限": round(r["bb_lower"], 1),
            }
            for i, r in enumerate(valid)
        ])
        st.dataframe(table, hide_index=True, use_container_width=True)

        st.markdown("#### 詳細分析する銘柄")
        top_show = valid[:50]
        labels = {r["code"]: f"{r['code']} {r['name']}｜スコア {r['momentum']:.1f}｜RSI {r['rsi']:.1f}" for r in top_show}
        selected = st.selectbox("銘柄を選択", list(labels), format_func=labels.get)
        if st.button("🔍 選択銘柄を詳細分析", use_container_width=True):
            st.session_state.analyze_code = selected

    if invalid:
        with st.expander(f"⚠️ 分析できなかった銘柄（{len(invalid)}件）"):
            for r in invalid[:100]:
                st.caption(f"{r.get('code','')} {r.get('name','')}: {r.get('error','不明なエラー')}")

st.markdown("---")
st.markdown("### 🔍 銘柄コードから個別分析")
col1, col2 = st.columns([3, 1])
with col1:
    manual = st.text_input("銘柄コード", placeholder="例：7203", max_chars=8)
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("分析を実行", type="secondary", use_container_width=True)
if run:
    code = normalize_code(manual)
    if code:
        st.session_state.analyze_code = code
    else:
        st.warning("4桁の銘柄コードを入力してください。")

code = st.session_state.analyze_code
if code:
    with st.spinner(f"{code} の詳細データを取得しています…"):
        detail = detailed_analysis(code)
    if detail.get("error"):
        st.error(detail["error"])
    else:
        st.markdown("---")
        st.subheader(f"🏢 {detail['name']} ({detail['code']})")
        st.caption(f"業種: {detail.get('sector', '—')}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("現在価格", f"¥{detail['current_price']:,.0f}")
        c2.metric("逆張りスコア", f"{detail['momentum']:.1f}/10")
        c3.metric("RSI14", f"{detail['rsi']:.1f}")
        c4.metric("BB下限", f"¥{detail['bb_lower']:,.0f}")

        st.plotly_chart(chart(detail["hist"].tail(180)), use_container_width=True)

        f1, f2, f3, f4 = st.columns(4)
        f1.metric("PER", fmt_num(detail.get("per")))
        f2.metric("PBR", fmt_num(detail.get("pbr")))
        f3.metric("ROE", "—" if detail.get("roe") is None else f"{detail['roe']*100:.1f}%")
        dy = detail.get("dividend_yield")
        if dy is not None:
            try:
                dy = float(dy)
                if dy > 1:
                    dy /= 100
                dy_text = f"{dy*100:.2f}%"
            except Exception:
                dy_text = "—"
        else:
            dy_text = "—"
        f4.metric("配当利回り", dy_text)

        st.markdown("#### 事業概要")
        st.write(detail.get("summary") or "情報が取得できませんでした。")
        st.caption(f"更新日時: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | 株価・企業情報: Yahoo Finance")
