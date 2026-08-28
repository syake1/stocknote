import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from stocknote_detail import render_extended_detail
from stocknote_fundamentals import get_fundamentals
from stocknote_universe import delete_universe as delete_saved_universe
from stocknote_universe import load_universe as load_saved_universe
from stocknote_universe import save_universe as save_saved_universe
from stocknote_tracking import load_active, merge_new_candidates

st.set_page_config(page_title="Stocknote 統合スキャナー", layout="wide")
st.title("🧭 Stocknote 統合スキャナー")
st.caption("保存したSBI CSVを母集団に、テクニカル・ファンダメンタル・市場環境を合わせて候補を評価します。")

st.markdown("## 📌 現在監視中の買い候補")
active_candidates = load_active()
if active_candidates:
    active_rows = [{
        "コード": r.get("code"), "銘柄名": r.get("name"),
        "現在値": r.get("current_price"), "騰落率%": r.get("return_pct"),
        "状態": r.get("status"), "初回検出日時": r.get("first_seen_at"),
        "最終更新日時": r.get("updated_at"),
    } for r in active_candidates]
    st.dataframe(pd.DataFrame(active_rows), hide_index=True, use_container_width=True)
else:
    st.info("監視候補を作成中です。保存済みSBIデータがあれば、この画面で自動的に再分析します。")

MARKETS = {
    "日経平均": "^N225", "日経225先物": "NKD=F", "SOX半導体指数": "^SOX",
    "S&P500": "^GSPC", "NASDAQ": "^IXIC", "VIX": "^VIX",
    "WTI原油": "CL=F", "ドル円": "JPY=X",
}


def normalize_code(value):
    text = str(value).strip().upper().replace(".T", "")
    text = re.sub(r"\.0$", "", text)
    m = re.search(r"([0-9A-Z]{4})", text)
    return m.group(1) if m else ""


def read_one_csv(uploaded):
    raw = uploaded.getvalue()
    last = None
    for enc in ("utf-8-sig", "cp932", "shift_jis"):
        try:
            df = pd.read_csv(io.BytesIO(raw), dtype=str, encoding=enc)
            break
        except Exception as exc:
            last = exc
    else:
        raise ValueError(f"{uploaded.name}: CSVを読み込めませんでした: {last}")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(how="all").copy()
    if df.empty:
        return pd.DataFrame(columns=["コード", "銘柄名", "入力CSV"])

    code_col = name_col = None
    for c in df.columns:
        key = c.lower()
        if code_col is None and key in {"code", "コード", "銘柄コード", "証券コード"}:
            code_col = c
        if name_col is None and key in {"name", "名称", "銘柄名", "会社名"}:
            name_col = c

    if code_col is None:
        best_col, best_count = None, 0
        for c in df.columns:
            count = df[c].astype(str).map(normalize_code).ne("").sum()
            if count > best_count:
                best_col, best_count = c, count
        code_col = best_col
    if code_col is None:
        raise ValueError(f"{uploaded.name}: 銘柄コード列を見つけられませんでした")

    if name_col is None:
        for c in df.columns:
            if c != code_col:
                name_col = c
                break

    out = pd.DataFrame()
    out["コード"] = df[code_col].map(normalize_code)
    out["銘柄名"] = df[name_col].fillna("").astype(str).str.strip() if name_col else ""
    out["入力CSV"] = uploaded.name
    return out[out["コード"] != ""].reset_index(drop=True)


def merge_uploaded(files):
    frames = [read_one_csv(f) for f in files]
    if not frames:
        return pd.DataFrame(columns=["コード", "銘柄名", "入力CSV"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("コード", keep="first").reset_index(drop=True)


def rsi14(close):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def batch_frame(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()
    for level in range(data.columns.nlevels):
        vals = data.columns.get_level_values(level)
        if ticker in vals:
            try:
                frame = data.xs(ticker, axis=1, level=level, drop_level=True).copy()
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = frame.columns.get_level_values(-1)
                return frame
            except Exception:
                pass
    return pd.DataFrame()


def one_download(code, period="14mo", interval="1d"):
    try:
        h = yf.download(f"{code}.T", period=period, interval=interval,
                        auto_adjust=False, progress=False, threads=False)
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        return h.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def technical_scores(code, name, hist):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {"コード": code, "銘柄名": name or code, "error": "株価データなし"}
    hist = hist.copy()
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(close) < 80:
        return {"コード": code, "銘柄名": name or code, "error": "履歴不足"}

    rsi = rsi14(close)
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    ma200 = close.rolling(200).mean()
    std25 = close.rolling(25).std()
    bb_upper = ma25 + 2 * std25
    bb_lower = ma25 - 2 * std25
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()

    px = float(close.iloc[-1])
    rv = float(rsi.iloc[-1])
    m25 = float(ma25.iloc[-1])
    m75 = float(ma75.iloc[-1])
    m200 = float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else np.nan
    blo = float(bb_lower.iloc[-1])
    bup = float(bb_upper.iloc[-1])
    md = float(macd.iloc[-1])
    sg = float(signal.iloc[-1])

    vr = 1.0
    if "Volume" in hist:
        v = pd.to_numeric(hist["Volume"], errors="coerce").dropna()
        if len(v) >= 21:
            avg = float(v.iloc[-21:-1].mean())
            if avg > 0:
                vr = float(v.iloc[-1] / avg)

    reversal = upper_wick_bear = False
    if all(c in hist.columns for c in ["Open", "High"]):
        o = pd.to_numeric(hist["Open"], errors="coerce").dropna()
        hi = pd.to_numeric(hist["High"], errors="coerce").dropna()
        if len(o) >= 2 and len(hi) >= 1:
            cprev, oprev = float(close.iloc[-2]), float(o.iloc[-2])
            cnow, onow = float(close.iloc[-1]), float(o.iloc[-1])
            reversal = bool(cprev < oprev and cnow > onow and cnow >= oprev and onow <= cprev)
            body = abs(cnow - onow)
            upper_wick = float(hi.iloc[-1]) - max(cnow, onow)
            upper_wick_bear = bool(cnow < onow and upper_wick > max(body * 1.5, px * 0.003))

    buy_rsi = float(np.clip((55 - rv) / 30 * 35, 0, 35))
    buy_bb = 25.0 if px <= blo * 1.02 else float(np.clip((m25 - px) / max(m25 - blo, 1e-9) * 20, 0, 20))
    buy_trend = (12.0 if m25 >= m75 else 4.0) + (5.0 if np.isfinite(m200) and px >= m200 else 0.0)
    buy_macd = 8.0 if md >= sg else 2.0
    buy_volume = float(np.clip((vr - 0.8) * 6, 0, 8))
    buy_candle = 12.0 if reversal else 0.0
    buy_score = float(np.clip(buy_rsi + buy_bb + buy_trend + buy_macd + buy_volume + buy_candle, 0, 100))

    short_rsi = float(np.clip((rv - 55) / 25 * 35, 0, 35))
    short_bb = 25.0 if px >= bup * 0.98 else float(np.clip((px - m25) / max(bup - m25, 1e-9) * 20, 0, 20))
    short_trend = (12.0 if m25 <= m75 else 4.0) + (5.0 if np.isfinite(m200) and px < m200 else 0.0)
    short_macd = 8.0 if md < sg else 2.0
    short_volume = float(np.clip((vr - 0.8) * 6, 0, 8))
    short_candle = 12.0 if upper_wick_bear else 0.0
    short_score = float(np.clip(short_rsi + short_bb + short_trend + short_macd + short_volume + short_candle, 0, 100))

    return {
        "コード": code, "銘柄名": name or code, "現在値": px, "RSI14": rv,
        "出来高倍率": vr, "MA25": m25, "MA75": m75, "MA200": m200,
        "BB下限": blo, "BB上限": bup, "MACD": md, "MACDシグナル": sg,
        "包み陽線": reversal, "上ヒゲ陰線": upper_wick_bear,
        "買いスコア": buy_score, "空売りスコア": short_score,
        "買い内訳": {"RSI": buy_rsi, "BB": buy_bb, "トレンド": buy_trend,
                     "MACD": buy_macd, "出来高": buy_volume, "ローソク足": buy_candle},
        "error": None,
    }


@st.cache_data(ttl=600, show_spinner=False)
def scan_items(items):
    items = [(normalize_code(c), str(n or "")) for c, n in items]
    items = [(c, n) for c, n in items if c]
    results = []
    for start in range(0, len(items), 20):
        batch = items[start:start + 20]
        tickers = [f"{c}.T" for c, _ in batch]
        try:
            data = yf.download(tickers, period="14mo", group_by="ticker",
                               auto_adjust=False, progress=False, threads=True)
        except Exception:
            data = pd.DataFrame()
        for code, name in batch:
            frame = batch_frame(data, f"{code}.T")
            if frame.empty or "Close" not in frame.columns:
                frame = one_download(code)
            try:
                results.append(technical_scores(code, name, frame))
            except Exception as exc:
                results.append({"コード": code, "銘柄名": name or code, "error": f"分析エラー: {exc}"})
    return results


@st.cache_data(ttl=1800, show_spinner=False)
def fundamental_employee(code):
    return get_fundamentals(code)


@st.cache_data(ttl=600, show_spinner=False)
def market_employee_score():
    data = {}
    for name, symbol in MARKETS.items():
        try:
            h = yf.download(symbol, period="2mo", interval="1d", auto_adjust=False,
                            progress=False, threads=False)
            if isinstance(h.columns, pd.MultiIndex):
                h.columns = h.columns.get_level_values(0)
            c = pd.to_numeric(h.get("Close"), errors="coerce").dropna()
            if len(c) >= 6:
                data[name] = {"now": float(c.iloc[-1]), "d5": float((c.iloc[-1] / c.iloc[-6] - 1) * 100)}
        except Exception:
            pass
    score = 50.0
    notes = []
    for name in ["S&P500", "NASDAQ", "日経平均", "日経225先物"]:
        v = data.get(name, {}).get("d5")
        if v is not None:
            if v >= 1.5: score += 4
            elif v <= -1.5: score -= 4
    sox = data.get("SOX半導体指数", {}).get("d5")
    if sox is not None:
        if sox >= 2: score += 8; notes.append("半導体強い")
        elif sox <= -2: score -= 8; notes.append("半導体弱い")
    vix = data.get("VIX", {}).get("now")
    if vix is not None:
        if vix >= 30: score -= 15; notes.append("VIX高水準")
        elif vix >= 22: score -= 8; notes.append("VIXやや高い")
        elif vix <= 16: score += 5; notes.append("VIX低位")
    oil = data.get("WTI原油", {}).get("d5")
    if oil is not None and oil >= 8:
        score -= 5; notes.append("原油急騰")
    fx = data.get("ドル円", {}).get("d5")
    if fx is not None:
        if 0.5 <= fx <= 3: score += 3; notes.append("円安傾向")
        elif fx <= -2: score -= 3; notes.append("円高進行")
    score = float(np.clip(score, 0, 100))
    regime = "🟢 リスクオン" if score >= 68 else "🔴 リスクオフ" if score <= 38 else "🟡 中立・混在"
    return score, regime, " / ".join(notes) if notes else "大きな偏りなし"


def combined_score(row, side, market_score):
    f = fundamental_employee(row["コード"])
    tech = float(row["買いスコア"] if side == "buy" else row["空売りスコア"])
    if side == "buy":
        final = tech * 0.50 + f["score"] * 0.35 + market_score * 0.15
    else:
        final = tech * 0.50 + (100 - f["score"]) * 0.35 + (100 - market_score) * 0.15
    return float(np.clip(final, 0, 100)), f


def meeting_rows(candidates, side, market_score):
    rows = []
    score_key = "買いスコア" if side == "buy" else "空売りスコア"
    for r in sorted(candidates, key=lambda x: x.get(score_key, 0), reverse=True)[:5]:
        final, f = combined_score(r, side, market_score)
        rows.append({
            "コード": r["コード"], "銘柄名": r["銘柄名"],
            "テクニカル社員": round(float(r[score_key]), 1),
            "ファンダ社員": round(f["score"], 1),
            "市場環境社員": round(market_score if side == "buy" else 100 - market_score, 1),
            "最終評価": round(final, 1), "RSI14": round(r["RSI14"], 1),
            "コメント": f["comment"],
        })
    return pd.DataFrame(rows).sort_values("最終評価", ascending=False, ignore_index=True) if rows else pd.DataFrame()


def fmt_num(v, suffix=""):
    return "—" if v is None else f"{float(v):.2f}{suffix}"


def fmt_pct(v):
    return "—" if v is None else f"{float(v) * 100:.1f}%"


def highlight_buy_score(row):
    """買い条件到達・接近をランキング上で色分けする。"""
    score = pd.to_numeric(row.get("買いスコア"), errors="coerce")
    if pd.notna(score) and score >= 75:
        return ["background-color: #fecaca; color: #7f1d1d; font-weight: 700"] * len(row)
    if pd.notna(score) and score >= 65:
        return ["background-color: #fef3c7; color: #78350f; font-weight: 700"] * len(row)
    return [""] * len(row)


def highlight_short_score(row):
    """空売り条件到達・接近をランキング上で青く色分けする。"""
    score = pd.to_numeric(row.get("空売りスコア"), errors="coerce")
    if pd.notna(score) and score >= 75:
        return ["background-color: #bfdbfe; color: #1e3a8a; font-weight: 700"] * len(row)
    if pd.notna(score) and score >= 65:
        return ["background-color: #dbeafe; color: #1e40af; font-weight: 700"] * len(row)
    return [""] * len(row)


def detail_history(code):
    h = one_download(code, "10mo", "1d")
    if h.empty or "Close" not in h:
        return pd.DataFrame()
    h = h.copy()
    c = pd.to_numeric(h["Close"], errors="coerce")
    h["MA25"] = c.rolling(25).mean()
    h["MA75"] = c.rolling(75).mean()
    h["MA200"] = c.rolling(200).mean()
    std = c.rolling(25).std()
    h["BB上限"] = h["MA25"] + 2 * std
    h["BB下限"] = h["MA25"] - 2 * std
    h["RSI14"] = rsi14(c)
    return h


def price_chart(h):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"], name="株価",
        increasing_line_color="#ef4444", decreasing_line_color="#3b82f6"))
    for col, label in [("MA25", "MA25"), ("MA75", "MA75"), ("MA200", "MA200"),
                       ("BB上限", "+2σ"), ("BB下限", "-2σ")]:
        fig.add_trace(go.Scatter(x=h.index, y=h[col], mode="lines", name=label))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def rsi_chart(h):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=h.index, y=h["RSI14"], mode="lines", name="RSI14"))
    fig.add_hline(y=30, line_dash="dash")
    fig.add_hline(y=70, line_dash="dash")
    fig.update_layout(height=260, yaxis_range=[0, 100], margin=dict(l=10, r=10, t=30, b=10))
    return fig


def score_radar(row, f, market_score):
    tech_parts = row.get("買い内訳", {})
    vals = [
        float(row["買いスコア"]), float(f["score"]), float(market_score),
        min(100.0, float(tech_parts.get("RSI", 0)) / 35 * 100),
        min(100.0, float(tech_parts.get("BB", 0)) / 25 * 100),
    ]
    cats = ["逆張りテクニカル", "ファンダメンタル", "市場環境", "RSI反発度", "BB押し目度"]
    fig = go.Figure(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]], fill="toself", name="評価"))
    fig.update_layout(height=360, polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                      showlegend=False, margin=dict(l=40, r=40, t=30, b=30))
    return fig


def show_buy_detail(row, market_score):
    final, f = combined_score(row, "buy", market_score)
    st.markdown(f"### {row['コード']} {row['銘柄名']} 総合分析")
    a, b, c, d = st.columns(4)
    a.metric("逆張りテクニカル", f"{row['買いスコア']:.1f}/100")
    b.metric("ファンダメンタル", f"{f['score']:.1f}/100")
    c.metric("市場環境", f"{market_score:.1f}/100")
    d.metric("総合評価", f"{final:.1f}/100")
    st.caption("総合評価 = テクニカル50% + ファンダメンタル35% + 市場環境15%")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("#### 📊 テクニカル詳細")
        t1, t2, t3, t4, t5, t6 = st.columns(6)
        t1.metric("RSI14", f"{row['RSI14']:.1f}")
        t2.metric("出来高倍率", f"{row['出来高倍率']:.2f}")
        t3.metric("MA25", f"¥{row['MA25']:,.0f}")
        t4.metric("MA75", f"¥{row['MA75']:,.0f}")
        t5.metric("BB下限", f"¥{row['BB下限']:,.0f}")
        t6.metric("MACD", f"{row['MACD']:.2f}")
        st.write("包み陽線: " + ("✅" if row.get("包み陽線") else "—"))
    with right:
        st.markdown("#### 🎯 逆張りチャンス評価")
        st.plotly_chart(score_radar(row, f, market_score), use_container_width=True,
                        config={"displayModeBar": False})

    st.markdown("#### 💰 ファンダメンタル分析")
    f1, f2, f3, f4, f5, f6, f7 = st.columns(7)
    f1.metric("PER", fmt_num(f["per"], "倍"))
    f2.metric("PBR", fmt_num(f["pbr"], "倍"))
    f3.metric("ROE", fmt_pct(f["roe"]))
    f4.metric("自己資本比率", fmt_pct(f["equity_ratio"]))
    f5.metric("営業利益率", fmt_pct(f["opm"]))
    f6.metric("売上成長率", fmt_pct(f["growth"]))
    f7.metric("配当利回り", fmt_pct(f["div"]))
    st.write("ファンダ社員所見: " + f["comment"])
    if f.get("sources"):
        with st.expander("ファンダメンタルの取得・計算元"):
            labels = {"per":"PER", "pbr":"PBR", "roe":"ROE", "equity_ratio":"自己資本比率",
                      "opm":"営業利益率", "growth":"売上成長率", "div":"配当利回り"}
            for key, source in f["sources"].items():
                if key in labels:
                    st.caption(f"{labels[key]}: {source}")

    h = detail_history(row["コード"])
    if not h.empty and all(c in h.columns for c in ["Open", "High", "Low", "Close"]):
        st.markdown("#### 📈 日足チャート：ローソク足・移動平均・ボリンジャーバンド")
        st.plotly_chart(price_chart(h.tail(180)), use_container_width=True, config={"displayModeBar": False})
        st.markdown("#### RSI14")
        st.plotly_chart(rsi_chart(h.tail(180)), use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("詳細チャート用の株価データを取得できませんでした。")

    render_extended_detail(row["コード"], row, f, final)


if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "universe" not in st.session_state:
    saved, meta = load_saved_universe()
    st.session_state.universe = saved
    st.session_state.saved_meta = meta

st.markdown("## 📁 SBI CSV母集団")
saved_meta = st.session_state.get("saved_meta") or {}
if st.session_state.universe is not None and not st.session_state.universe.empty:
    st.success(f"保存済み母集団: {len(st.session_state.universe)}銘柄 / 保存日時: {saved_meta.get('saved_at', '不明')}")
else:
    st.info("まだ保存済みCSVはありません。最初にSBIのCSVを登録してください。")

files = st.file_uploader("CSVを登録・差し替え（複数可）", type=["csv"], accept_multiple_files=True)
if files:
    try:
        universe = merge_uploaded(files)
        meta = save_saved_universe(universe, [f.name for f in files])
        st.session_state.universe = universe
        st.session_state.saved_meta = meta
        st.session_state.scan_results = None
        st.success(f"{len(files)}ファイル・{len(universe)}銘柄を保存しました。次回から再アップロード不要です。")
    except Exception as exc:
        st.error(str(exc))

c1, c2 = st.columns(2)
with c1:
    if st.button("🗑️ 登録CSVを削除", use_container_width=True):
        delete_saved_universe()
        st.session_state.universe = None
        st.session_state.saved_meta = None
        st.session_state.scan_results = None
        scan_items.clear()
        st.rerun()
with c2:
    run_now = st.button("🔄 今すぐ再分析", type="primary", use_container_width=True)

u = st.session_state.universe
performed_scan = False
if u is not None and not u.empty:
    with st.expander("保存中の銘柄を確認"):
        st.dataframe(u.head(100), hide_index=True, use_container_width=True)
    items = tuple((r["コード"], r["銘柄名"]) for _, r in u.iterrows())
    if run_now or st.session_state.scan_results is None:
        with st.spinner(f"保存済み{len(items)}銘柄を分析中…"):
            if run_now:
                scan_items.clear()
                fundamental_employee.clear()
                market_employee_score.clear()
            st.session_state.scan_results = scan_items(items)
            performed_scan = True

if st.session_state.scan_results is not None:
    ok = [r for r in st.session_state.scan_results if not r.get("error")]
    bad = [r for r in st.session_state.scan_results if r.get("error")]
    st.caption(f"分析成功 {len(ok)}銘柄 / 失敗 {len(bad)}銘柄")
    if not ok:
        st.error("分析できた銘柄がありません。")
    else:
        buy = sorted(ok, key=lambda x: x["買いスコア"], reverse=True)
        short = sorted(ok, key=lambda x: x["空売りスコア"], reverse=True)
        if performed_scan:
            new_candidates = []
            for r in buy:
                new_candidates.append({
                    "code": r["コード"], "name": r["銘柄名"], "price": r["現在値"],
                    "score": r["買いスコア"], "rsi": r["RSI14"],
                    "ma25": r["MA25"], "ma75": r["MA75"], "ma200": r["MA200"],
                    "macd": r["MACD"], "macd_signal": r["MACDシグナル"],
                    "volume_ratio": r["出来高倍率"],
                })
                if len(new_candidates) >= 10:
                    break
            merge_new_candidates(new_candidates)
        with st.spinner("市場環境を確認中…"):
            market_score, regime, market_note = market_employee_score()
        st.info(f"市場環境社員: {regime}  {market_score:.0f}/100　{market_note}")

        tab_buy, tab_short, tab_meeting = st.tabs(["📈 買い候補", "📉 空売り候補", "👥 AI社員会議"])
        with tab_buy:
            st.subheader("買い候補ランキング")
            st.caption("🔴 75点以上＝買い条件到達　🟡 65〜74.9点＝買い条件接近")
            cols = ["コード", "銘柄名", "買いスコア", "RSI14", "現在値", "出来高倍率",
                    "BB下限", "MA25", "MA75", "包み陽線"]
            buy_table = pd.DataFrame(buy)[cols].head(50)
            st.dataframe(buy_table.style.apply(highlight_buy_score, axis=1),
                         hide_index=True, use_container_width=True)
            labels = [f"{r['コード']} {r['銘柄名']}" for r in buy[:20]]
            selected = st.selectbox("🔎 上位候補の詳細を見る", labels, key="buy_detail")
            if selected:
                code = selected.split()[0]
                row = next(r for r in buy if r["コード"] == code)
                with st.spinner(f"{code} の詳細分析中…"):
                    show_buy_detail(row, market_score)

        with tab_short:
            st.subheader("空売り候補ランキング")
            st.caption("🔵 75点以上＝空売り条件到達　薄い青＝65〜74.9点・条件接近")
            cols = ["コード", "銘柄名", "空売りスコア", "RSI14", "現在値", "出来高倍率",
                    "BB上限", "MA25", "MA75", "上ヒゲ陰線"]
            short_table = pd.DataFrame(short)[cols].head(50)
            st.dataframe(short_table.style.apply(highlight_short_score, axis=1),
                         hide_index=True, use_container_width=True)
            st.caption("空売りは貸借銘柄・在庫・逆日歩など売建可否を証券会社で別途確認してください。")

        with tab_meeting:
            st.write("上位候補をテクニカル・ファンダメンタル・市場環境の3社員で再評価します。")
            if st.button("👥 上位5銘柄をAI社員会議で再評価", use_container_width=True):
                st.markdown("#### 買い会議")
                st.dataframe(meeting_rows(buy, "buy", market_score), hide_index=True, use_container_width=True)
                st.markdown("#### 空売り会議")
                st.dataframe(meeting_rows(short, "short", market_score), hide_index=True, use_container_width=True)

    if bad:
        with st.expander(f"⚠️ 分析できなかった銘柄（{len(bad)}件）"):
            for r in bad[:100]:
                st.caption(f"{r.get('コード', '')} {r.get('銘柄名', '')}: {r.get('error', '不明')}")

st.markdown("---")
st.caption(f"更新: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | 保存CSVは差し替えるまで継続利用 | ファンダメンタルはYahoo Finance＋財務諸表から補完計算")
