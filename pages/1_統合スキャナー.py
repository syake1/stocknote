import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stocknote 統合スキャナー", layout="wide")
st.title("🧭 Stocknote 統合スキャナー")
st.caption("SBIなどのCSVを母集団にして、買い候補・空売り候補を同時抽出。上位候補はテクニカル社員＋ファンダメンタル社員で再評価します。")


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

    code_col = None
    name_col = None
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
    if name_col:
        out["銘柄名"] = df[name_col].fillna("").astype(str).str.strip()
    else:
        out["銘柄名"] = ""
    out["入力CSV"] = uploaded.name
    return out[out["コード"] != ""].reset_index(drop=True)


def load_universe(files):
    frames = [read_one_csv(f) for f in files]
    if not frames:
        return pd.DataFrame(columns=["コード", "銘柄名", "入力CSV"])
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("コード", keep="first").reset_index(drop=True)


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


def one_download(code):
    try:
        h = yf.download(f"{code}.T", period="14mo", auto_adjust=False, progress=False, threads=False, timeout=20)
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        return h
    except Exception:
        return pd.DataFrame()


def technical_scores(code, name, hist):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {"コード": code, "銘柄名": name or code, "error": "株価データなし"}
    hist = hist.copy()
    close = pd.to_numeric(hist["Close"], errors="coerce")
    high = pd.to_numeric(hist.get("High"), errors="coerce")
    low = pd.to_numeric(hist.get("Low"), errors="coerce")
    open_ = pd.to_numeric(hist.get("Open"), errors="coerce")
    volume = pd.to_numeric(hist.get("Volume"), errors="coerce")
    valid = close.notna()
    close = close[valid]
    if len(close) < 80:
        return {"コード": code, "銘柄名": name or code, "error": "履歴不足"}

    rsi = rsi14(close)
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    ma200 = close.rolling(200).mean()
    std25 = close.rolling(25).std()
    bb_upper = ma25 + 2 * std25
    bb_lower = ma25 - 2 * std25
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
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

    if volume is not None and len(volume) >= 25:
        v = volume.loc[valid].tail(len(close))
        v20 = float(v.iloc[-21:-1].mean()) if len(v) >= 21 else np.nan
        vr = float(v.iloc[-1] / v20) if v20 and np.isfinite(v20) and v20 > 0 else 1.0
    else:
        vr = 1.0

    prev_bear = False
    reversal = False
    upper_wick = False
    if open_ is not None and high is not None and low is not None:
        o = open_.loc[valid].tail(len(close))
        h = high.loc[valid].tail(len(close))
        l = low.loc[valid].tail(len(close))
        if len(o) >= 2 and pd.notna(o.iloc[-1]) and pd.notna(o.iloc[-2]):
            prev_bear = close.iloc[-2] < o.iloc[-2]
            reversal = bool(prev_bear and close.iloc[-1] > o.iloc[-1] and close.iloc[-1] >= o.iloc[-2] and o.iloc[-1] <= close.iloc[-2])
            body = abs(close.iloc[-1] - o.iloc[-1])
            upper_wick = bool((h.iloc[-1] - max(close.iloc[-1], o.iloc[-1])) > max(body * 1.5, px * 0.003))

    buy_rsi = float(np.clip((55 - rv) / 30 * 35, 0, 35))
    buy_bb = 25.0 if px <= blo * 1.02 else float(np.clip((m25 - px) / max(m25 - blo, 1e-9) * 20, 0, 20))
    buy_trend = 12.0 if m25 >= m75 else 4.0
    if np.isfinite(m200) and px >= m200:
        buy_trend += 5.0
    buy_macd = 8.0 if md >= sg else 2.0
    buy_volume = float(np.clip((vr - 0.8) * 6, 0, 8))
    buy_candle = 12.0 if reversal else 0.0
    buy_score = float(np.clip(buy_rsi + buy_bb + buy_trend + buy_macd + buy_volume + buy_candle, 0, 100))

    short_rsi = float(np.clip((rv - 55) / 25 * 35, 0, 35))
    short_bb = 25.0 if px >= bup * 0.98 else float(np.clip((px - m25) / max(bup - m25, 1e-9) * 20, 0, 20))
    short_trend = 12.0 if m25 <= m75 else 4.0
    if np.isfinite(m200) and px < m200:
        short_trend += 5.0
    short_macd = 8.0 if md < sg else 2.0
    short_volume = float(np.clip((vr - 0.8) * 6, 0, 8))
    short_candle = 12.0 if upper_wick and close.iloc[-1] < o.iloc[-1] else 0.0
    short_score = float(np.clip(short_rsi + short_bb + short_trend + short_macd + short_volume + short_candle, 0, 100))

    return {
        "コード": code, "銘柄名": name or code,
        "現在値": px, "RSI14": rv, "出来高倍率": vr,
        "MA25": m25, "MA75": m75, "MA200": m200,
        "BB下限": blo, "BB上限": bup,
        "MACD": md, "MACDシグナル": sg,
        "包み陽線": reversal, "上ヒゲ陰線": short_candle > 0,
        "買いスコア": buy_score, "空売りスコア": short_score,
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
            data = yf.download(tickers, period="14mo", group_by="ticker", auto_adjust=False, progress=False, threads=True, timeout=25)
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
    try:
        tk = yf.Ticker(f"{code}.T")
        info = tk.info or {}
    except Exception:
        return {"score": 50.0, "comment": "ファンダメンタル取得不可", "per": None, "pbr": None, "roe": None, "div": None}

    def num(key):
        try:
            v = info.get(key)
            return None if v is None or pd.isna(v) else float(v)
        except Exception:
            return None

    per = num("trailingPE")
    pbr = num("priceToBook")
    roe = num("returnOnEquity")
    opm = num("operatingMargins")
    growth = num("revenueGrowth")
    div = num("dividendYield")
    if div is not None and div > 1:
        div /= 100

    score = 50.0
    notes = []
    if per is not None:
        if 0 < per <= 15: score += 8; notes.append("PER割安")
        elif per >= 35: score -= 7; notes.append("PER高め")
    if pbr is not None:
        if 0 < pbr <= 1.2: score += 7; notes.append("PBR低め")
        elif pbr >= 4: score -= 6; notes.append("PBR高め")
    if roe is not None:
        if roe >= 0.10: score += 10; notes.append("ROE良好")
        elif roe < 0: score -= 12; notes.append("ROEマイナス")
    if opm is not None:
        if opm >= 0.10: score += 8; notes.append("営業利益率良好")
        elif opm < 0: score -= 10; notes.append("営業赤字")
    if growth is not None:
        if growth >= 0.05: score += 8; notes.append("増収")
        elif growth < 0: score -= 7; notes.append("減収")
    if div is not None and div >= 0.03:
        score += 5; notes.append("配当3%以上")

    score = float(np.clip(score, 0, 100))
    return {"score": score, "comment": "・".join(notes) if notes else "大きな加減点なし", "per": per, "pbr": pbr, "roe": roe, "div": div}


def meeting_rows(candidates, side):
    rows = []
    score_key = "買いスコア" if side == "buy" else "空売りスコア"
    top = sorted(candidates, key=lambda x: x.get(score_key, 0), reverse=True)[:5]
    for r in top:
        f = fundamental_employee(r["コード"])
        tech = float(r[score_key])
        if side == "buy":
            final = tech * 0.65 + f["score"] * 0.35
            fund_view = f["comment"]
        else:
            # 空売りでは強いファンダは逆風、弱いファンダは追い風として反転利用
            short_fund = 100 - f["score"]
            final = tech * 0.75 + short_fund * 0.25
            fund_view = "強い企業ほど空売りには逆風。" + f["comment"]
        rows.append({
            "コード": r["コード"], "銘柄名": r["銘柄名"],
            "テクニカル社員": round(tech, 1),
            "ファンダ社員": round(f["score"], 1),
            "最終評価": round(float(final), 1),
            "RSI14": round(r["RSI14"], 1),
            "コメント": fund_view,
        })
    return pd.DataFrame(rows).sort_values("最終評価", ascending=False, ignore_index=True) if rows else pd.DataFrame()


if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if "universe" not in st.session_state:
    st.session_state.universe = None

files = st.file_uploader("CSVをアップロード（複数可）", type=["csv"], accept_multiple_files=True)
if files:
    try:
        universe = load_universe(files)
        st.session_state.universe = universe
        st.success(f"{len(files)}ファイル・{len(universe)}銘柄を読み込みました。")
        st.dataframe(universe.head(20), hide_index=True, use_container_width=True)
    except Exception as exc:
        st.error(str(exc))

if st.session_state.universe is not None and not st.session_state.universe.empty:
    u = st.session_state.universe
    items = tuple((r["コード"], r["銘柄名"]) for _, r in u.iterrows())
    if st.button("🚀 買い・空売りを一括分析", type="primary", use_container_width=True):
        with st.spinner(f"{len(items)}銘柄を分析中…"):
            scan_items.clear()
            st.session_state.scan_results = scan_items(items)

if st.session_state.scan_results is not None:
    ok = [r for r in st.session_state.scan_results if not r.get("error")]
    bad = [r for r in st.session_state.scan_results if r.get("error")]
    st.caption(f"分析成功 {len(ok)}銘柄 / 失敗 {len(bad)}銘柄")
    if not ok:
        st.error("分析できた銘柄がありません。")
    else:
        buy = sorted(ok, key=lambda x: x["買いスコア"], reverse=True)
        short = sorted(ok, key=lambda x: x["空売りスコア"], reverse=True)
        tab_buy, tab_short, tab_meeting = st.tabs(["📈 買い候補", "📉 空売り候補", "👥 AI社員会議"])

        with tab_buy:
            st.subheader("買い候補ランキング")
            cols = ["コード", "銘柄名", "買いスコア", "RSI14", "現在値", "出来高倍率", "BB下限", "MA25", "MA75", "包み陽線"]
            st.dataframe(pd.DataFrame(buy)[cols].head(50), hide_index=True, use_container_width=True)

        with tab_short:
            st.subheader("空売り候補ランキング")
            cols = ["コード", "銘柄名", "空売りスコア", "RSI14", "現在値", "出来高倍率", "BB上限", "MA25", "MA75", "上ヒゲ陰線"]
            st.dataframe(pd.DataFrame(short)[cols].head(50), hide_index=True, use_container_width=True)
            st.caption("空売りは貸借銘柄・在庫・逆日歩など売建可否を証券会社で別途確認してください。")

        with tab_meeting:
            st.write("上位候補だけファンダメンタルを追加取得して最終評価します。")
            if st.button("👥 上位5銘柄をAI社員会議で再評価", use_container_width=True):
                with st.spinner("テクニカル社員とファンダメンタル社員が評価中…"):
                    st.markdown("#### 買い会議")
                    st.dataframe(meeting_rows(buy, "buy"), hide_index=True, use_container_width=True)
                    st.markdown("#### 空売り会議")
                    st.dataframe(meeting_rows(short, "short"), hide_index=True, use_container_width=True)
                    st.caption("AI社員は現時点ではルールベース評価です。最終判断や自動発注は行いません。")

    if bad:
        with st.expander(f"⚠️ 分析できなかった銘柄（{len(bad)}件）"):
            for r in bad[:100]:
                st.caption(f"{r.get('コード','')} {r.get('銘柄名','')}: {r.get('error','不明')}")

st.markdown("---")
st.caption(f"更新: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | 株価・ファンダメンタル: Yahoo Finance")
