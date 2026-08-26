import math
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Stocknote 市場環境社員", layout="wide")
st.title("🌍 Stocknote 市場環境社員")
st.caption("世界の株価指数・日経先物・半導体・原油・為替・金利・商品をまとめて確認し、日本株への追い風／逆風を判定します。")

MARKETS = {
    "日経平均": "^N225",
    "日経225先物": "NKD=F",
    "SOX半導体指数": "^SOX",
    "S&P500": "^GSPC",
    "NASDAQ": "^IXIC",
    "NYダウ": "^DJI",
    "VIX": "^VIX",
    "WTI原油": "CL=F",
    "Brent原油": "BZ=F",
    "金": "GC=F",
    "銅": "HG=F",
    "ドル円": "JPY=X",
    "米10年金利": "^TNX",
    "米30年金利": "^TYX",
}


def _flat_history(symbol):
    try:
        h = yf.download(symbol, period="8mo", interval="1d", auto_adjust=False,
                        progress=False, threads=False, timeout=20)
        if h is None or h.empty:
            return pd.DataFrame()
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        if "Close" not in h.columns:
            return pd.DataFrame()
        h = h.dropna(subset=["Close"]).copy()
        return h
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_market_data():
    rows = []
    history_map = {}
    for name, symbol in MARKETS.items():
        h = _flat_history(symbol)
        history_map[name] = h
        if h.empty or len(h) < 25:
            rows.append({"市場": name, "シンボル": symbol, "取得": False})
            continue
        close = pd.to_numeric(h["Close"], errors="coerce").dropna()
        if len(close) < 25:
            rows.append({"市場": name, "シンボル": symbol, "取得": False})
            continue
        cur = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else cur
        p5 = float(close.iloc[-6]) if len(close) >= 6 else float(close.iloc[0])
        p20 = float(close.iloc[-21]) if len(close) >= 21 else float(close.iloc[0])
        ma20 = float(close.rolling(20).mean().iloc[-1])
        ma60 = float(close.rolling(60).mean().iloc[-1]) if len(close) >= 60 else np.nan
        rows.append({
            "市場": name,
            "シンボル": symbol,
            "現在値": cur,
            "1日%": (cur / prev - 1) * 100 if prev else np.nan,
            "5日%": (cur / p5 - 1) * 100 if p5 else np.nan,
            "20日%": (cur / p20 - 1) * 100 if p20 else np.nan,
            "MA20": ma20,
            "MA60": ma60,
            "20日線上": cur >= ma20,
            "60日線上": bool(cur >= ma60) if np.isfinite(ma60) else None,
            "取得": True,
        })
    return rows


def row_map(rows):
    return {r["市場"]: r for r in rows if r.get("取得")}


def pct(rows, name, key="1日%"):
    r = rows.get(name)
    if not r:
        return None
    v = r.get(key)
    try:
        return float(v) if v is not None and np.isfinite(float(v)) else None
    except Exception:
        return None


def market_employee(rows):
    m = row_map(rows)
    score = 50.0
    notes = []

    # 世界株・日本株
    for name in ["S&P500", "NASDAQ", "日経平均", "日経225先物"]:
        v5 = pct(m, name, "5日%")
        if v5 is None:
            continue
        if v5 >= 1.5:
            score += 4
        elif v5 <= -1.5:
            score -= 4

    # 半導体は日本株への影響が大きいため重め
    sox5 = pct(m, "SOX半導体指数", "5日%")
    sox20 = pct(m, "SOX半導体指数", "20日%")
    if sox5 is not None:
        if sox5 >= 2:
            score += 8; notes.append("半導体が強い")
        elif sox5 <= -2:
            score -= 8; notes.append("半導体が弱い")
    if sox20 is not None:
        if sox20 >= 5: score += 4
        elif sox20 <= -5: score -= 4

    # VIX
    vix = m.get("VIX")
    if vix:
        vv = float(vix["現在値"])
        if vv >= 30:
            score -= 15; notes.append("VIX高水準")
        elif vv >= 22:
            score -= 8; notes.append("VIXやや高い")
        elif vv <= 16:
            score += 5; notes.append("VIX低位")

    # ドル円: 円安は輸出株には追い風、急変は警戒
    fx5 = pct(m, "ドル円", "5日%")
    if fx5 is not None:
        if 0.5 <= fx5 <= 3:
            score += 3; notes.append("円安傾向")
        elif fx5 <= -2:
            score -= 3; notes.append("円高が進行")
        elif fx5 >= 4:
            score -= 2; notes.append("為替変動が急")

    # 原油急騰はコスト上昇として日本株全体には逆風寄り
    oil5 = pct(m, "WTI原油", "5日%")
    if oil5 is not None:
        if oil5 >= 8:
            score -= 5; notes.append("原油急騰")
        elif oil5 <= -8:
            notes.append("原油急落")

    score = float(np.clip(score, 0, 100))
    if score >= 68:
        regime = "🟢 リスクオン寄り"
        jp = "日本株には追い風がやや優勢。買い候補を通常どおり検討。"
    elif score <= 38:
        regime = "🔴 リスクオフ寄り"
        jp = "日本株には逆風が優勢。買いは絞り、空売り・現金比率も意識。"
    else:
        regime = "🟡 中立・混在"
        jp = "指数だけで方向を決めず、個別銘柄のテクニカルとファンダを優先。"

    sox_view = "データなし"
    if sox5 is not None:
        if sox5 >= 2:
            sox_view = "強い：半導体関連には追い風"
        elif sox5 <= -2:
            sox_view = "弱い：半導体関連は警戒"
        else:
            sox_view = "中立"

    oil_view = "データなし"
    if oil5 is not None:
        if oil5 >= 5:
            oil_view = "上昇：資源株には追い風、コスト高業種には逆風"
        elif oil5 <= -5:
            oil_view = "下落：資源株には逆風、輸送・製造コストには追い風"
        else:
            oil_view = "大きな変化なし"

    return {
        "score": score,
        "regime": regime,
        "jp": jp,
        "notes": notes,
        "sox": sox_view,
        "oil": oil_view,
    }


if st.button("🔄 市場環境を更新", type="primary", use_container_width=True):
    load_market_data.clear()

with st.spinner("世界市場を確認しています…"):
    rows = load_market_data()

ok = [r for r in rows if r.get("取得")]
bad = [r for r in rows if not r.get("取得")]
view = market_employee(rows)

st.markdown("## 👤 市場環境社員の判断")
c1, c2, c3 = st.columns(3)
c1.metric("市場環境スコア", f"{view['score']:.0f}/100")
c2.metric("地合い", view["regime"])
c3.metric("取得成功", f"{len(ok)}/{len(rows)}")
st.info(view["jp"])

ca, cb = st.columns(2)
with ca:
    st.markdown("### 💻 半導体")
    st.write(view["sox"])
with cb:
    st.markdown("### 🛢️ 原油")
    st.write(view["oil"])

if view["notes"]:
    st.caption("注目点: " + " / ".join(view["notes"]))

st.markdown("## 📊 世界市場一覧")
if ok:
    df = pd.DataFrame(ok)
    show = df[["市場", "現在値", "1日%", "5日%", "20日%", "20日線上", "60日線上"]].copy()
    for c in ["現在値", "1日%", "5日%", "20日%"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    st.dataframe(show, hide_index=True, use_container_width=True)
else:
    st.error("市場データを取得できませんでした。")

if bad:
    with st.expander(f"⚠️ 取得できなかった市場（{len(bad)}件）"):
        for r in bad:
            st.caption(f"{r['市場']} ({r['シンボル']})")

st.markdown("## 🧭 会議での使い方")
st.write("市場環境社員 → テクニカル社員 → ファンダメンタル社員の順に確認し、個別銘柄の最終判断を行います。市場環境がリスクオフでも機械的に全銘柄を売買禁止にはせず、個別の強さを残して評価します。")

st.caption(f"更新: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | データ: Yahoo Finance経由 | 自動発注は行いません")
