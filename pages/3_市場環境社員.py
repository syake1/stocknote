import math
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

from stocknote_sector_rotation import download_sector_rotation, rank_rotation_stocks
from stocknote_sector_entry import add_sector_candidate

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
DB_PATH = os.getenv("STOCKNOTE_DB_PATH", "data/stocknote.sqlite3")
ROTATION_RULE_VERSION = 2


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


@st.cache_data(ttl=600, show_spinner=False)
def load_sector_rotation(rule_version=ROTATION_RULE_VERSION):
    del rule_version
    return download_sector_rotation()


@st.cache_data(ttl=600, show_spinner=False)
def load_stock_chart(code):
    return _flat_history(f"{code}.T")


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


if st.button("🔄 市場環境を更新", type="primary", width="stretch"):
    load_market_data.clear()
    load_sector_rotation.clear()

with st.spinner("世界市場を確認しています…"):
    rows = load_market_data()
with st.spinner("日本株の業種別資金流入を確認しています…"):
    sector_rows, sector_date = load_sector_rotation(ROTATION_RULE_VERSION)

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

st.markdown("## 🔁 本日の業種循環")
if sector_rows:
    for row in sector_rows:
        if "1日上昇銘柄%" not in row:
            stocks = row.get("銘柄候補", [])
            row["1日上昇銘柄%"] = (
                sum(float(stock.get("1日%", 0) or 0) > 0 for stock in stocks)
                / len(stocks) * 100
                if stocks else 0.0
            )
        day1 = float(row.get("1日%", 0.0) or 0.0)
        day5 = float(row.get("5日%", 0.0) or 0.0)
        acceleration = day1 - (day5 - day1) / 4.0
        row.setdefault("1日加速pt", acceleration)
        row.setdefault("初動点", float(np.clip(
            50.0 + acceleration * 18.0
            + (float(row.get("1日上昇銘柄%", 50.0) or 50.0) - 50.0) * 0.25
            - max(day5 - 2.5, 0.0) * 5.0
            - max(float(row.get("20日%", 0.0) or 0.0) - 6.0, 0.0) * 2.0,
            0.0,
            100.0,
        )))
    rotation_early = [row for row in sector_rows if row["判定"] == "循環初動"]
    rotation_buys = [row for row in sector_rows if row["判定"] == "循環買い"]
    rotation_overheated = [row for row in sector_rows if row["判定"] == "上昇後・過熱"]
    rotation_active = [
        row for row in sector_rows if row["判定"] in {"循環初動", "循環買い"}
    ]
    rotation_stocks = rank_rotation_stocks(sector_rows, view["score"])
    early_top = sector_rows[:3]
    asof = pd.Timestamp(sector_date).strftime("%Y/%m/%d") if sector_date is not None else "最新日足"

    st.markdown("### 業種から銘柄チャートを見る")
    st.caption("業種を押し、次に銘柄を押すと日足チャートが表示されます。")
    sector_names = [row["業種・テーマ"] for row in sector_rows]
    chart_sector = st.pills(
        "業種",
        sector_names,
        default=sector_names[0],
        selection_mode="single",
        wrap=True,
        key="rotation_chart_sector",
    )
    if chart_sector:
        chart_sector_row = next(row for row in sector_rows if row["業種・テーマ"] == chart_sector)
        chart_stocks = chart_sector_row.get("銘柄候補", [])
        if chart_stocks:
            stock_labels = {
                f"{item['コード']} {item['銘柄名']}": item
                for item in chart_stocks
            }
            chart_stock_label = st.pills(
                "銘柄",
                list(stock_labels),
                default=next(iter(stock_labels)),
                selection_mode="single",
                wrap=True,
                key=f"rotation_chart_stock_{chart_sector}",
            )
            if chart_stock_label:
                chart_stock = stock_labels[chart_stock_label]
                with st.spinner(f"{chart_stock_label} のチャートを読み込んでいます…"):
                    price_history = load_stock_chart(chart_stock["コード"])
                if price_history.empty:
                    st.warning("この銘柄のチャートデータを取得できませんでした。")
                else:
                    price_chart = price_history[["Close"]].rename(columns={"Close": "終値"}).copy()
                    price_chart["5日線"] = price_chart["終値"].rolling(5).mean()
                    price_chart["20日線"] = price_chart["終値"].rolling(20).mean()
                    price_chart["60日線"] = price_chart["終値"].rolling(60).mean()
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("現在値", f"¥{chart_stock['現在値']:,.1f}")
                    m2.metric("1日", f"{chart_stock['1日%']:+.2f}%")
                    m3.metric("5日", f"{chart_stock['5日%']:+.2f}%")
                    m4.metric("20日", f"{chart_stock['20日%']:+.2f}%")
                    st.line_chart(
                        price_chart.tail(130),
                        y=["終値", "5日線", "20日線", "60日線"],
                        x_label="日付",
                        y_label="株価（円）",
                    )

    st.markdown("### 本日の循環初動（早い順）")
    if rotation_active:
        if rotation_early:
            names = "、".join(row["業種・テーマ"] for row in rotation_early)
            st.success(f"最優先・先回り監視（循環初動）：{names}")
        if rotation_buys:
            names = "、".join(row["業種・テーマ"] for row in rotation_buys)
            st.success(f"買い候補（循環買い）：{names}")
        if rotation_overheated:
            names = "、".join(row["業種・テーマ"] for row in rotation_overheated)
            st.warning(f"上昇後のため追わない：{names}")

        st.markdown("#### 本日の循環セクター銘柄")
        if view["score"] <= 38:
            st.warning("地合いはリスクオフです。下表は買いサインではなく監視優先です。")
        else:
            st.info("『循環初動』を最上位に表示します。5日線上抜けは初動確認、5日線付近は上抜け前の先回り監視です。パラボリックは参考表示だけにします。")
        if rotation_stocks:
            ranked_df = pd.DataFrame(rotation_stocks)
            if "業種判定" not in ranked_df.columns:
                sector_states = {
                    row["業種・テーマ"]: row["判定"] for row in sector_rows
                }
                ranked_df["業種判定"] = (
                    ranked_df["業種・テーマ"].map(sector_states).fillna("循環買い")
                )
            if "5日線上抜け" not in ranked_df.columns:
                ranked_df["5日線上抜け"] = False
            if "5日線乖離%" not in ranked_df.columns:
                ranked_df["5日線乖離%"] = float("nan")
            if "初動段階" not in ranked_df.columns:
                ranked_df["初動段階"] = "条件待ち"
            if "過熱判定" not in ranked_df.columns:
                ranked_df["過熱判定"] = "未過熱"
            if "1日加速pt" not in ranked_df.columns:
                ranked_df["1日加速pt"] = float("nan")
            ranked_df.insert(0, "順位", range(1, len(ranked_df) + 1))
            st.dataframe(
                ranked_df[["順位", "初動段階", "業種・テーマ", "業種判定", "コード", "銘柄名", "扱い", "過熱判定", "5日線上抜け", "5日線乖離%", "1日加速pt", "循環点", "現在値", "1日%", "5日%", "20日%"]],
                hide_index=True,
                width="stretch",
                column_config={
                    "循環点": st.column_config.NumberColumn(format="%.1f"),
                    "現在値": st.column_config.NumberColumn(format="¥%,.1f"),
                    "1日%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "5日%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "20日%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "5日線乖離%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "1日加速pt": st.column_config.NumberColumn(format="%+.2f"),
                },
            )
        else:
            st.info("循環初動・循環買い業種の中に、個別銘柄の条件がそろった候補はありません。")

        st.markdown("#### 循環初動・循環買いを買い候補へ入れる")
        selected_sector = st.selectbox(
            "循環セクター",
            [row["業種・テーマ"] for row in rotation_active],
        )
        selected_row = next(row for row in rotation_active if row["業種・テーマ"] == selected_sector)
        stock_options = rank_rotation_stocks([selected_row], view["score"], limit=20)
        if stock_options:
            selected_code = st.selectbox(
                "登録する代表銘柄",
                [item["コード"] for item in stock_options],
                format_func=lambda code: next(
                    f"{item['コード']} {item['銘柄名']}　現在値 ¥{item['現在値']:,.1f}　5日 {item['5日%']:+.2f}%"
                    for item in stock_options if item["コード"] == code
                ),
            )
            selected_stock = next(item for item in stock_options if item["コード"] == selected_code)
            if st.button("買い候補へ入れる（注文は送信しません）", type="primary", width="stretch"):
                added = add_sector_candidate(
                    DB_PATH,
                    selected_stock["コード"],
                    selected_stock["銘柄名"],
                    selected_sector,
                    asof.replace("/", "-"),
                    selected_stock["現在値"],
                )
                if added:
                    st.success(f"{selected_stock['コード']} {selected_stock['銘柄名']} を追跡へ登録しました。")
                else:
                    st.info("この銘柄は本日の循環候補として登録済みです。")
            st.page_link("pages/2_追跡・保有管理.py", label="追跡・保有管理で確認")
            st.caption("ここでは買い候補への記録だけを行います。証券会社への発注は行いません。")
    else:
        st.info("本日は『循環初動』『循環買い』の条件を満たす業種はありません。無理に業種買いをしない判定です。")
    st.caption(f"判定日：{asof}。代表銘柄の等ウェイト平均で、短期資金の向きを毎日判定しています。")

    st.markdown("#### 業種の初動順位")
    top_cols = st.columns(3)
    for column, row in zip(top_cols, early_top):
        column.metric(
            row["業種・テーマ"],
            row["判定"],
            f"初動点 {row.get('初動点', 0):.1f} / 1日加速 {row.get('1日加速pt', 0):+.2f}",
        )

    sector_df = pd.DataFrame(sector_rows)
    display_columns = [
        "業種・テーマ", "判定", "売買目安", "初動点", "1日加速pt", "1日%", "5日%", "20日%",
        "1日上昇銘柄%", "5日上昇銘柄%", "20日線上銘柄%", "取得銘柄数",
    ]
    st.dataframe(
        sector_df[display_columns],
        hide_index=True,
        width="stretch",
        column_config={
            "1日%": st.column_config.NumberColumn(format="%+.2f%%"),
            "5日%": st.column_config.NumberColumn(format="%+.2f%%"),
            "20日%": st.column_config.NumberColumn(format="%+.2f%%"),
            "初動点": st.column_config.NumberColumn(format="%.1f"),
            "1日加速pt": st.column_config.NumberColumn(format="%+.2f"),
            "1日上昇銘柄%": st.column_config.NumberColumn(format="%.0f%%"),
            "5日上昇銘柄%": st.column_config.NumberColumn(format="%.0f%%"),
            "20日線上銘柄%": st.column_config.NumberColumn(format="%.0f%%"),
        },
    )
    st.bar_chart(
        sector_df.set_index("業種・テーマ")[["5日%"]],
        horizontal=True,
    )
    st.caption("循環初動＝当日+0.15%以上、業種内の半数以上が上昇し、直近4日の平均ペースより当日が加速、かつ5日・20日で未過熱の業種。循環初動を最上位にし、5日+6%以上または20日+12%以上は『上昇後・過熱』として追いません。")
else:
    st.warning("業種別データを取得できませんでした。世界市場の判定だけを表示しています。")

st.markdown("## 📊 世界市場一覧")
if ok:
    df = pd.DataFrame(ok)
    show = df[["市場", "現在値", "1日%", "5日%", "20日%", "20日線上", "60日線上"]].copy()
    for c in ["現在値", "1日%", "5日%", "20日%"]:
        show[c] = pd.to_numeric(show[c], errors="coerce").round(2)
    st.dataframe(show, hide_index=True, width="stretch")
else:
    st.error("市場データを取得できませんでした。")

if bad:
    with st.expander(f"⚠️ 取得できなかった市場（{len(bad)}件）"):
        for r in bad:
            st.caption(f"{r['市場']} ({r['シンボル']})")

st.markdown("## 🧭 会議での使い方")
st.write("市場環境社員 → テクニカル社員 → ファンダメンタル社員の順に確認し、個別銘柄の最終判断を行います。市場環境がリスクオフでも機械的に全銘柄を売買禁止にはせず、個別の強さを残して評価します。")

st.caption(f"更新: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | データ: Yahoo Finance経由 | 自動発注は行いません")
