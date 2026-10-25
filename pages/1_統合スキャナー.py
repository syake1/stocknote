import io
import re
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from stocknote_detail import render_extended_detail
from stocknote_candidate_view import candidate_status, price_change, render_candidate_table, signed_change
from stocknote_decline_counter import (
    PRICE_LABELS,
    START_MODE_BB_UPPER,
    START_MODE_LABELS,
    START_MODE_RESET,
    calculate_decline_count,
    completed_daily_history,
    decline_summary,
    signal_event_records,
    strategy_trade_records,
    summarize_signal_events,
    summarize_strategy_trades,
)
from stocknote_fundamentals import get_fundamentals
from stocknote_universe import delete_universe as delete_saved_universe
from stocknote_universe import load_universe as load_saved_universe
from stocknote_universe import save_universe as save_saved_universe
from stocknote_tracking import load_active, merge_new_candidates
from stocknote_technicals import (
    SELECTION_BASELINE_DATE,
    baseline_pullback_zone,
    calculate as calculate_technical_snapshot,
    daily_trend_context,
    quarterly_strength_context,
)
from stocknote_short import assess_short, number
from stocknote_sector_rotation import download_sector_rotation, rank_rotation_stocks
from stocknote_rotation_history import analyze_rotation_patterns, load_rotation_history

st.set_page_config(page_title="Stocknote 統合スキャナー", layout="wide")
st.title("🧭 Stocknote 統合スキャナー")
st.caption("保存したSBI CSVを母集団に、テクニカル・ファンダメンタル・市場環境を合わせて候補を評価します。")
st.success(f"✅ 買い候補は{SELECTION_BASELINE_DATE.replace('-', '/')}基準：75日線付近の押し目＋反転確認")

SCANNER_RULE_VERSION = 7
ENTRY_RULE_VERSION = 5

with st.sidebar:
    st.markdown("### 下げ日数カウンター")
    decline_start_label = st.selectbox(
        "数え始めの起点",
        list(START_MODE_LABELS),
        index=0,
        help="BB上限モードは、ボリンジャーバンド(20日・2σ)の上限に高値が触れたあと、最初の陰線から数えます。",
    )
    decline_start_mode = START_MODE_LABELS[decline_start_label]
    decline_signal_count = st.slider("買いシグナルの本数", 5, 30, 10, 1)
    decline_add_count = st.slider("追加買いの本数", 10, 60, 20, 1)
    decline_compare_bars = st.slider("比較する足の本数", 1, 10, 2, 1)
    decline_price_label = st.selectbox("判定に使う価格", list(PRICE_LABELS), index=0)
    decline_candidate_range = st.slider(
        "買い候補に表示する範囲",
        1,
        60,
        (decline_signal_count, max(decline_signal_count, decline_add_count - 1)),
    )
    if decline_add_count <= decline_signal_count:
        st.warning("追加買いの本数は買いシグナルより大きくしてください。")
    st.caption("日足の確定後だけ更新します。証券会社への注文は送りません。")

decline_price_column = PRICE_LABELS[decline_price_label]

st.markdown("## 📌 現在監視中の候補")
stored_active_candidates = load_active()
# Hide candidates created by the previous high-zone rules. New-rule entries
# always carry both the MA75 touch flag and its current distance.
active_candidates = [
    r for r in stored_active_candidates
    if r.get("entry_rule_version") == ENTRY_RULE_VERSION
    and r.get("ma75_touched") is True
    and r.get("bb_lower_touched") is True
    and isinstance(r.get("ma75_distance_pct"), (int, float))
    and r["ma75_distance_pct"] <= 5.0
]
hidden_old_candidates = len(stored_active_candidates) - len(active_candidates)
if hidden_old_candidates:
    st.caption(f"旧条件の監視候補 {hidden_old_candidates}銘柄は非表示にしました。")
if active_candidates:
    render_candidate_table(active_candidates, key="scanner_candidates")

    # st.dataframeの行はスマホでは押しても詳細表示にならないため、
    # 明示的な選択欄を用意する。
    active_by_code = {str(r.get("code")): r for r in active_candidates if r.get("code")}
    if active_by_code:
        selected_active_code = st.selectbox(
            "👇 監視中の銘柄を選んで詳細を見る",
            list(active_by_code),
            format_func=lambda code: f"{code} {active_by_code[code].get('name') or ''}",
            key="active_candidate_detail",
        )
        selected_active = active_by_code[selected_active_code]
        with st.expander(
            f"📊 {selected_active_code} {selected_active.get('name') or ''} の詳細",
            expanded=True,
        ):
            st.markdown(
                f"### {selected_active_code} "
                f"{selected_active.get('name') or ''}"
            )
            a1, a2, a3, a4 = st.columns(4)
            price = selected_active.get("current_price")
            score = selected_active.get("score")
            rsi = selected_active.get("rsi")
            bb_position = selected_active.get("bb_position")
            amount_change, percent_change = price_change(selected_active)
            a1.metric("現在値", f"¥{price:,.0f}" if isinstance(price, (int, float)) else "—")
            a2.metric("逆張りスコア", f"{score:.1f}/100" if isinstance(score, (int, float)) else "—")
            a3.metric("RSI14", f"{rsi:.1f}" if isinstance(rsi, (int, float)) else "—")
            a4.metric("初回からの値上がり・値下がり", signed_change(amount_change), signed_change(percent_change, percent=True) if percent_change is not None else None)
            st.write(f"状態：**{candidate_status(selected_active)}**")
            st.caption(
                f"BB下限到達：確認済み（検出時の直近3営業日） / "
                f"現在のBB位置：{bb_position:.2f}σ" if isinstance(bb_position, (int, float))
                else "BB下限到達：確認済み（検出時の直近3営業日）"
            )
            st.caption(
                f"日足判定：{selected_active.get('trend_reason') or '—'}　／　"
                f"四半期足判定：{selected_active.get('quarterly_reason') or '—'}"
            )
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
    value = 100 - 100 / (1 + rs)
    value = value.mask((loss == 0) & (gain > 0), 100.0)
    value = value.mask((loss == 0) & (gain == 0), 50.0)
    return value.fillna(50)


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


def one_download(code, period="5y", interval="1d"):
    try:
        h = yf.download(f"{code}.T", period=period, interval=interval,
                        auto_adjust=False, progress=False, threads=False)
        if isinstance(h.columns, pd.MultiIndex):
            h.columns = h.columns.get_level_values(0)
        return h.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def technical_scores(
    code,
    name,
    hist,
    decline_signal_count=10,
    decline_add_count=20,
    decline_compare_bars=2,
    decline_price_column="Close",
    decline_start_mode=START_MODE_RESET,
):
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {"コード": code, "銘柄名": name or code, "error": "株価データなし"}
    hist = hist.copy()
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(close) < 80:
        return {"コード": code, "銘柄名": name or code, "error": "履歴不足"}

    completed_history = completed_daily_history(hist)
    counted_history = calculate_decline_count(
        completed_history,
        signal_count=decline_signal_count,
        add_count=decline_add_count,
        compare_bars=decline_compare_bars,
        price_column=decline_price_column,
        start_mode=decline_start_mode,
    )
    count_summary = decline_summary(
        counted_history,
        signal_count=decline_signal_count,
        add_count=decline_add_count,
    )
    count_events = signal_event_records(counted_history, thresholds=(8, 10, 12, 15), horizon=5)
    single_buy_trades = strategy_trade_records(
        counted_history,
        signal_count=decline_signal_count,
        add_count=decline_add_count,
        include_add=False,
    )
    added_buy_trades = strategy_trade_records(
        counted_history,
        signal_count=decline_signal_count,
        add_count=decline_add_count,
        include_add=True,
    )
    recent_count_rows = []
    for bar_time, bar in counted_history.tail(40).iterrows():
        event = ""
        if bool(bar["買い到達"]):
            event = f"買い{decline_signal_count}本"
        elif bool(bar["追加買い到達"]):
            event = f"追加{decline_add_count}本"
        elif bool(bar["リセット"]):
            event = "リセット"
        elif bool(bar["数え直し開始"]):
            event = "1本目"
        recent_count_rows.append({
            "日付": pd.Timestamp(bar_time).strftime("%Y/%m/%d"),
            "終値": float(bar["Close"]),
            "判定価格": float(bar[decline_price_column]),
            "下げ日数": int(bar["下げ日数"]),
            "判定": event,
        })

    rsi = rsi14(close)
    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    ma200 = close.rolling(200).mean()
    std25 = close.rolling(25).std()
    bb_upper = ma25 + 2 * std25
    bb_lower = ma25 - 2 * std25
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    trend = daily_trend_context(hist)
    quarterly = quarterly_strength_context(hist)
    technical_snapshot = calculate_technical_snapshot(code, name, hist)
    psar = technical_snapshot.get("psar") if technical_snapshot else np.nan
    psar_bull = bool(technical_snapshot and technical_snapshot.get("psar_bull"))
    psar_buy_turn = bool(technical_snapshot and technical_snapshot.get("psar_buy_turn"))
    psar_bar_time = technical_snapshot.get("psar_bar_time") if technical_snapshot else None

    px = float(close.iloc[-1])
    rv = float(rsi.iloc[-1])
    prev_px = float(close.iloc[-2])
    prev_rsi = float(rsi.iloc[-2])
    m5 = float(ma5.iloc[-1])
    ma5_cross_up = bool(prev_px <= float(ma5.iloc[-2]) and px > m5)
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

    reversal = lower_wick_reversal = upper_wick_bear = bearish_engulfing = False
    low = pd.to_numeric(hist["Low"], errors="coerce") if "Low" in hist else None
    if all(c in hist.columns for c in ["Open", "High"]):
        o = pd.to_numeric(hist["Open"], errors="coerce").dropna()
        hi = pd.to_numeric(hist["High"], errors="coerce").dropna()
        if len(o) >= 2 and len(hi) >= 1:
            cprev, oprev = float(close.iloc[-2]), float(o.iloc[-2])
            cnow, onow = float(close.iloc[-1]), float(o.iloc[-1])
            reversal = bool(cprev < oprev and cnow > onow and cnow >= oprev and onow <= cprev)
            bearish_engulfing = bool(cprev > oprev and cnow < onow and cnow <= oprev and onow >= cprev)
            body = abs(cnow - onow)
            upper_wick = float(hi.iloc[-1]) - max(cnow, onow)
            upper_wick_bear = bool(cnow < onow and upper_wick > max(body * 1.5, px * 0.003))
            if low is not None and pd.notna(low.iloc[-1]):
                body_bottom = min(cnow, onow)
                lower_wick = body_bottom - float(low.iloc[-1])
                lower_wick_reversal = bool(cnow >= onow and lower_wick >= max(body, px * 0.005))

    recent_ma75 = pd.concat(
        [low.rename("low"), ma75.rename("ma75")], axis=1
    ).dropna().tail(3) if low is not None else pd.DataFrame()
    ma75_touched = bool(not recent_ma75.empty and (recent_ma75["low"] <= recent_ma75["ma75"]).any())
    ma75_distance_pct = float((px / m75 - 1.0) * 100.0)
    bb_position = float((px - m25) / max(float(std25.iloc[-1]), 1e-9))
    recent_bb = pd.concat(
        [low.rename("low"), bb_lower.rename("lower")], axis=1
    ).dropna().tail(3) if low is not None else pd.DataFrame()
    bb_touch_mask = (
        recent_bb["low"] <= recent_bb["lower"]
        if not recent_bb.empty else pd.Series(dtype=bool)
    )
    bb_lower_touched = bool(not bb_touch_mask.empty and bb_touch_mask.any())
    bb_lower_touch_date = None
    if bb_lower_touched:
        bb_lower_touch_date = pd.Timestamp(recent_bb.index[bb_touch_mask][-1]).strftime("%Y/%m/%d")
    bb_lower_distance_pct = float((px / blo - 1.0) * 100.0)
    pullback_zone = baseline_pullback_zone(
        rv,
        px,
        m25,
        bb_position,
        ma75_touched,
        m75,
    )
    rebound_confirmed = bool(
        reversal or lower_wick_reversal or (px > prev_px and rv > prev_rsi)
    )

    buy_rsi = float(np.clip((55 - rv) / 30 * 35, 0, 35))
    buy_bb = 25.0 if px <= blo * 1.02 else float(np.clip((m25 - px) / max(m25 - blo, 1e-9) * 20, 0, 20))
    buy_trend = (12.0 if m25 >= m75 else 4.0) + (5.0 if np.isfinite(m200) and px >= m200 else 0.0)
    buy_macd = 8.0 if md >= sg else 2.0
    buy_volume = float(np.clip((vr - 0.8) * 6, 0, 8))
    buy_candle = 12.0 if reversal else 0.0
    ichimoku = 8.0 if trend and trend["tenkan_cross_up"] else 5.0 if trend and trend["tenkan_above_kijun"] else 0.0
    buy_score = float(np.clip(buy_rsi + buy_bb + buy_trend + buy_macd + buy_volume + buy_candle + ichimoku, 0, 100))
    quarterly_ok = bool(quarterly and quarterly["quarterly_qualified"])
    daily_bb_overextended = bool(px > bup * 1.03)
    buy_eligible = bool(
        trend and trend["buy_eligible"] and quarterly_ok
        and not daily_bb_overextended and pullback_zone and rebound_confirmed
    )
    buy_signal = bool(buy_eligible and ma5_cross_up)
    if not buy_eligible:
        buy_score = min(buy_score, 44.0)

    short_rsi = float(np.clip((rv - 55) / 25 * 35, 0, 35))
    short_bb = 25.0 if px >= bup * 0.98 else float(np.clip((px - m25) / max(bup - m25, 1e-9) * 20, 0, 20))
    short_trend = (12.0 if m25 <= m75 else 4.0) + (5.0 if np.isfinite(m200) and px < m200 else 0.0)
    short_macd = 8.0 if md < sg else 2.0
    short_volume = float(np.clip((vr - 0.8) * 6, 0, 8))
    short_candle = 12.0 if bearish_engulfing else 8.0 if upper_wick_bear else 0.0
    short_score = float(np.clip(short_rsi + short_bb + short_trend + short_macd + short_volume + short_candle, 0, 100))
    short_trend_ok = bool(trend and trend.get("short_eligible"))
    if not short_trend_ok:
        short_score = min(short_score, 44.0)
    elif trend.get("tenkan_cross_down"):
        short_score = min(100.0, short_score + 8.0)

    return {
        "コード": code, "銘柄名": name or code, "現在値": px, "RSI14": rv,
        "出来高倍率": vr, "MA5": m5, "MA25": m25, "MA75": m75, "MA200": m200,
        "BB下限": blo, "BB上限": bup, "MACD": md, "MACDシグナル": sg,
        "包み陽線": reversal, "下ヒゲ陽線": lower_wick_reversal,
        "上ヒゲ陰線": upper_wick_bear, "包み陰線": bearish_engulfing,
        "買いスコア": buy_score, "空売りスコア": short_score,
        "買い対象": buy_eligible, "押し目条件": pullback_zone,
        "買いサイン": buy_signal,
        "下げ日数": count_summary["current_count"],
        "下げ判定日": (
            count_summary["asof"].strftime("%Y/%m/%d")
            if count_summary["asof"] is not None else None
        ),
        "下げ買い到達日": (
            count_summary["signal_date"].strftime("%Y/%m/%d")
            if count_summary["signal_date"] is not None else None
        ),
        "下げ追加到達日": (
            count_summary["add_date"].strftime("%Y/%m/%d")
            if count_summary["add_date"] is not None else None
        ),
        "下げ買い本日到達": count_summary["signal_today"],
        "下げ追加本日到達": count_summary["add_today"],
        "下げ判定価格": decline_price_column,
        "_下げカウント推移": recent_count_rows,
        "_下げバックテスト": count_events,
        "_下げ単独買い取引": single_buy_trades,
        "_下げ追加買い取引": added_buy_trades,
        "5日線上抜け": ma5_cross_up,
        "5日線乖離%": (px / m5 - 1) * 100,
        "5日線判定日": str(close.index[-1]),
        "パラボリック": psar,
        "パラボリック買い状態": psar_bull,
        "パラボリック買い転換": psar_buy_turn,
        "パラボリック判定": "買い転換" if psar_buy_turn else "買い継続" if psar_bull else "売り",
        "パラボリック判定日": psar_bar_time,
        "反転確認": rebound_confirmed, "75日線接触": ma75_touched,
        "BB下限到達": bb_lower_touched,
        "BB下限到達日": bb_lower_touch_date,
        "BB下限乖離%": bb_lower_distance_pct,
        "75日線乖離%": ma75_distance_pct, "BB位置σ": bb_position,
        "四半期足強度": quarterly["quarterly_score"] if quarterly else 0.0,
        "四半期足適合": quarterly_ok,
        "四半期足判定": quarterly["quarterly_reason"] if quarterly else "四半期足の履歴不足",
        "四半期足形状": quarterly["quarterly_pattern"] if quarterly else "履歴不足",
        "高値圏持ち合い": quarterly["quarterly_high_level_consolidation"] if quarterly else False,
        "四半期RSI14": quarterly["quarterly_rsi"] if quarterly else np.nan,
        "四半期MACD上向き": quarterly["quarterly_macd_up"] if quarterly else False,
        "四半期大上ヒゲ": quarterly["quarterly_large_upper_wick"] if quarterly else False,
        "月足大上ヒゲ": quarterly["monthly_large_upper_wick"] if quarterly else False,
        "日足BB上抜け過熱": daily_bb_overextended,
        "日足BB上方乖離%": float((px / bup - 1) * 100),
        "確定四半期": quarterly["quarterly_last_confirmed"] if quarterly else None,
        "一目位置": trend["cloud_position"] if trend else "データ不足",
        "転換線": trend["tenkan"] if trend else np.nan,
        "基準線": trend["kijun"] if trend else np.nan,
        "転換線上抜け": bool(trend and trend["tenkan_cross_up"]),
        "転換線下抜け": bool(trend and trend.get("tenkan_cross_down")),
        "空売りトレンド適合": short_trend_ok,
        "トレンド判定": trend["trend_reason"] if trend else "日足履歴不足",
        "買い内訳": {"RSI": buy_rsi, "BB": buy_bb, "トレンド": buy_trend,
                     "MACD": buy_macd, "出来高": buy_volume, "ローソク足": buy_candle},
        "空売り内訳": {"RSI": short_rsi, "BB": short_bb, "トレンド": short_trend,
                       "MACD": short_macd, "出来高": short_volume, "ローソク足": short_candle},
        "error": None,
    }


@st.cache_data(ttl=600, show_spinner=False)
def scan_items(
    items,
    decline_signal_count=10,
    decline_add_count=20,
    decline_compare_bars=2,
    decline_price_column="Close",
    decline_start_mode=START_MODE_RESET,
):
    items = [(normalize_code(c), str(n or "")) for c, n in items]
    items = [(c, n) for c, n in items if c]
    results = []
    for start in range(0, len(items), 20):
        batch = items[start:start + 20]
        tickers = [f"{c}.T" for c, _ in batch]
        try:
            data = yf.download(tickers, period="5y", group_by="ticker",
                               auto_adjust=False, progress=False, threads=4)
        except Exception:
            data = pd.DataFrame()
        for code, name in batch:
            frame = batch_frame(data, f"{code}.T")
            if frame.empty or "Close" not in frame.columns:
                frame = one_download(code)
            try:
                results.append(technical_scores(
                    code,
                    name,
                    frame,
                    decline_signal_count,
                    decline_add_count,
                    decline_compare_bars,
                    decline_price_column,
                    decline_start_mode,
                ))
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


@st.cache_data(ttl=600, show_spinner=False)
def sector_rotation_employee():
    return download_sector_rotation()


def combined_score(row, side, market_score):
    f = row.get("_fundamentals") or fundamental_employee(row["コード"])
    tech = float(row["買いスコア"] if side == "buy" else row["空売りスコア"])
    if side == "buy":
        final = tech * 0.80 + f["score"] * 0.20
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
    """黒背景を保ち、買い候補の識別列だけを赤くする。"""
    score = pd.to_numeric(row.get("買いスコア"), errors="coerce")
    styles = [""] * len(row)
    if pd.notna(score) and score >= 75:
        color = "color: #ff5252; font-weight: 800"
    elif pd.notna(score) and score >= 65:
        color = "color: #ff9e9e; font-weight: 700"
    else:
        return styles
    for key in ("コード", "銘柄名", "買いスコア"):
        if key in row.index:
            styles[row.index.get_loc(key)] = color
    return styles


def highlight_short_score(row):
    """黒背景を保ち、空売り候補の識別列だけを青くする。"""
    score = pd.to_numeric(row.get("空売りスコア"), errors="coerce")
    styles = [""] * len(row)
    if pd.notna(score) and score >= 75:
        color = "color: #42a5f5; font-weight: 800"
    elif pd.notna(score) and score >= 65:
        color = "color: #90caf9; font-weight: 700"
    else:
        return styles
    for key in ("コード", "銘柄名", "空売りスコア"):
        if key in row.index:
            styles[row.index.get_loc(key)] = color
    return styles


TIMEFRAME_CONFIG = {
    "日足": {"period": "2y", "interval": "1d", "windows": (25, 75, 200), "bars": 180},
    "週足": {"period": "10y", "interval": "1wk", "windows": (13, 26, 52), "bars": 156},
    "月足": {"period": "max", "interval": "1mo", "windows": (6, 12, 24), "bars": 120},
    "四半期足": {"period": "max", "interval": "1mo", "windows": (4, 8, 12), "bars": 80},
}


def detail_history(code, timeframe="日足"):
    config = TIMEFRAME_CONFIG[timeframe]
    h = one_download(code, config["period"], config["interval"])
    if h.empty or "Close" not in h:
        return pd.DataFrame()
    h = h.copy()
    if timeframe == "四半期足":
        aggregations = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
        if "Volume" in h.columns:
            aggregations["Volume"] = "sum"
        h = h.resample("QE-DEC").agg(aggregations).dropna(subset=["Close"])
    c = pd.to_numeric(h["Close"], errors="coerce")
    short, middle, long = config["windows"]
    h["MA短期"] = c.rolling(short).mean()
    h["MA中期"] = c.rolling(middle).mean()
    h["MA長期"] = c.rolling(long).mean()
    std = c.rolling(middle).std()
    h["BB上限"] = h["MA中期"] + 2 * std
    h["BB下限"] = h["MA中期"] - 2 * std
    h["RSI14"] = rsi14(c)
    if timeframe == "日足":
        h["MA5"] = c.rolling(5).mean()
    return h


WEEKLY_SHADOW_STYLE = {
    "上向き": {"fill": "#ef4444", "badge": "rgba(239, 68, 68, 0.18)", "label": "買い候補"},
    "下向き": {"fill": "#3b82f6", "badge": "rgba(59, 130, 246, 0.18)", "label": "空売り候補"},
    "横ばい": {"fill": "#94a3b8", "badge": "rgba(148, 163, 184, 0.18)", "label": "様子見"},
}


def weekly_trend_shadow(h):
    """Map the last confirmed weekly trend to each daily bar without look-ahead."""
    if h.empty or "Close" not in h.columns or not isinstance(h.index, pd.DatetimeIndex):
        return pd.Series(dtype="object")

    close = pd.to_numeric(h["Close"], errors="coerce").dropna().sort_index()
    if close.empty:
        return pd.Series(dtype="object")

    weekly_close = close.resample("W-FRI").last().dropna()
    weekly_ma13 = weekly_close.rolling(13, min_periods=13).mean()
    weekly_slope = weekly_ma13.diff()
    weekly_state = pd.Series("横ばい", index=weekly_close.index, dtype="object")
    weekly_state[(weekly_close > weekly_ma13) & (weekly_slope > 0)] = "上向き"
    weekly_state[(weekly_close < weekly_ma13) & (weekly_slope < 0)] = "下向き"

    daily_state = weekly_state.reindex(h.index, method="ffill").fillna("横ばい")
    daily_state.index = h.index
    return daily_state


def add_weekly_shadow(fig, h):
    states = weekly_trend_shadow(h)
    if states.empty:
        return

    start = 0
    for position in range(1, len(states) + 1):
        if position < len(states) and states.iloc[position] == states.iloc[start]:
            continue
        state = states.iloc[start]
        x0 = h.index[start]
        x1 = h.index[position] if position < len(h) else h.index[-1] + pd.Timedelta(days=1)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=WEEKLY_SHADOW_STYLE[state]["fill"],
            opacity=0.055 if state != "横ばい" else 0.025,
            layer="below",
            line_width=0,
        )
        start = position

    current = states.iloc[-1]
    fig.add_annotation(
        x=0.01,
        y=0.99,
        xref="paper",
        yref="paper",
        text=f"週足の影: {WEEKLY_SHADOW_STYLE[current]['label']}（{current}・13週線）",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        bgcolor=WEEKLY_SHADOW_STYLE[current]["badge"],
        bordercolor=WEEKLY_SHADOW_STYLE[current]["fill"],
        borderwidth=1,
        borderpad=5,
        font=dict(size=12),
    )


def price_chart(h, timeframe):
    windows = TIMEFRAME_CONFIG[timeframe]["windows"]
    fig = go.Figure()
    if timeframe == "日足":
        add_weekly_shadow(fig, h)
    fig.add_trace(go.Candlestick(
        x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=h["Close"], name="株価",
        increasing_line_color="#ef4444", increasing_fillcolor="#ef4444",
        decreasing_line_color="#3b82f6", decreasing_fillcolor="#3b82f6"))
    if timeframe == "日足" and "MA5" in h:
        fig.add_trace(go.Scatter(x=h.index, y=h["MA5"], mode="lines", name="MA5", line=dict(width=2)))
    for col, label in [("MA短期", f"MA{windows[0]}"), ("MA中期", f"MA{windows[1]}"), ("MA長期", f"MA{windows[2]}"),
                       ("BB上限", "+2σ"), ("BB下限", "-2σ")]:
        fig.add_trace(go.Scatter(x=h.index, y=h[col], mode="lines", name=label))
    fig.update_layout(height=520, xaxis_rangeslider_visible=False,
                      margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h"))
    return fig


def stock_identity_from_row(row):
    """Return a normalized stock code and name from any scanner table row."""
    code = ""
    for column in ("コード", "code", "銘柄"):
        if column in row and not pd.isna(row.get(column)):
            code = normalize_code(row.get(column))
            if code:
                break

    name = ""
    for column in ("銘柄名", "name"):
        if column in row and not pd.isna(row.get(column)):
            name = str(row.get(column)).strip()
            if name:
                break
    if not name and "銘柄" in row and not pd.isna(row.get("銘柄")):
        name = re.sub(rf"^\s*{re.escape(code)}\s*", "", str(row.get("銘柄"))).strip()
    return code, name


def render_selected_daily_chart(code, name, key):
    with st.container(border=True):
        st.markdown(f"#### {code} {name} 日足チャート")
        with st.spinner(f"{code} のチャートを取得中…"):
            h = detail_history(code, "日足")
        if h.empty or not all(column in h.columns for column in ("Open", "High", "Low", "Close")):
            st.warning("この銘柄のチャートデータを取得できませんでした。")
            return
        st.plotly_chart(
            price_chart(h.tail(TIMEFRAME_CONFIG["日足"]["bars"]), "日足"),
            width="stretch",
            key=f"{key}_chart_{code}",
            config={"displayModeBar": False},
        )
        st.caption("行の選択を外すとチャートを閉じられます。")


def render_clickable_stock_table(
    frame,
    *,
    key,
    height=None,
    column_config=None,
    style_func=None,
    show_chart=True,
):
    """Render a stock table whose selected row opens its daily chart."""
    table = pd.DataFrame(frame).reset_index(drop=True)
    st.caption("👆 見たい銘柄の行をクリックすると、すぐ下に日足チャートが出ます。")
    display_table = table.style.apply(style_func, axis=1) if style_func else table
    dataframe_args = {
        "hide_index": True,
        "width": "stretch",
        "key": key,
        "on_select": "rerun",
        "selection_mode": "single-row",
        "alt": "銘柄を選択すると日足チャートが開く一覧表",
    }
    if height is not None:
        dataframe_args["height"] = height
    if column_config:
        dataframe_args["column_config"] = column_config
    event = st.dataframe(display_table, **dataframe_args)

    selected_rows = event.selection.rows
    if not selected_rows:
        return None
    try:
        row_number = int(selected_rows[0])
    except (TypeError, ValueError):
        return None
    if not 0 <= row_number < len(table):
        return None
    code, name = stock_identity_from_row(table.iloc[row_number])
    if not code:
        st.warning("選択した行の銘柄コードを読み取れませんでした。")
        return None
    if show_chart:
        render_selected_daily_chart(code, name, key)
    return code


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
    a.metric("押し目テクニカル", f"{row['買いスコア']:.1f}/100")
    b.metric("ファンダメンタル", f"{f['score']:.1f}/100")
    c.metric("市場環境", f"{market_score:.1f}/100")
    d.metric("総合評価", f"{final:.1f}/100")
    st.caption("総合評価 = 逆張りテクニカル80% + ファンダメンタル20%（市場環境は参考表示）")

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
        st.markdown("#### 🎯 上昇トレンド押し目評価")
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

    timeframe = st.selectbox("チャートの期間", list(TIMEFRAME_CONFIG), key=f"timeframe_{row['コード']}")
    h = detail_history(row["コード"], timeframe)
    if not h.empty and all(c in h.columns for c in ["Open", "High", "Low", "Close"]):
        st.markdown(f"#### 📈 {timeframe}チャート：ローソク足・移動平均・ボリンジャーバンド")
        last = h.dropna(subset=["Open", "Close"]).iloc[-1]
        last_date = pd.Timestamp(last.name).strftime("%Y/%m/%d")
        direction = "赤・上昇足" if float(last["Close"]) >= float(last["Open"]) else "青・下降足"
        forming = ""
        today = pd.Timestamp.now()
        if timeframe == "日足" and pd.Timestamp(last.name).date() >= today.date():
            forming = "・形成中"
        elif timeframe == "週足" and pd.Timestamp(last.name).to_period("W") == today.to_period("W"):
            forming = "・形成中"
        elif timeframe == "月足" and pd.Timestamp(last.name).to_period("M") == today.to_period("M"):
            forming = "・形成中"
        elif timeframe == "四半期足" and pd.Timestamp(last.name).to_period("Q") == today.to_period("Q"):
            forming = "・形成中"
        st.caption(f"最終足 {last_date}{forming}：始値 ¥{float(last['Open']):,.0f} → 終値 ¥{float(last['Close']):,.0f}（{direction}）")
        if forming:
            st.warning(f"現在の{timeframe}は未確定です。上ヒゲや終値は期間終了まで変化します。")
        bars = TIMEFRAME_CONFIG[timeframe]["bars"]
        st.plotly_chart(price_chart(h.tail(bars), timeframe), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f"#### {timeframe} RSI14")
        st.plotly_chart(rsi_chart(h.tail(bars)), use_container_width=True, config={"displayModeBar": False})
    else:
        st.warning("詳細チャート用の株価データを取得できませんでした。")

    render_extended_detail(row["コード"], row, f, final)


if "scan_results" not in st.session_state:
    st.session_state.scan_results = None
if st.session_state.get("scanner_rule_version") != SCANNER_RULE_VERSION:
    st.session_state.scan_results = None
    st.session_state.scanner_rule_version = SCANNER_RULE_VERSION
    scan_items.clear()
decline_setting_signature = (
    decline_signal_count,
    decline_add_count,
    decline_compare_bars,
    decline_price_column,
    decline_start_mode,
)
if st.session_state.get("decline_setting_signature") != decline_setting_signature:
    st.session_state.scan_results = None
    st.session_state.decline_setting_signature = decline_setting_signature
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
        render_clickable_stock_table(
            u,
            key="saved_universe_stocks",
            height=650,
        )
    items = tuple((r["コード"], r["銘柄名"]) for _, r in u.iterrows())
    if run_now or st.session_state.scan_results is None:
        with st.spinner(f"保存済み{len(items)}銘柄を分析中…"):
            if run_now:
                scan_items.clear()
                fundamental_employee.clear()
                market_employee_score.clear()
            st.session_state.scan_results = scan_items(
                items,
                decline_signal_count,
                decline_add_count,
                decline_compare_bars,
                decline_price_column,
                decline_start_mode,
            )
            performed_scan = True

if st.session_state.scan_results is not None:
    ok = [r for r in st.session_state.scan_results if not r.get("error")]
    bad = [r for r in st.session_state.scan_results if r.get("error")]
    st.caption(f"分析成功 {len(ok)}銘柄 / 失敗 {len(bad)}銘柄")
    if not ok:
        st.error("分析できた銘柄がありません。")
    else:
        universe_by_code = {str(r.get("コード", "")).strip(): r.to_dict() for _, r in u.iterrows()}
        # A known forecast PER above 25 is watch-only for the user's value-oriented buy rules.
        for r in ok:
            # session_state may still contain scan rows created before the
            # quarterly pattern fields were introduced.  Supply compatible
            # defaults so a hot Streamlit redeploy never crashes mid-scan.
            r.setdefault("四半期足形状", "上昇トレンド" if r.get("四半期足適合") else "対象外")
            r.setdefault("高値圏持ち合い", False)
            source = universe_by_code.get(str(r["コード"]), {})
            forecast_per = number(source.get("PER(株価収益率)(予)(倍)"))
            r["予想PER"] = forecast_per
            if forecast_per is not None and forecast_per > 25 and r.get("買い対象"):
                r["買い対象"] = False
                r["トレンド判定"] = "予想PER25倍超（買いは監視のみ）"
            r.setdefault("下げ日数", 0)
            r.setdefault("下げ判定日", None)
            r.setdefault("下げ買い到達日", None)
            r.setdefault("下げ追加到達日", None)
            r.setdefault("下げ買い本日到達", False)
            r.setdefault("下げ追加本日到達", False)
            r.setdefault("BB下限到達日", None)
            r.setdefault("BB下限乖離%", np.nan)
            r.setdefault("_下げカウント推移", [])
            r.setdefault("_下げバックテスト", [])
            r.setdefault("_下げ単独買い取引", [])
            r.setdefault("_下げ追加買い取引", [])

            count = int(r.get("下げ日数") or 0)
            if count == decline_add_count:
                count_status = f"追加買い{decline_add_count}本到達"
            elif count > decline_add_count:
                count_status = "追加買い到達後"
            elif count == decline_signal_count:
                count_status = f"買い{decline_signal_count}本到達"
            elif decline_signal_count < count < decline_add_count:
                count_status = "買い候補継続"
            else:
                count_status = "到達待ち"
            r["下げカウント状態"] = count_status

        count_min, count_max = decline_candidate_range
        decline_candidates = sorted(
            [r for r in ok if count_min <= int(r.get("下げ日数") or 0) <= count_max],
            key=lambda row: (
                not bool(row.get("下げ買い本日到達")),
                int(row.get("下げ日数") or 0),
                str(row.get("コード", "")),
            ),
        )
        decline_add_candidates = sorted(
            [r for r in ok if int(r.get("下げ日数") or 0) >= decline_add_count],
            key=lambda row: (-int(row.get("下げ日数") or 0), str(row.get("コード", ""))),
        )
        bb_lower_candidates = sorted(
            [r for r in ok if bool(r.get("BB下限到達"))],
            key=lambda row: (
                str(row.get("BB下限到達日") or ""),
                -float(row.get("BB位置σ") or 0.0),
            ),
            reverse=True,
        )
        decline_backtest_events = []
        decline_single_trades = []
        decline_added_trades = []
        for r in ok:
            identity = {"コード": r["コード"], "銘柄名": r["銘柄名"]}
            decline_backtest_events.extend(
                [{**identity, **event} for event in r.get("_下げバックテスト", [])]
            )
            decline_single_trades.extend(
                [{**identity, **trade} for trade in r.get("_下げ単独買い取引", [])]
            )
            decline_added_trades.extend(
                [{**identity, **trade} for trade in r.get("_下げ追加買い取引", [])]
            )
        buy_pre = [r for r in ok if r.get("買い対象")]
        with st.spinner("逆張り候補のファンダメンタルを確認中…"):
            for r in buy_pre:
                f = fundamental_employee(r["コード"])
                r["_fundamentals"] = f
                r["ファンダ点"] = float(f["score"])
                r["総合買い評価"] = float(r["買いスコア"] * 0.8 + f["score"] * 0.2)
                if f.get("available") and f["score"] < 40:
                    r["買い対象"] = False
                    r["トレンド判定"] = "逆張り条件は合格・ファンダ40点未満で除外"
        monitored = sorted([r for r in buy_pre if r.get("買い対象")],
                           key=lambda x: x["総合買い評価"], reverse=True)
        buy = monitored
        short_pre = sorted([r for r in ok if r.get("空売りトレンド適合")],
                           key=lambda x: x["空売りスコア"], reverse=True)[:20]
        if performed_scan:
            new_candidates = []
            for r in monitored:
                new_candidates.append({
                    "code": r["コード"], "name": r["銘柄名"], "price": r["現在値"],
                    "entry_rule_version": ENTRY_RULE_VERSION,
                    "pullback_zone": r["押し目条件"], "rebound_confirmed": r["反転確認"],
                    "bb_position": r["BB位置σ"],
                    "bb_lower_touched": r["BB下限到達"],
                    "score": r["総合買い評価"], "technical_score": r["買いスコア"],
                    "fundamental_score": r["ファンダ点"],
                    "fundamental_available": r["_fundamentals"].get("available", 0),
                    "fundamental_comment": r["_fundamentals"].get("comment"),
                    "rsi": r["RSI14"],
                    "ma5": r["MA5"], "ma25": r["MA25"],
                    "ma75": r["MA75"], "ma200": r["MA200"],
                    "ma5_cross_up": r["5日線上抜け"],
                    "ma5_distance_pct": r["5日線乖離%"],
                    "ma5_bar_time": r["5日線判定日"],
                    "ma75_touched": r["75日線接触"],
                    "ma75_distance_pct": r["75日線乖離%"],
                    "macd": r["MACD"], "macd_signal": r["MACDシグナル"],
                    "volume_ratio": r["出来高倍率"],
                    "cloud_position": r["一目位置"], "tenkan": r["転換線"],
                    "kijun": r["基準線"], "tenkan_cross_up": r["転換線上抜け"],
                    "buy_eligible": r["買い対象"], "trend_reason": r["トレンド判定"],
                    "psar": r["パラボリック"],
                    "psar_bull": r["パラボリック買い状態"],
                    "psar_buy_turn": r["パラボリック買い転換"],
                    "psar_bar_time": r["パラボリック判定日"],
                    "quarterly_score": r["四半期足強度"],
                    "quarterly_rsi": r["四半期RSI14"],
                    "quarterly_qualified": r["四半期足適合"],
                    "quarterly_reason": r["四半期足判定"],
                    "quarterly_pattern": r.get("四半期足形状", "対象外"),
                    "quarterly_high_level_consolidation": r.get("高値圏持ち合い", False),
                })
                if len(new_candidates) >= 10:
                    break
            merge_new_candidates(new_candidates)
        with st.spinner("市場環境を確認中…"):
            market_score, regime, market_note = market_employee_score()
        with st.spinner("循環セクターの本日候補を確認中…"):
            sector_rows, sector_date = sector_rotation_employee()
            rotation_candidates = rank_rotation_stocks(sector_rows, market_score)
            rotation_patterns = analyze_rotation_patterns(
                rotation_candidates,
                load_rotation_history(),
                market_score,
                str(sector_date)[:10] if sector_date is not None else None,
            )
        short = []
        with st.spinner("信用売り候補のファンダメンタルを確認中…"):
            for r in short_pre:
                f = fundamental_employee(r["コード"])
                r.update(assess_short(r, f, universe_by_code.get(str(r["コード"]), {}), market_score))
                short.append(r)
        short.sort(key=lambda x: x["空売り総合評価"], reverse=True)
        st.info(f"市場環境社員: {regime}  {market_score:.0f}/100　{market_note}")

        tab_buy, tab_rotation, tab_short, tab_meeting, tab_decline_bt = st.tabs(
            ["📈 買い候補", "🔁 循環セクター（参考）", "📉 空売り候補", "👥 AI社員会議", "下げ日数バックテスト"]
        )
        with tab_buy:
            st.subheader(f"下げ日数カウンター買い候補（{count_min}〜{count_max}本目）")
            if decline_start_mode == START_MODE_BB_UPPER:
                st.caption(
                    "ボリンジャーバンド(20日・2σ)の上限に高値が触れたあと、最初に前日比安となった日を1本目として数えます。"
                    f"{decline_price_label}が{decline_compare_bars}本前より高い日は0へリセットし、"
                    "次にバンド上限へ触れるまで数え直しません。"
                )
            else:
                st.caption(
                    f"{decline_price_label}が{decline_compare_bars}本前より高い日は0へリセットし、"
                    "その後に最初の前日比安となった日を1本目として数えます。"
                )
            all_count_rows = []
            for row in sorted(
                ok,
                key=lambda item: (-int(item.get("下げ日数") or 0), str(item.get("コード", ""))),
            ):
                all_count_rows.append({
                    "コード": row["コード"],
                    "銘柄名": row["銘柄名"],
                    "現在カウント": int(row.get("下げ日数") or 0),
                    "状態": row.get("下げカウント状態", "到達待ち"),
                    "判定日": row.get("下げ判定日"),
                    f"{decline_signal_count}本到達日": row.get("下げ買い到達日"),
                    f"{decline_add_count}本到達日": row.get("下げ追加到達日"),
                })

            reached_today = sum(bool(row["下げ買い本日到達"]) for row in decline_candidates)
            st.info(
                f"全{len(ok)}銘柄を判定し、{count_min}〜{count_max}本目の買い候補は"
                f"{len(decline_candidates)}銘柄です。本日{decline_signal_count}本へ到達した銘柄は"
                f"{reached_today}件です。"
            )
            if decline_candidates:
                count_rows = []
                for row in decline_candidates:
                    count_rows.append({
                        "コード": row["コード"],
                        "銘柄名": row["銘柄名"],
                        "現在カウント": int(row["下げ日数"]),
                        "状態": row["下げカウント状態"],
                        "判定日": row["下げ判定日"],
                        f"{decline_signal_count}本到達日": row["下げ買い到達日"],
                        f"{decline_add_count}本到達日": row["下げ追加到達日"],
                        "現在値": row["現在値"],
                        "RSI14": row["RSI14"],
                        "5日線上抜け": row["5日線上抜け"],
                        "パラボリック": row["パラボリック判定"],
                    })
                render_clickable_stock_table(
                    pd.DataFrame(count_rows),
                    key="decline_candidate_stocks",
                    height=min(72 + 35 * len(count_rows), 700),
                    column_config={
                        "現在値": st.column_config.NumberColumn(format="¥%,.1f"),
                        "RSI14": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
            else:
                st.info(f"現在、{count_min}〜{count_max}本目の銘柄はありません。")

            with st.expander(f"全{len(all_count_rows)}銘柄の現在カウントを見る"):
                st.caption("買い候補以外も含め、分析できた全銘柄を現在カウントの大きい順に表示します。")
                render_clickable_stock_table(
                    pd.DataFrame(all_count_rows),
                    key="all_decline_count_stocks",
                    height=650,
                )

            if decline_add_candidates:
                with st.expander(f"追加買いの目安（{decline_add_count}本以上・{len(decline_add_candidates)}件）"):
                    render_clickable_stock_table(
                        pd.DataFrame(decline_add_candidates)[
                            ["コード", "銘柄名", "下げ日数", "下げカウント状態", "下げ判定日", "下げ追加到達日", "現在値", "RSI14"]
                        ],
                        key="decline_add_candidate_stocks",
                    )

            progression_source = decline_candidates or sorted(
                ok,
                key=lambda row: int(row.get("下げ日数") or 0),
                reverse=True,
            )[:3]
            if progression_source:
                progression_by_label = {
                    f"{row['コード']} {row['銘柄名']}": row for row in progression_source[:20]
                }
                default_progression = list(progression_by_label)[: min(3, len(progression_by_label))]
                st.caption(
                    "次の3銘柄枠は、日ごとのカウント推移を比較するための選択欄です。"
                    "買い候補の件数を3銘柄に制限するものではありません。"
                )
                selected_progression = st.multiselect(
                    "推移を確認する銘柄（比較表示は最大3銘柄）",
                    list(progression_by_label),
                    default=default_progression,
                    max_selections=3,
                    key="decline_progression_symbols",
                )
                if selected_progression:
                    progression_frames = []
                    for label in selected_progression:
                        row = progression_by_label[label]
                        frame = pd.DataFrame(row.get("_下げカウント推移", [])).copy()
                        if not frame.empty:
                            frame.insert(0, "銘柄", label)
                            progression_frames.append(frame)
                    if progression_frames:
                        with st.expander("テスト確認用：直近40日足のカウント推移", expanded=True):
                            render_clickable_stock_table(
                                pd.concat(progression_frames, ignore_index=True),
                                key="decline_progression_stocks",
                                column_config={
                                    "終値": st.column_config.NumberColumn(format="¥%,.1f"),
                                    "判定価格": st.column_config.NumberColumn(format="¥%,.1f"),
                                },
                            )

            st.warning("カウント到達は統計上の候補表示です。自動注文は行いません。")
            st.markdown("---")
            st.subheader("ボリンジャーバンド下限到達銘柄")
            st.caption(
                "直近3営業日の安値が、その日の25日ボリンジャーバンド下限（-2σ）以下に"
                "到達した銘柄を表示します。反転・5日線上抜け・パラボリック成立前も含みます。"
            )
            st.info(
                f"全{len(ok)}銘柄のうち、直近3営業日にボリンジャーバンド下限へ"
                f"到達した銘柄は{len(bb_lower_candidates)}銘柄です。"
            )
            if bb_lower_candidates:
                bb_rows = pd.DataFrame(bb_lower_candidates)[[
                    "コード", "銘柄名", "BB下限到達日", "現在値", "BB下限",
                    "BB下限乖離%", "BB位置σ", "RSI14", "下げ日数",
                    "5日線上抜け", "パラボリック判定", "反転確認",
                ]]
                render_clickable_stock_table(
                    bb_rows,
                    key="bb_lower_candidate_stocks",
                    height=min(72 + 35 * len(bb_rows), 700),
                    column_config={
                        "現在値": st.column_config.NumberColumn(format="¥%,.1f"),
                        "BB下限": st.column_config.NumberColumn(format="¥%,.1f"),
                        "BB下限乖離%": st.column_config.NumberColumn(format="%+.2f%%"),
                        "BB位置σ": st.column_config.NumberColumn(format="%.2fσ"),
                        "RSI14": st.column_config.NumberColumn(format="%.1f"),
                    },
                )
            else:
                st.info("直近3営業日にボリンジャーバンド下限へ到達した銘柄はありません。")

            st.markdown("---")
            st.subheader("10月9日基準の押し目買い候補")
            st.success("75日線付近の上昇トレンド押し目と反転確認で選びます。5日線とパラボリックは参考表示で、必須条件にしません。")
            st.caption("総合買い評価は逆張り80%・ファンダ20%。RSI 28〜45、25日線以下、BB位置-0.5σ以下、75日線接触を判定します。")
            cols = ["コード", "銘柄名", "総合買い評価", "買いスコア", "ファンダ点",
                    "RSI14", "BB位置σ", "現在値", "75日線乖離%", "75日線接触",
                    "BB下限到達", "5日線上抜け", "5日線乖離%", "MA5",
                    "パラボリック判定", "反転確認", "一目位置",
                    "出来高倍率", "包み陽線", "下ヒゲ陽線"]
            if buy:
                buy_table = pd.DataFrame(buy)[cols].head(50)
                render_clickable_stock_table(
                    buy_table,
                    key="baseline_buy_candidate_stocks",
                    style_func=highlight_buy_score,
                )
                labels = [f"{r['コード']} {r['銘柄名']}" for r in buy[:20]]
                selected = st.selectbox(
                    "総合分析も見る",
                    ["選択してください", *labels],
                    key="buy_detail",
                )
                if selected != "選択してください":
                    code = selected.split()[0]
                    row = next(r for r in buy if r["コード"] == code)
                    with st.spinner(f"{code} の詳細分析中…"):
                        show_buy_detail(row, market_score)
            else:
                st.info("現在、10月9日基準の押し目条件を満たす銘柄はありません。")

        with tab_rotation:
            st.subheader("循環セクターの参考表示")
            st.caption("業種の資金の向きを確認する表です。この銘柄順位は買い候補やDiscord通知には使いません。")
            st.warning("朝の成行は自動発注しません。SBIの寄り付き気配が前日終値より3%以上高い場合は、飛びつかず見送り判定です。")
            if market_score <= 38:
                st.warning("地合いはリスクオフです。候補は表示しますが『監視優先』で、買いサインにはしません。")
            if rotation_patterns:
                early_count = sum(bool(row["初動加速"]) for row in rotation_patterns)
                st.info(f"循環初動参考：{early_count}銘柄（売買判定には不使用）")
                rotation_df = pd.DataFrame(rotation_patterns)
                # A running Streamlit process may briefly retain rows made
                # before the early-rotation and MA5 columns were introduced.
                cached_defaults = {
                    "業種判定": "循環買い",
                    "初動理由": "条件待ち",
                    "初動段階": "条件待ち",
                    "過熱判定": "未過熱",
                    "5日線上抜け": False,
                    "5日線乖離%": np.nan,
                }
                for column, default in cached_defaults.items():
                    if column not in rotation_df.columns:
                        rotation_df[column] = default
                rotation_df.insert(0, "順位", range(1, len(rotation_df) + 1))
                render_clickable_stock_table(
                    rotation_df[["順位", "初動段階", "業種・テーマ", "業種判定", "コード", "銘柄名", "初動理由", "翌朝の扱い", "過熱判定", "5日線上抜け", "5日線乖離%", "過去5回登場", "前回順位", "順位変化", "5日加速pt", "循環点", "現在値", "1日%", "5日%", "20日%"]],
                    key="rotation_candidate_stocks",
                    column_config={
                        "循環点": st.column_config.NumberColumn(format="%.1f"),
                        "現在値": st.column_config.NumberColumn(format="¥%,.1f"),
                        "1日%": st.column_config.NumberColumn(format="%+.2f%%"),
                        "5日%": st.column_config.NumberColumn(format="%+.2f%%"),
                        "20日%": st.column_config.NumberColumn(format="%+.2f%%"),
                        "5日加速pt": st.column_config.NumberColumn(format="%+.2f"),
                        "5日線乖離%": st.column_config.NumberColumn(format="%+.2f%%"),
                    },
                )
                st.page_link("pages/3_市場環境社員.py", label="業種全体とチャートを確認")
            else:
                st.info("本日は、表示できる循環セクター参考銘柄はありません。")

        with tab_short:
            st.subheader("信用売り総合ランキング")
            st.caption("下降トレンド・割高/業績・信用倍率・市場環境を総合評価します。信用情報がない銘柄は実行候補にしません。")
            if short:
                cols = ["コード", "銘柄名", "空売り総合評価", "空売り状態", "予想PER", "信用倍率",
                        "貸借確認", "空売りスコア", "空売りファンダ", "RSI14", "一目位置",
                        "転換線下抜け", "包み陰線", "出来高倍率", "空売り理由"]
                render_clickable_stock_table(
                    pd.DataFrame(short)[cols].head(20),
                    key="short_candidate_stocks",
                )
            else:
                st.info("現在、日足の下降トレンド条件を満たす信用売り候補はありません。")
            st.warning("発注前にSBIで売建可能数・貸株料・逆日歩・決算日を必ず確認してください。Stocknoteは自動発注しません。")

        with tab_meeting:
            st.write("上位候補をテクニカル・ファンダメンタル・市場環境の3社員で再評価します。")
            if st.button("👥 上位5銘柄をAI社員会議で再評価", use_container_width=True):
                st.markdown("#### 買い会議")
                render_clickable_stock_table(
                    meeting_rows(buy, "buy", market_score),
                    key="meeting_buy_stocks",
                )
                st.markdown("#### 空売り会議")
                if short:
                    render_clickable_stock_table(
                        pd.DataFrame(short)[["コード", "銘柄名", "空売り総合評価", "空売り状態", "予想PER", "信用倍率", "空売り理由"]].head(5),
                        key="meeting_short_stocks",
                    )
                else:
                    st.info("信用売り会議の対象はありません。")

        with tab_decline_bt:
            st.subheader("下げ日数カウンターのバックテスト")
            st.caption(
                "各シグナル日の終値で仮想購入し、5営業日以内に一度でも終値が上回った割合と、"
                "5営業日目の勝率・平均損益・期間中の最大ドローダウンを比較します。"
                f"（数え始めの起点：{decline_start_label}）"
            )
            signal_summary = summarize_signal_events(
                decline_backtest_events,
                thresholds=(8, 10, 12, 15),
            )
            st.dataframe(
                signal_summary,
                hide_index=True,
                width="stretch",
                column_config={
                    "5日以内上昇率%": st.column_config.NumberColumn(format="%.1f%%"),
                    "勝率%": st.column_config.NumberColumn(format="%.1f%%"),
                    "平均損益%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "最大ドローダウン%": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )

            st.markdown("#### 買いのみ・追加買いありの比較")
            strategy_summary = pd.DataFrame([
                summarize_strategy_trades(
                    decline_single_trades,
                    f"{decline_signal_count}本目で1回買い",
                ),
                summarize_strategy_trades(
                    decline_added_trades,
                    f"{decline_signal_count}本目＋{decline_add_count}本目で同額追加",
                ),
            ])
            st.dataframe(
                strategy_summary,
                hide_index=True,
                width="stretch",
                column_config={
                    "勝率%": st.column_config.NumberColumn(format="%.1f%%"),
                    "平均損益%": st.column_config.NumberColumn(format="%+.2f%%"),
                    "最大ドローダウン%": st.column_config.NumberColumn(format="%+.2f%%"),
                },
            )
            st.caption(
                "追加買い比較は同じ株数を追加し、カウントが0へリセットした日の終値で手仕舞う仮定です。"
                "手数料・税金・スリッページは含みません。"
            )
            if decline_backtest_events:
                with st.expander("バックテスト対象シグナルを確認"):
                    event_table = pd.DataFrame(decline_backtest_events).rename(columns={
                        "threshold": "本数",
                        "signal_date": "シグナル日",
                        "entry_price": "買値",
                        "within_horizon_up": "5日以内上昇",
                        "final_return_pct": "5日損益%",
                        "max_drawdown_pct": "最大DD%",
                    })
                    render_clickable_stock_table(
                        event_table[["コード", "銘柄名", "本数", "シグナル日", "買値", "5日以内上昇", "5日損益%", "最大DD%"]],
                        key="backtest_signal_stocks",
                    )

    if bad:
        with st.expander(f"⚠️ 分析できなかった銘柄（{len(bad)}件）"):
            for r in bad[:100]:
                st.caption(f"{r.get('コード', '')} {r.get('銘柄名', '')}: {r.get('error', '不明')}")

st.markdown("---")
st.caption(f"更新: {datetime.now().strftime('%Y/%m/%d %H:%M:%S')} | 保存CSVは差し替えるまで継続利用 | ファンダメンタルはYahoo Finance＋財務諸表から補完計算")
