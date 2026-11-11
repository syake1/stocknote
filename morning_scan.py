"""Headless Stocknote morning scan for GitHub Actions.

Reads data/saved_universe.csv, evaluates daily technical conditions, and posts
ranked buy candidates to Discord when DISCORD_WEBHOOK is configured.
"""
import os
import re
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from monitor_candidates import notify
from stocknote_fundamentals import get_fundamentals
from stocknote_holding_analysis import market_score as current_market_score
from stocknote_short import assess_short, short_technical_snapshot, short_watch_reason
from stocknote_tracking import filter_new_notifications, merge_new_candidates
from stocknote_technicals import baseline_pullback_zone, daily_trend_context
from stocknote_sector_rotation import download_sector_rotation, rank_rotation_stocks
from stocknote_rotation_history import analyze_rotation_patterns, save_rotation_snapshot

UNIVERSE = os.getenv("STOCKNOTE_UNIVERSE", "data/saved_universe.csv")
WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
SEND_MEETING = os.getenv("STOCKNOTE_SEND_MEETING", "0") == "1"
MEETING_LABEL = os.getenv("STOCKNOTE_MEETING_LABEL", "朝の買い候補").strip() or "朝の買い候補"
TOP_N = int(os.getenv("STOCKNOTE_TOP_N", "10"))
MIN_SCORE = float(os.getenv("STOCKNOTE_MIN_BUY_SCORE", "0"))
ENTRY_RULE_VERSION = 5
MIN_FUNDAMENTAL_SCORE = float(os.getenv("STOCKNOTE_MIN_FUNDAMENTAL_SCORE", "40"))
FUNDAMENTAL_POOL = int(os.getenv("STOCKNOTE_FUNDAMENTAL_POOL", "20"))
SHORT_TOP_N = int(os.getenv("STOCKNOTE_SHORT_TOP_N", "5"))
SHORT_POOL = int(os.getenv("STOCKNOTE_SHORT_POOL", "12"))


def normalize_code(value):
    text = str(value).strip().upper().replace(".T", "")
    text = re.sub(r"\.0$", "", text)
    m = re.search(r"([0-9A-Z]{4})", text)
    return m.group(1) if m else ""


def rsi14(close):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def touched_ma75_recently(low, ma75, lookback=3):
    """Return True when price reached or broke the 75-day line recently."""
    aligned = pd.concat([low.rename("low"), ma75.rename("ma75")], axis=1).dropna().tail(lookback)
    return bool(not aligned.empty and (aligned["low"] <= aligned["ma75"]).any())


def touched_bb_lower_recently(low, lower, lookback=3):
    """Return True only when the daily low reached the lower Bollinger band."""
    aligned = pd.concat([low.rename("low"), lower.rename("lower")], axis=1).dropna().tail(lookback)
    return bool(not aligned.empty and (aligned["low"] <= aligned["lower"]).any())


def add_fundamental_judgement(row, fundamentals):
    """Keep contrarian timing primary, then use fundamentals as quality control."""
    result = dict(row)
    fundamental_score = float(fundamentals.get("score", 50.0))
    available = int(fundamentals.get("available", 0) or 0)
    if available and fundamental_score < MIN_FUNDAMENTAL_SCORE:
        return None
    technical_score = float(result["score"])
    result.update({
        "technical_score": technical_score,
        "fundamental_score": fundamental_score,
        "fundamental_comment": fundamentals.get("comment", "ファンダメンタル取得不可"),
        "fundamental_available": available,
        # The original Bollinger-band contrarian signal remains the main axis.
        "score": technical_score * 0.8 + fundamental_score * 0.2,
    })
    return result


def download_daily_history(code):
    return yf.download(
        f"{code}.T", period="14mo", interval="1d", auto_adjust=False,
        progress=False, threads=False,
    )


def score_one(code, name, hist=None):
    h = hist if hist is not None else download_daily_history(code)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    if h.empty or "Close" not in h or len(h) < 205:
        return None
    close = pd.to_numeric(h["Close"], errors="coerce").dropna()
    if len(close) < 205:
        return None
    rsi = rsi14(close)
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    ma200 = close.rolling(200).mean()
    trend = daily_trend_context(h)
    if not trend or not trend["buy_eligible"]:
        return None
    # Daily trend filter: compare with five trading days ago.
    ma75_now, ma75_prev = ma75.iloc[-1], ma75.iloc[-6]
    ma200_now, ma200_prev = ma200.iloc[-1], ma200.iloc[-6]
    if any(pd.isna(x) for x in (ma75_now, ma75_prev, ma200_now, ma200_prev)):
        return None
    if float(ma75_now) < float(ma75_prev) or float(ma200_now) < float(ma200_prev):
        return None
    std25 = close.rolling(25).std()
    lower = ma25 - 2 * std25
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    px, rv = float(close.iloc[-1]), float(rsi.iloc[-1])
    prev_px, prev_rsi = float(close.iloc[-2]), float(rsi.iloc[-2])
    m25, m75 = float(ma25.iloc[-1]), float(ma75.iloc[-1])
    m200 = float(ma200.iloc[-1])
    blo = float(lower.iloc[-1])
    std_now = float(std25.iloc[-1])
    bb_position = (px - m25) / max(std_now, 1e-9)
    md, sg = float(macd.iloc[-1]), float(signal.iloc[-1])
    vr = 1.0
    if "Volume" in h:
        v = pd.to_numeric(h["Volume"], errors="coerce").dropna()
        if len(v) >= 21 and float(v.iloc[-21:-1].mean()) > 0:
            vr = float(v.iloc[-1] / v.iloc[-21:-1].mean())
    reversal = False
    lower_wick_reversal = False
    low = pd.to_numeric(h["Low"], errors="coerce") if "Low" in h else None
    if "Open" in h and len(h) >= 2:
        o = pd.to_numeric(h["Open"], errors="coerce")
        if pd.notna(o.iloc[-1]) and pd.notna(o.iloc[-2]):
            reversal = bool(close.iloc[-2] < o.iloc[-2] and close.iloc[-1] > o.iloc[-1]
                            and close.iloc[-1] >= o.iloc[-2] and o.iloc[-1] <= close.iloc[-2])
        if low is not None and pd.notna(low.iloc[-1]) and pd.notna(o.iloc[-1]):
            body_bottom = min(float(o.iloc[-1]), px)
            body = abs(px - float(o.iloc[-1]))
            lower_wick = body_bottom - float(low.iloc[-1])
            lower_wick_reversal = bool(px >= float(o.iloc[-1]) and lower_wick >= max(body, px * 0.005))

    # ボリンジャーバンド逆張りを主軸にする。75日線まで下げていない
    # 高値圏銘柄は、RSIや25日線だけが条件を満たしても候補にしない。
    ma75_touched = bool(low is not None and touched_ma75_recently(low, ma75, lookback=3))
    bb_lower_touched = bool(low is not None and touched_bb_lower_recently(low, lower, lookback=3))
    ma75_distance_pct = (px / m75 - 1.0) * 100.0
    pullback_zone = baseline_pullback_zone(
        rv,
        px,
        m25,
        bb_position,
        ma75_touched,
        m75,
    )
    rebound_confirmed = bool(
        reversal or lower_wick_reversal
        or (px > prev_px and rv > prev_rsi)
    )
    if not pullback_zone or not rebound_confirmed:
        return None

    buy_rsi = float(np.clip((45-rv)/17*35, 0, 35))
    buy_bb = 25.0 if px <= blo*1.02 else float(np.clip((m25-px)/max(m25-blo,1e-9)*20,0,20))
    buy_trend = (12.0 if m25 >= m75 else 4.0) + (5.0 if px >= m200 else 0.0)
    buy_macd = 8.0 if md >= sg else 2.0
    buy_volume = float(np.clip((vr-0.8)*6,0,8))
    buy_candle = 12.0 if reversal else 0.0
    ichimoku = 8.0 if trend["tenkan_cross_up"] else 5.0 if trend["tenkan_above_kijun"] else 0.0
    score = float(np.clip(buy_rsi+buy_bb+buy_trend+buy_macd+buy_volume+buy_candle+ichimoku,0,100))
    return {"code":code,"name":name or code,"price":px,"rsi":rv,"vr":vr,"score":score,
            "reversal":reversal,"lower_wick_reversal":lower_wick_reversal,
            "pullback_zone":pullback_zone,"rebound_confirmed":rebound_confirmed,
            "bb_position":bb_position,"ma75_touched":ma75_touched,
            "bb_lower_touched":bb_lower_touched,
            "ma75_distance_pct":ma75_distance_pct,
            "ma75_slope":"flat_or_up","ma200_slope":"flat_or_up", **trend}


def build_discord_message(rows, total, meeting_label=MEETING_LABEL, now=None):
    """Build a distinct morning/night message, including zero-result reports."""
    now = now or datetime.now().strftime("%Y/%m/%d %H:%M")
    heading = f"📊 **Stocknote {meeting_label}**  {now}"
    if rows:
        lines = [heading, f"母集団 {total}銘柄 / 上位 {len(rows)}銘柄"]
        for i, r in enumerate(rows, 1):
            if r["reversal"]:
                candle = " / 包み陽線"
            elif r.get("lower_wick_reversal"):
                candle = " / 下ヒゲ陽線"
            else:
                candle = " / 終値・RSI反転"
            ichi = " / 転換線↑基準線" if r.get("tenkan_above_kijun") else " / 転換線≤基準線"
            lines.append(
                f"{i}. **{r['code']} {r['name']}**  総合 {r['score']:.1f}"
                f" / 逆張り {r.get('technical_score', r['score']):.1f}"
                f" / ファンダ {r.get('fundamental_score', 50):.1f}"
                f" / RSI {r['rsi']:.1f} / BB {r['bb_position']:.2f}σ"
                f" / 75日線乖離 {r.get('ma75_distance_pct', 0):+.1f}%"
                f" / ¥{r['price']:,.0f} / {r.get('cloud_position','—')}{ichi}"
                f" / 出来高 {r['vr']:.2f}倍{candle}"
                f" / {r.get('fundamental_comment', 'ファンダ未取得')}"
            )
    else:
        lines = [heading, f"母集団 {total}銘柄を確認しましたが、分析可能な買い候補はありませんでした。"]
    return "\n".join(lines)


def post_discord(rows, total):
    text = build_discord_message(rows, total)
    print(text)
    if not WEBHOOK:
        print("DISCORD_WEBHOOK is not configured; skipping Discord post.")
        return
    resp = requests.post(WEBHOOK, json={"content": text[:1950]}, timeout=20)
    resp.raise_for_status()


def post_discord_text(text):
    print(text)
    if not WEBHOOK:
        print("DISCORD_WEBHOOK is not configured; skipping Discord post.")
        return
    resp = requests.post(WEBHOOK, json={"content": text[:1950]}, timeout=20)
    resp.raise_for_status()


def build_rotation_discord_message(sector_rows, candidates, now=None):
    """Always show sector rotation, even when no sector has reached buy status."""
    now = now or datetime.now().strftime("%Y/%m/%d %H:%M")
    actionable = [
        row for row in sector_rows or []
        if row.get("判定") in {"循環初動", "循環買い"}
    ]
    displayed = (actionable or list(sector_rows or []))[:3]
    lines = [f"🔁 **Stocknote 本日の循環セクター**  {now}"]
    if not displayed:
        lines.append("セクターデータを取得できませんでした。次回の定時処理で再確認します。")
        return "\n".join(lines)
    if not actionable:
        lines.append("循環初動・循環買いは0業種です。資金流入点の上位を監視表示します。")
    for rank, row in enumerate(displayed, 1):
        lines.append(
            f"{rank}. **{row['業種・テーマ']}**  {row.get('判定', '様子見')}"
            f" / 1日 {float(row.get('1日%', 0)):+.2f}%"
            f" / 5日 {float(row.get('5日%', 0)):+.2f}%"
            f" / 初動 {float(row.get('初動点', 0)):.0f}点"
        )
    if candidates:
        lines.append("**セクター内の買い検討・監視銘柄**")
        for rank, row in enumerate(candidates[:5], 1):
            lines.append(
                f"{rank}. **{row['コード']} {row['銘柄名']}**"
                f" / {row['業種・テーマ']} / {row.get('扱い', '監視')}"
                f" / 1日 {float(row.get('1日%', 0)):+.2f}%"
            )
    else:
        lines.append("セクター内の個別銘柄は5日線条件待ちです。")
    lines.append("上昇後のセクターは追わず、初動と5日線付近を優先します。")
    return "\n".join(lines)


def build_short_discord_message(rows, total, now=None):
    """Always report the top technical short watches and their execution status."""
    now = now or datetime.now().strftime("%Y/%m/%d %H:%M")
    lines = [f"📉 **Stocknote 信用売り候補・監視**  {now}"]
    if not rows:
        lines.append(f"母集団 {total}銘柄 / 本日は5日・25日線割れ等の監視条件に該当なし")
        return "\n".join(lines)
    lines.append(f"母集団 {total}銘柄 / 上位 {min(len(rows), SHORT_TOP_N)}銘柄")
    for rank, row in enumerate(rows[:SHORT_TOP_N], 1):
        lines.append(
            f"{rank}. **{row['コード']} {row['銘柄名']}**"
            f" / {row.get('空売り区分', '監視')} / 総合 {float(row.get('空売り総合評価', 0)):.1f}"
            f" / RSI {float(row.get('RSI14', 0)):.1f}"
            f" / 出来高 {float(row.get('出来高倍率', 1)):.1f}倍"
            f" / {row.get('空売り監視理由', '下落条件を監視')}"
        )
    if any(row.get("貸借確認") is not True for row in rows[:SHORT_TOP_N]):
        lines.append("SBIの売建可否を未確認の銘柄は監視のみです。貸株料・逆日歩・決算日も確認してください。")
    lines.append("信用売りの自動発注は行いません。")
    return "\n".join(lines)


def update_rotation_ranking(market_score=50.0):
    """Save sector history and return the rows used by the daily report."""
    sector_rows, sector_date = download_sector_rotation()
    candidates = rank_rotation_stocks(sector_rows, market_score=market_score)
    asof = str(sector_date)[:10] if sector_date is not None else datetime.now().strftime("%Y-%m-%d")
    history = save_rotation_snapshot(candidates, asof)
    patterns = analyze_rotation_patterns(candidates, history, market_score=market_score, asof=asof)
    early = [row for row in patterns if row["初動加速"]]
    print(
        f"rotation reference: {len(candidates)} leaders / {len(early)} early accelerations; "
        "included in the daily sector report"
    )
    return sector_rows, patterns


def main():
    if not os.path.exists(UNIVERSE):
        raise SystemExit(f"Saved universe not found: {UNIVERSE}. Register CSV in Stocknote first.")
    df = pd.read_csv(UNIVERSE, dtype=str, encoding="utf-8-sig")
    code_col = "コード" if "コード" in df.columns else df.columns[0]
    name_col = "銘柄名" if "銘柄名" in df.columns else None
    items=[]
    for _, row in df.iterrows():
        code=normalize_code(row.get(code_col,""))
        if code:
            items.append((code, str(row.get(name_col, code)) if name_col else code))
    universe_by_code = {}
    for _, row in df.iterrows():
        code = normalize_code(row.get(code_col, ""))
        if code:
            universe_by_code[code] = row.to_dict()
    results=[]
    short_technical = []
    for i,(code,name) in enumerate(items,1):
        try:
            hist = download_daily_history(code)
            r=score_one(code,name,hist=hist)
            if r: results.append(r)
            short_row = short_technical_snapshot(code, name, hist)
            if short_row:
                reason = short_watch_reason(short_row)
                if reason:
                    short_row["空売り監視理由"] = reason
                    short_technical.append(short_row)
        except Exception as exc:
            print(f"WARN {code}: {exc}", file=sys.stderr)
        if i % 25 == 0: print(f"scanned {i}/{len(items)}")
    # First select genuine Bollinger-band/75-day-line pullbacks. Fundamentals
    # are deliberately evaluated afterwards, so valuation never promotes a
    # high-priced stock that has not actually pulled back.
    results.sort(key=lambda x:x["score"], reverse=True)
    pool_size = max(TOP_N, FUNDAMENTAL_POOL)
    judged = []
    for row in results[:pool_size]:
        try:
            fundamentals = get_fundamentals(row["code"])
        except Exception as exc:
            print(f"WARN fundamentals {row['code']}: {exc}", file=sys.stderr)
            fundamentals = {"score": 50.0, "available": 0, "comment": "ファンダメンタル取得不可"}
        enriched = add_fundamental_judgement(row, fundamentals)
        if enriched:
            judged.append(enriched)
    judged.sort(key=lambda x:x["score"], reverse=True)
    candidates = [row for row in judged if row["score"] >= MIN_SCORE][:TOP_N]
    try:
        market_score = current_market_score()
    except Exception as exc:
        print(f"WARN market score: {exc}", file=sys.stderr)
        market_score = 50.0
    short_technical.sort(
        key=lambda row: (
            bool(row.get("空売りトレンド適合")),
            float(row.get("空売りスコア") or 0.0),
        ),
        reverse=True,
    )
    short_rows = []
    for row in short_technical[:SHORT_POOL]:
        try:
            fundamentals = get_fundamentals(row["コード"])
        except Exception as exc:
            print(f"WARN short fundamentals {row['コード']}: {exc}", file=sys.stderr)
            fundamentals = {}
        row.update(assess_short(
            row,
            fundamentals,
            universe_by_code.get(row["コード"], {}),
            market_score,
        ))
        if row.get("空売りトレンド適合") and row["空売り状態"] in {"空売り条件到達", "空売り条件接近"}:
            row["空売り区分"] = "実行候補"
        elif row.get("空売りトレンド適合"):
            row["空売り区分"] = "本命監視"
        else:
            row["空売り区分"] = "早期監視"
        short_rows.append(row)
    short_rows.sort(
        key=lambda row: (
            row.get("空売り区分") == "実行候補",
            row.get("空売り区分") == "本命監視",
            float(row.get("空売り総合評価") or 0.0),
        ),
        reverse=True,
    )
    try:
        sector_rows, rotation_patterns = update_rotation_ranking(market_score)
    except Exception as exc:
        print(f"WARN rotation ranking: {exc}", file=sys.stderr)
        sector_rows, rotation_patterns = [], []
    if SEND_MEETING:
        post_discord(candidates, len(items))
        post_discord_text(build_rotation_discord_message(sector_rows, rotation_patterns))
        post_discord_text(build_short_discord_message(short_rows, len(items)))
    # This is an upsert, never a replacement: a zero-result scan leaves the
    # active 14-day monitoring list untouched.
    events = merge_new_candidates(candidates)
    notify(filter_new_notifications(events))
    print(
        f"new scan: {len(candidates)} buy candidates / {len(short_rows)} short watches "
        f"/ {len(rotation_patterns)} sector leaders from {len(items)} stocks"
    )


if __name__ == "__main__":
    main()
