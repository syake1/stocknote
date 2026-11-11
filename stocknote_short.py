"""Conservative short-selling assessment using only available real data."""
import math
import re

import numpy as np
import pandas as pd

from stocknote_technicals import daily_trend_context, rsi14


def number(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("倍", "").replace("%", "")
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        result = float(match.group())
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def first_value(row, aliases):
    for key in aliases:
        if key in row:
            value = row.get(key)
            if value is not None and str(value).strip().lower() not in {"", "nan", "none", "—", "-"}:
                return value
    return None


def loanable_value(row):
    raw = first_value(row, ["貸借銘柄", "貸借区分", "信用区分", "売建可否", "空売り可否"])
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if any(x in text for x in ("貸借", "売建可", "空売り可", "○", "可")) and not any(x in text for x in ("不可", "×")):
        return True
    if any(x in text for x in ("不可", "×", "信用銘柄")):
        return False
    return None


def short_fundamental_score(f, forecast_per=None):
    """High means weaker/more expensive fundamentals; never invent missing values."""
    score = 50.0
    per = number(forecast_per)
    if per is None:
        per = number(f.get("per"))
    pbr, roe = number(f.get("pbr")), number(f.get("roe"))
    opm, growth = number(f.get("opm")), number(f.get("growth"))
    reasons = []
    if per is not None:
        if per >= 35: score += 18; reasons.append("PER35倍以上")
        elif per >= 25: score += 10; reasons.append("PER25倍以上")
        elif 0 < per <= 15: score -= 12; reasons.append("PER割安")
    if pbr is not None:
        if pbr >= 4: score += 10; reasons.append("PBR高い")
        elif 0 < pbr <= 1.2: score -= 7; reasons.append("PBR低い")
    if roe is not None:
        if roe < 0: score += 14; reasons.append("ROEマイナス")
        elif roe >= .10: score -= 10; reasons.append("ROE良好")
    if opm is not None:
        if opm < 0: score += 15; reasons.append("営業赤字")
        elif opm >= .10: score -= 8; reasons.append("営業利益率良好")
    if growth is not None:
        if growth < 0: score += 12; reasons.append("減収")
        elif growth >= .05: score -= 8; reasons.append("増収")
    return max(0.0, min(100.0, score)), reasons, per


def credit_score(ratio):
    ratio = number(ratio)
    if ratio is None:
        return None
    if ratio < 1:
        return 15.0  # short crowding / squeeze and reverse-stock-fee caution
    if ratio <= 3:
        return 45.0
    if ratio <= 10:
        return 80.0
    return 65.0


def short_watch_reason(technical):
    """Return an early-watch reason before the full long-term downtrend is complete."""
    price = number(technical.get("現在値"))
    ma5 = number(technical.get("MA5"))
    ma25 = number(technical.get("MA25"))
    ma75 = number(technical.get("MA75"))
    rsi = number(technical.get("RSI14"))
    score = number(technical.get("空売りスコア")) or 0.0
    if price is None or ma5 is None or ma25 is None:
        return None
    reasons = []
    if technical.get("空売りトレンド適合"):
        reasons.append("長期下降トレンド")
    if price < ma5 and price < ma25:
        reasons.append("5日・25日線の下")
    elif price < ma5:
        reasons.append("5日線割れ")
    if technical.get("転換線下抜け"):
        reasons.append("転換線下抜け")
    if technical.get("包み陰線") or technical.get("上ヒゲ陰線"):
        reasons.append("弱いローソク足")
    if ma75 is not None and price < ma75:
        reasons.append("75日線の下")
    if rsi is not None and rsi >= 60 and price < ma5:
        reasons.append("高RSIから失速")
    if score >= 45:
        reasons.append("空売り点45以上")

    early_trigger = bool(
        technical.get("空売りトレンド適合")
        or technical.get("転換線下抜け")
        or technical.get("包み陰線")
        or technical.get("上ヒゲ陰線")
        or (price < ma5 and price < ma25)
        or (rsi is not None and rsi >= 60 and price < ma5)
    )
    return "・".join(dict.fromkeys(reasons)) if early_trigger and reasons else None


def short_technical_snapshot(code, name, hist):
    """Build the shared daily technical snapshot used for short watchlists."""
    if hist is None or hist.empty:
        return None
    frame = hist.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if not {"Open", "High", "Low", "Close"}.issubset(frame.columns):
        return None
    frame = frame.dropna(subset=["High", "Low", "Close"])
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 205:
        return None
    frame = frame.loc[close.index]
    open_ = pd.to_numeric(frame["Open"], errors="coerce")
    high = pd.to_numeric(frame["High"], errors="coerce")
    ma5 = close.rolling(5).mean()
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    ma200 = close.rolling(200).mean()
    std25 = close.rolling(25).std()
    upper = ma25 + 2 * std25
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    trend = daily_trend_context(frame)
    values = (
        close.iloc[-1], ma5.iloc[-1], ma25.iloc[-1], ma75.iloc[-1],
        ma200.iloc[-1], upper.iloc[-1], macd.iloc[-1], signal.iloc[-1],
    )
    if any(pd.isna(value) for value in values):
        return None

    px = float(close.iloc[-1])
    m5, m25, m75, m200 = (float(series.iloc[-1]) for series in (ma5, ma25, ma75, ma200))
    bup = float(upper.iloc[-1])
    rv = float(rsi14(close).iloc[-1])
    md, sg = float(macd.iloc[-1]), float(signal.iloc[-1])
    volume_ratio = 1.0
    if "Volume" in frame:
        volume = pd.to_numeric(frame["Volume"], errors="coerce")
        base = volume.iloc[-21:-1].mean() if len(volume) >= 21 else np.nan
        if pd.notna(volume.iloc[-1]) and pd.notna(base) and float(base) > 0:
            volume_ratio = float(volume.iloc[-1] / base)

    bearish_engulfing = bool(
        close.iloc[-2] > open_.iloc[-2]
        and close.iloc[-1] < open_.iloc[-1]
        and open_.iloc[-1] >= close.iloc[-2]
        and close.iloc[-1] <= open_.iloc[-2]
    )
    body = abs(px - float(open_.iloc[-1]))
    upper_wick = float(high.iloc[-1]) - max(px, float(open_.iloc[-1]))
    upper_wick_bear = bool(
        px < float(open_.iloc[-1])
        and upper_wick >= max(body, px * 0.005)
    )

    short_rsi = float(np.clip((rv - 55) / 25 * 35, 0, 35))
    short_bb = (
        25.0 if px >= bup * 0.98
        else float(np.clip((px - m25) / max(bup - m25, 1e-9) * 20, 0, 20))
    )
    short_trend = (12.0 if m25 <= m75 else 4.0) + (5.0 if px < m200 else 0.0)
    short_macd = 8.0 if md < sg else 2.0
    short_volume = float(np.clip((volume_ratio - 0.8) * 6, 0, 8))
    short_candle = 12.0 if bearish_engulfing else 8.0 if upper_wick_bear else 0.0
    score = float(np.clip(
        short_rsi + short_bb + short_trend + short_macd + short_volume + short_candle,
        0,
        100,
    ))
    trend_ok = bool(trend and trend.get("short_eligible"))
    if not trend_ok:
        score = min(score, 44.0)
    elif trend.get("tenkan_cross_down"):
        score = min(100.0, score + 8.0)
    return {
        "コード": str(code),
        "銘柄名": name or str(code),
        "現在値": px,
        "RSI14": rv,
        "MA5": m5,
        "MA25": m25,
        "MA75": m75,
        "MA200": m200,
        "出来高倍率": volume_ratio,
        "包み陰線": bearish_engulfing,
        "上ヒゲ陰線": upper_wick_bear,
        "転換線下抜け": bool(trend and trend.get("tenkan_cross_down")),
        "一目位置": trend.get("cloud_position", "データ不足") if trend else "データ不足",
        "空売りスコア": score,
        "空売りトレンド適合": trend_ok,
    }


def assess_short(technical, fundamental, universe_row, market_score):
    per_csv = first_value(universe_row, ["PER(株価収益率)(予)(倍)", "予想PER", "PER", "PER(倍)"])
    ratio_raw = first_value(universe_row, ["信用倍率", "信用倍率(倍)", "信用取引倍率", "貸借倍率"])
    ratio = number(ratio_raw)
    loanable = loanable_value(universe_row)
    fscore, reasons, per = short_fundamental_score(fundamental, per_csv)
    tscore = float(technical.get("空売りスコア", 0))
    cscore = credit_score(ratio)
    bearish_market = 100.0 - float(market_score)
    # Unknown credit data is not treated as neutral: it contributes zero and
    # prevents an actionable label until the broker data is confirmed.
    total = tscore * .45 + fscore * .25 + (cscore or 0.0) * .15 + bearish_market * .15
    trend_ok = bool(technical.get("空売りトレンド適合"))
    confirmed = ratio is not None and loanable is True
    if not trend_ok:
        status = "トレンド条件外"
    elif not confirmed:
        status = "SBI信用データ確認待ち"
    elif ratio < 1:
        status = "踏み上げ・逆日歩注意"
    elif total >= 75:
        status = "空売り条件到達"
    elif total >= 65:
        status = "空売り条件接近"
    else:
        status = "見送り"
    return {
        "空売り総合評価": round(total, 1), "空売りファンダ": round(fscore, 1),
        "信用倍率": ratio, "貸借確認": loanable, "予想PER": per,
        "空売り状態": status, "空売り理由": "・".join(reasons) if reasons else "大きな悪化材料なし",
    }
