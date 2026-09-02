"""Technical calculations shared by the scheduled Stocknote jobs."""
import numpy as np
import pandas as pd
import yfinance as yf


def rsi14(close):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def parabolic_sar(high, low, step=0.02, maximum=0.2):
    if len(high) < 2:
        return pd.Series(index=high.index, dtype=float)
    sar = pd.Series(index=high.index, dtype=float)
    bull, af, ep = True, step, float(high.iloc[0])
    sar.iloc[0] = float(low.iloc[0])
    for i in range(1, len(high)):
        value = float(sar.iloc[i - 1]) + af * (ep - float(sar.iloc[i - 1]))
        if bull:
            value = min(value, float(low.iloc[i - 1]), float(low.iloc[max(0, i - 2)]))
            if float(low.iloc[i]) < value:
                bull, value, ep, af = False, ep, float(low.iloc[i]), step
            elif float(high.iloc[i]) > ep:
                ep, af = float(high.iloc[i]), min(maximum, af + step)
        else:
            value = max(value, float(high.iloc[i - 1]), float(high.iloc[max(0, i - 2)]))
            if float(high.iloc[i]) > value:
                bull, value, ep, af = True, ep, float(high.iloc[i]), step
            elif float(low.iloc[i]) < ep:
                ep, af = float(low.iloc[i]), min(maximum, af + step)
        sar.iloc[i] = value
    return sar


def daily_trend_context(hist):
    """Return the daily uptrend/pullback gate used by every buy decision."""
    if hist is None or hist.empty:
        return None
    frame = hist.copy()
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if not {"Close", "High", "Low"}.issubset(frame.columns):
        return None
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 205:
        return None
    high = pd.to_numeric(frame.loc[close.index, "High"], errors="coerce")
    low = pd.to_numeric(frame.loc[close.index, "Low"], errors="coerce")
    ma75, ma200 = close.rolling(75).mean(), close.rolling(200).mean()
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    span_a = ((tenkan + kijun) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    values = (ma75.iloc[-1], ma75.iloc[-6], ma200.iloc[-1], ma200.iloc[-6],
              tenkan.iloc[-1], kijun.iloc[-1], span_a.iloc[-1], span_b.iloc[-1])
    if any(pd.isna(x) for x in values):
        return None
    px = float(close.iloc[-1])
    cloud_top, cloud_bottom = max(float(span_a.iloc[-1]), float(span_b.iloc[-1])), min(float(span_a.iloc[-1]), float(span_b.iloc[-1]))
    cloud_position = "雲の上" if px > cloud_top else "雲の下" if px < cloud_bottom else "雲の中"
    ma75_up = float(ma75.iloc[-1]) >= float(ma75.iloc[-6])
    ma200_up = float(ma200.iloc[-1]) >= float(ma200.iloc[-6])
    tenkan_above = float(tenkan.iloc[-1]) > float(kijun.iloc[-1])
    tenkan_cross = bool(tenkan.iloc[-2] <= kijun.iloc[-2] and tenkan.iloc[-1] > kijun.iloc[-1])
    tenkan_cross_down = bool(tenkan.iloc[-2] >= kijun.iloc[-2] and tenkan.iloc[-1] < kijun.iloc[-1])
    chikou_confirmed = px > float(close.iloc[-27])
    eligible = bool(cloud_position == "雲の上" and ma75_up and ma200_up and px >= float(ma200.iloc[-1]))
    short_eligible = bool(cloud_position == "雲の下" and not ma75_up and not ma200_up
                          and px <= float(ma200.iloc[-1]))
    if cloud_position == "雲の下":
        reason = "一目均衡表の雲の下"
    elif cloud_position == "雲の中":
        reason = "一目均衡表の雲の中（監視のみ）"
    elif not ma75_up or not ma200_up:
        reason = "75日線または200日線が下向き"
    elif px < float(ma200.iloc[-1]):
        reason = "株価が200日線より下"
    else:
        reason = "上昇基調の押し目対象"
    return {
        "cloud_top": cloud_top, "cloud_bottom": cloud_bottom,
        "cloud_position": cloud_position, "tenkan": float(tenkan.iloc[-1]),
        "kijun": float(kijun.iloc[-1]), "tenkan_above_kijun": tenkan_above,
        "tenkan_cross_up": tenkan_cross, "tenkan_cross_down": tenkan_cross_down,
        "chikou_confirmed": chikou_confirmed,
        "ma75_up": ma75_up, "ma200_up": ma200_up,
        "buy_eligible": eligible, "short_eligible": short_eligible,
        "trend_reason": reason,
    }


def calculate(code, name, hist):
    if hist is None or hist.empty:
        return None
    if isinstance(hist.columns, pd.MultiIndex):
        hist = hist.copy()
        hist.columns = hist.columns.get_level_values(0)
    required = {"Close", "High", "Low"}
    if not required.issubset(hist.columns):
        return None
    frame = hist.copy().dropna(subset=list(required))
    close = pd.to_numeric(frame["Close"], errors="coerce").dropna()
    if len(close) < 80:
        return None
    high = pd.to_numeric(frame.loc[close.index, "High"], errors="coerce")
    low = pd.to_numeric(frame.loc[close.index, "Low"], errors="coerce")
    ma5, ma25 = close.rolling(5).mean(), close.rolling(25).mean()
    ma75, ma200 = close.rolling(75).mean(), close.rolling(200).mean()
    std = close.rolling(25).std()
    upper, lower = ma25 + 2 * std, ma25 - 2 * std
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    prev = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev).abs(), (low-prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    psar = parabolic_sar(high, low)
    # A buy turn is only the latest bar changing from SAR above price to SAR below price.
    psar_bull = close > psar
    psar_buy_turn = bool(len(psar_bull) >= 2 and (not bool(psar_bull.iloc[-2])) and bool(psar_bull.iloc[-1]))
    latest_bar = close.index[-1]
    try:
        psar_bar_time = latest_bar.isoformat()
    except AttributeError:
        psar_bar_time = str(latest_bar)
    volume = pd.to_numeric(frame.get("Volume"), errors="coerce") if "Volume" in frame else None
    vol, vr = None, None
    if volume is not None and not volume.dropna().empty:
        vol = float(volume.dropna().iloc[-1])
        base = volume.iloc[-21:-1].mean() if len(volume) >= 21 else np.nan
        vr = float(vol/base) if pd.notna(base) and base > 0 else None
    px, rv = float(close.iloc[-1]), float(rsi14(close).iloc[-1])
    m25, m75 = float(ma25.iloc[-1]), float(ma75.iloc[-1])
    blo = float(lower.iloc[-1])
    bb_pos = float((px-m25) / max(float(std.iloc[-1]), 1e-9))
    score = float(np.clip((55-rv)/30*35, 0, 35))
    score += 25.0 if px <= blo*1.02 else float(np.clip((m25-px)/max(m25-blo,1e-9)*20,0,20))
    score += 12.0 if m25 >= m75 else 4.0
    score += 8.0 if float(macd.iloc[-1]) >= float(signal.iloc[-1]) else 2.0
    score += float(np.clip(((vr or 1)-0.8)*6, 0, 8))
    return {
        "code": str(code), "name": name or str(code), "price": px,
        "rsi": rv, "bb_position": bb_pos, "ma5": float(ma5.iloc[-1]),
        "ma25": m25, "ma75": m75,
        "ma200": float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else None,
        "macd": float(macd.iloc[-1]), "macd_signal": float(signal.iloc[-1]),
        "volume": vol, "volume_ratio": vr, "psar": float(psar.iloc[-1]),
        "psar_bull": bool(psar_bull.iloc[-1]), "psar_buy_turn": psar_buy_turn,
        "psar_bar_time": psar_bar_time,
        "atr": float(atr.iloc[-1]), "score": float(np.clip(score, 0, 100)),
    }


def download_and_calculate(code, name=None, intraday=False):
    period, interval = ("60d", "15m") if intraday else ("14mo", "1d")
    hist = yf.download(f"{code}.T", period=period, interval=interval,
                       auto_adjust=False, progress=False, threads=False)
    result = calculate(code, name, hist)
    if result is None:
        return None
    daily = hist if not intraday else yf.download(
        f"{code}.T", period="14mo", interval="1d", auto_adjust=False,
        progress=False, threads=False)
    trend = daily_trend_context(daily)
    if trend:
        result.update(trend)
        # RSI is only a watch trigger. A failed daily trend gate can never
        # become a buy-ready state from a high intraday countertrend score.
        if not trend["buy_eligible"]:
            result["score"] = min(result["score"], 44.0)
        elif trend["tenkan_cross_up"] or trend["tenkan_above_kijun"]:
            result["score"] = min(100.0, result["score"] + 8.0)
    return result
