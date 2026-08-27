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
        "atr": float(atr.iloc[-1]), "score": float(np.clip(score, 0, 100)),
    }


def download_and_calculate(code, name=None, intraday=False):
    period, interval = ("60d", "15m") if intraday else ("14mo", "1d")
    hist = yf.download(f"{code}.T", period=period, interval=interval,
                       auto_adjust=False, progress=False, threads=False)
    return calculate(code, name, hist)
