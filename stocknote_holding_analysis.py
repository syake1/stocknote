"""Current analysis for registered holdings, independent of scanner rankings."""
from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

from stocknote_fundamentals import get_fundamentals
from stocknote_technicals import daily_trend_context, quarterly_strength_context, rsi14


MARKETS = {
    "日経平均": "^N225", "日経225先物": "NKD=F", "SOX半導体指数": "^SOX",
    "S&P500": "^GSPC", "NASDAQ": "^IXIC", "VIX": "^VIX",
    "WTI原油": "CL=F", "ドル円": "JPY=X",
}


def combined_buy_score(technical_score, fundamental_score, market_score):
    return float(np.clip(
        float(technical_score) * 0.50
        + float(fundamental_score) * 0.35
        + float(market_score) * 0.15,
        0, 100,
    ))


def _download(symbol, period="14mo"):
    try:
        hist = yf.download(symbol, period=period, interval="1d", auto_adjust=False,
                           progress=False, threads=False)
        if isinstance(hist.columns, pd.MultiIndex):
            hist.columns = hist.columns.get_level_values(0)
        return hist.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def technical_buy_score(code):
    hist = _download(f"{code}.T", "5y")
    if hist.empty or "Close" not in hist:
        return None
    close = pd.to_numeric(hist["Close"], errors="coerce").dropna()
    if len(close) < 80:
        return None
    rsi = rsi14(close)
    ma25, ma75, ma200 = close.rolling(25).mean(), close.rolling(75).mean(), close.rolling(200).mean()
    std25 = close.rolling(25).std()
    upper, lower = ma25 + 2 * std25, ma25 - 2 * std25
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    trend = daily_trend_context(hist)
    quarterly = quarterly_strength_context(hist)
    px, rv = float(close.iloc[-1]), float(rsi.iloc[-1])
    m25, m75 = float(ma25.iloc[-1]), float(ma75.iloc[-1])
    m200 = float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else np.nan
    blo = float(lower.iloc[-1])
    bup = float(upper.iloc[-1])
    daily_bb_overextended = bool(px > bup * 1.03)
    vr = 1.0
    if "Volume" in hist:
        volume = pd.to_numeric(hist["Volume"], errors="coerce").dropna()
        if len(volume) >= 21 and float(volume.iloc[-21:-1].mean()) > 0:
            vr = float(volume.iloc[-1] / volume.iloc[-21:-1].mean())
    reversal = False
    if "Open" in hist and len(hist) >= 2:
        opens = pd.to_numeric(hist.loc[close.index, "Open"], errors="coerce")
        reversal = bool(close.iloc[-2] < opens.iloc[-2] and close.iloc[-1] > opens.iloc[-1]
                        and close.iloc[-1] >= opens.iloc[-2] and opens.iloc[-1] <= close.iloc[-2])
    score = (
        float(np.clip((55 - rv) / 30 * 35, 0, 35))
        + (25.0 if px <= blo * 1.02 else float(np.clip((m25 - px) / max(m25 - blo, 1e-9) * 20, 0, 20)))
        + (12.0 if m25 >= m75 else 4.0)
        + (5.0 if np.isfinite(m200) and px >= m200 else 0.0)
        + (8.0 if float(macd.iloc[-1]) >= float(signal.iloc[-1]) else 2.0)
        + float(np.clip((vr - 0.8) * 6, 0, 8))
        + (12.0 if reversal else 0.0)
        + (8.0 if trend and trend["tenkan_cross_up"] else 5.0 if trend and trend["tenkan_above_kijun"] else 0.0)
    )
    score = float(np.clip(score, 0, 100))
    if (not trend or not trend["buy_eligible"] or not quarterly
            or not quarterly["quarterly_qualified"] or daily_bb_overextended):
        score = min(score, 44.0)
    return {
        "current_price": px, "technical_score": score, "rsi": rv,
        "volume_ratio": vr, "cloud_position": trend["cloud_position"] if trend else "データ不足",
        "trend_reason": trend["trend_reason"] if trend else "日足履歴不足",
        "quarterly_score": quarterly["quarterly_score"] if quarterly else None,
        "quarterly_rsi": quarterly["quarterly_rsi"] if quarterly else None,
        "quarterly_reason": quarterly["quarterly_reason"] if quarterly else "四半期足の履歴不足",
        "quarterly_pattern": quarterly["quarterly_pattern"] if quarterly else "履歴不足",
        "quarterly_high_level_consolidation": quarterly["quarterly_high_level_consolidation"] if quarterly else None,
        "quarterly_large_upper_wick": quarterly["quarterly_large_upper_wick"] if quarterly else None,
        "monthly_large_upper_wick": quarterly["monthly_large_upper_wick"] if quarterly else None,
        "daily_bb_overextended": daily_bb_overextended,
    }


def market_score():
    data = {}
    for name, symbol in MARKETS.items():
        hist = _download(symbol, "2mo")
        close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna() if not hist.empty else pd.Series(dtype=float)
        if len(close) >= 6:
            data[name] = {"now": float(close.iloc[-1]), "d5": float((close.iloc[-1] / close.iloc[-6] - 1) * 100)}
    score = 50.0
    for name in ("S&P500", "NASDAQ", "日経平均", "日経225先物"):
        value = data.get(name, {}).get("d5")
        if value is not None:
            score += 4 if value >= 1.5 else -4 if value <= -1.5 else 0
    sox = data.get("SOX半導体指数", {}).get("d5")
    if sox is not None:
        score += 8 if sox >= 2 else -8 if sox <= -2 else 0
    vix = data.get("VIX", {}).get("now")
    if vix is not None:
        score += -15 if vix >= 30 else -8 if vix >= 22 else 5 if vix <= 16 else 0
    oil = data.get("WTI原油", {}).get("d5")
    if oil is not None and oil >= 8:
        score -= 5
    fx = data.get("ドル円", {}).get("d5")
    if fx is not None:
        score += 3 if 0.5 <= fx <= 3 else -3 if fx <= -2 else 0
    return float(np.clip(score, 0, 100))


def analyze_holding(code, current_market_score):
    technical = technical_buy_score(code)
    if technical is None:
        return {"error": "株価データを取得できず、現在の総合得点を計算できません"}
    fundamental = get_fundamentals(code)
    fundamental_score = float(fundamental["score"])
    return {
        **technical,
        "fundamental_score": fundamental_score,
        "market_score": float(current_market_score),
        "total_score": combined_buy_score(technical["technical_score"], fundamental_score, current_market_score),
        "per": fundamental.get("per"),
        "fundamental_comment": fundamental.get("comment", ""),
        "error": None,
    }
