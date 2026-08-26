import numpy as np
import pandas as pd
import yfinance as yf


def _num(value):
    try:
        value = float(value)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def _first_row(frame, labels):
    if frame is None or frame.empty:
        return None
    for label in labels:
        if label in frame.index:
            series = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
            if not series.empty:
                return _num(series.iloc[0])
    return None


def _ttm_sum(frame, labels):
    if frame is None or frame.empty:
        return None
    for label in labels:
        if label in frame.index:
            values = pd.to_numeric(frame.loc[label], errors="coerce").dropna()
            if len(values) >= 4:
                return _num(values.iloc[:4].sum())
            if not values.empty:
                return _num(values.iloc[0])
    return None


def get_fundamentals(code):
    ticker = yf.Ticker(f"{code}.T")
    try:
        info = ticker.info or {}
    except Exception:
        info = {}

    sources = {}
    per = _num(info.get("trailingPE"))
    pbr = _num(info.get("priceToBook"))
    roe = _num(info.get("returnOnEquity"))
    opm = _num(info.get("operatingMargins"))
    growth = _num(info.get("revenueGrowth"))
    dividend_yield = _num(info.get("dividendYield"))
    if dividend_yield is not None and dividend_yield > 1:
        dividend_yield /= 100

    for key, value in [("per", per), ("pbr", pbr), ("roe", roe), ("opm", opm), ("growth", growth), ("div", dividend_yield)]:
        if value is not None:
            sources[key] = "Yahoo Finance"

    try:
        bs = ticker.balance_sheet
    except Exception:
        bs = pd.DataFrame()
    try:
        inc = ticker.financials
    except Exception:
        inc = pd.DataFrame()
    try:
        qinc = ticker.quarterly_financials
    except Exception:
        qinc = pd.DataFrame()

    equity = _first_row(bs, ["Stockholders Equity", "Total Stockholder Equity", "StockholdersEquity", "Common Stock Equity"])
    assets = _first_row(bs, ["Total Assets", "TotalAssets"])
    equity_ratio = equity / assets if equity is not None and assets not in (None, 0) else None
    if equity_ratio is not None:
        sources["equity_ratio"] = "財務諸表から計算"

    annual_revenue = _first_row(inc, ["Total Revenue", "TotalRevenue", "Operating Revenue"])
    annual_operating = _first_row(inc, ["Operating Income", "OperatingIncome"])
    annual_net = _first_row(inc, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    ttm_net = _ttm_sum(qinc, ["Net Income", "NetIncome", "Net Income Common Stockholders"])
    net_income = ttm_net if ttm_net is not None else annual_net

    if opm is None and annual_revenue not in (None, 0) and annual_operating is not None:
        opm = annual_operating / annual_revenue
        sources["opm"] = "財務諸表から計算"

    if growth is None and inc is not None and not inc.empty:
        for label in ["Total Revenue", "TotalRevenue", "Operating Revenue"]:
            if label in inc.index:
                vals = pd.to_numeric(inc.loc[label], errors="coerce").dropna()
                if len(vals) >= 2 and vals.iloc[1] != 0:
                    growth = float(vals.iloc[0] / vals.iloc[1] - 1)
                    sources["growth"] = "財務諸表から計算"
                break

    current_price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    market_cap = _num(info.get("marketCap"))
    shares = _num(info.get("sharesOutstanding"))

    try:
        fast = ticker.fast_info
        if current_price is None:
            current_price = _num(fast.get("last_price"))
        if market_cap is None:
            market_cap = _num(fast.get("market_cap"))
        if shares is None:
            shares = _num(fast.get("shares"))
    except Exception:
        pass

    if current_price is None:
        try:
            hist = ticker.history(period="5d", auto_adjust=False)
            close = pd.to_numeric(hist.get("Close"), errors="coerce").dropna()
            if not close.empty:
                current_price = _num(close.iloc[-1])
        except Exception:
            pass

    if market_cap is None and current_price is not None and shares is not None:
        market_cap = current_price * shares

    if per is None and market_cap is not None and net_income is not None and net_income > 0:
        per = market_cap / net_income
        sources["per"] = "時価総額÷純利益で計算"

    if pbr is None and market_cap is not None and equity is not None and equity > 0:
        pbr = market_cap / equity
        sources["pbr"] = "時価総額÷自己資本で計算"

    if roe is None and net_income is not None and equity is not None and equity > 0:
        roe = net_income / equity
        sources["roe"] = "純利益÷自己資本で計算"

    if dividend_yield is None and current_price not in (None, 0):
        try:
            dividends = ticker.dividends
            if dividends is not None and not dividends.empty:
                idx = dividends.index
                if getattr(idx, "tz", None) is not None:
                    now = pd.Timestamp.now(tz=idx.tz)
                else:
                    now = pd.Timestamp.now()
                cutoff = now - pd.Timedelta(days=370)
                trailing = pd.to_numeric(dividends[dividends.index >= cutoff], errors="coerce").dropna().sum()
                if trailing > 0:
                    dividend_yield = float(trailing / current_price)
                    sources["div"] = "直近約1年配当÷株価で計算"
        except Exception:
            pass

    score = 50.0
    notes = []
    if per is not None:
        if 0 < per <= 15:
            score += 8; notes.append("PER割安")
        elif per >= 35:
            score -= 7; notes.append("PER高め")
    if pbr is not None:
        if 0 < pbr <= 1.2:
            score += 7; notes.append("PBR低め")
        elif pbr >= 4:
            score -= 6; notes.append("PBR高め")
    if roe is not None:
        if roe >= 0.10:
            score += 10; notes.append("ROE良好")
        elif roe < 0:
            score -= 12; notes.append("ROEマイナス")
    if equity_ratio is not None:
        if equity_ratio >= 0.50:
            score += 6; notes.append("自己資本比率良好")
        elif equity_ratio < 0.20:
            score -= 6; notes.append("自己資本比率低め")
    if opm is not None:
        if opm >= 0.10:
            score += 8; notes.append("営業利益率良好")
        elif opm < 0:
            score -= 10; notes.append("営業赤字")
    if growth is not None:
        if growth >= 0.05:
            score += 8; notes.append("増収")
        elif growth < 0:
            score -= 7; notes.append("減収")
    if dividend_yield is not None and dividend_yield >= 0.03:
        score += 5; notes.append("配当3%以上")

    values = [per, pbr, roe, equity_ratio, opm, growth, dividend_yield]
    available = sum(v is not None for v in values)
    comment = "・".join(notes) if notes else ("大きな加減点なし" if available else "ファンダメンタル取得不可")

    return {
        "score": float(np.clip(score, 0, 100)),
        "comment": comment,
        "per": per,
        "pbr": pbr,
        "roe": roe,
        "equity_ratio": equity_ratio,
        "opm": opm,
        "growth": growth,
        "div": dividend_yield,
        "available": available,
        "sources": sources,
    }
