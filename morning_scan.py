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
from stocknote_tracking import filter_new_notifications, merge_new_candidates

UNIVERSE = os.getenv("STOCKNOTE_UNIVERSE", "data/saved_universe.csv")
WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
TOP_N = int(os.getenv("STOCKNOTE_TOP_N", "10"))
MIN_SCORE = float(os.getenv("STOCKNOTE_MIN_BUY_SCORE", "55"))


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


def score_one(code, name):
    h = yf.download(f"{code}.T", period="14mo", interval="1d", auto_adjust=False,
                    progress=False, threads=False)
    if isinstance(h.columns, pd.MultiIndex):
        h.columns = h.columns.get_level_values(0)
    if h.empty or "Close" not in h or len(h) < 80:
        return None
    close = pd.to_numeric(h["Close"], errors="coerce").dropna()
    if len(close) < 80:
        return None
    rsi = rsi14(close)
    ma25 = close.rolling(25).mean()
    ma75 = close.rolling(75).mean()
    ma200 = close.rolling(200).mean()
    std25 = close.rolling(25).std()
    lower = ma25 - 2 * std25
    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    px, rv = float(close.iloc[-1]), float(rsi.iloc[-1])
    m25, m75 = float(ma25.iloc[-1]), float(ma75.iloc[-1])
    m200 = float(ma200.iloc[-1]) if pd.notna(ma200.iloc[-1]) else np.nan
    blo = float(lower.iloc[-1])
    md, sg = float(macd.iloc[-1]), float(signal.iloc[-1])
    vr = 1.0
    if "Volume" in h:
        v = pd.to_numeric(h["Volume"], errors="coerce").dropna()
        if len(v) >= 21 and float(v.iloc[-21:-1].mean()) > 0:
            vr = float(v.iloc[-1] / v.iloc[-21:-1].mean())
    reversal = False
    if "Open" in h and len(h) >= 2:
        o = pd.to_numeric(h["Open"], errors="coerce")
        if pd.notna(o.iloc[-1]) and pd.notna(o.iloc[-2]):
            reversal = bool(close.iloc[-2] < o.iloc[-2] and close.iloc[-1] > o.iloc[-1]
                            and close.iloc[-1] >= o.iloc[-2] and o.iloc[-1] <= close.iloc[-2])
    buy_rsi = float(np.clip((55-rv)/30*35, 0, 35))
    buy_bb = 25.0 if px <= blo*1.02 else float(np.clip((m25-px)/max(m25-blo,1e-9)*20,0,20))
    buy_trend = (12.0 if m25 >= m75 else 4.0) + (5.0 if np.isfinite(m200) and px >= m200 else 0.0)
    buy_macd = 8.0 if md >= sg else 2.0
    buy_volume = float(np.clip((vr-0.8)*6,0,8))
    buy_candle = 12.0 if reversal else 0.0
    score = float(np.clip(buy_rsi+buy_bb+buy_trend+buy_macd+buy_volume+buy_candle,0,100))
    return {"code":code,"name":name or code,"price":px,"rsi":rv,"vr":vr,"score":score,"reversal":reversal}


def post_discord(rows, total):
    now = datetime.now().strftime("%Y/%m/%d %H:%M")
    if rows:
        lines = [f"📊 **Stocknote 朝の買い候補**  {now}", f"母集団 {total}銘柄 / 上位 {len(rows)}銘柄"]
        for i, r in enumerate(rows, 1):
            candle = " / 包み陽線" if r["reversal"] else ""
            lines.append(f"{i}. **{r['code']} {r['name']}**  score {r['score']:.1f} / RSI {r['rsi']:.1f} / ¥{r['price']:,.0f} / 出来高 {r['vr']:.2f}倍{candle}")
    else:
        lines = [f"📊 **Stocknote 朝スキャン**  {now}", f"母集団 {total}銘柄を確認しましたが、分析可能な買い候補はありませんでした。"]
    text = "\n".join(lines)
    print(text)
    if not WEBHOOK:
        print("DISCORD_WEBHOOK is not configured; skipping Discord post.")
        return
    resp = requests.post(WEBHOOK, json={"content": text[:1950]}, timeout=20)
    resp.raise_for_status()


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
    results=[]
    for i,(code,name) in enumerate(items,1):
        try:
            r=score_one(code,name)
            if r: results.append(r)
        except Exception as exc:
            print(f"WARN {code}: {exc}", file=sys.stderr)
        if i % 25 == 0: print(f"scanned {i}/{len(items)}")
    results.sort(key=lambda x:x["score"], reverse=True)
    candidates = [row for row in results if row["score"] >= MIN_SCORE][:TOP_N]
    # This is an upsert, never a replacement: a zero-result scan leaves the
    # active 14-day monitoring list untouched.
    events = merge_new_candidates(candidates)
    notify(filter_new_notifications(events))
    print(f"new scan: {len(candidates)} candidates from {len(items)} stocks")


if __name__ == "__main__":
    main()
