"""Scan all domestic TSE operating companies for verified surge setups."""

from __future__ import annotations

import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf

from stocknote_surge import (
    historical_market_order_events,
    load_jpx_universe,
    rank_surge_candidates,
    save_surge_backtest,
    save_surge_candidates,
    summarize_market_order_backtest,
    surge_snapshot,
    verified_morning_threshold,
)


WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
SEND_DISCORD = os.getenv("SURGE_SEND_DISCORD", "1") == "1"
BATCH_SIZE = int(os.getenv("SURGE_BATCH_SIZE", "80"))
RESULT_LIMIT = int(os.getenv("SURGE_RESULT_LIMIT", "100"))
UNIVERSE_LIMIT = int(os.getenv("SURGE_UNIVERSE_LIMIT", "0"))


def batch_frame(data, ticker):
    if data is None or data.empty:
        return pd.DataFrame()
    if not isinstance(data.columns, pd.MultiIndex):
        return data.copy()
    for level in range(data.columns.nlevels):
        if ticker not in data.columns.get_level_values(level):
            continue
        try:
            frame = data.xs(ticker, axis=1, level=level, drop_level=True).copy()
            if isinstance(frame.columns, pd.MultiIndex):
                frame.columns = frame.columns.get_level_values(-1)
            return frame.dropna(how="all")
        except Exception:
            continue
    return pd.DataFrame()


def download_batch(items):
    tickers = [f"{row.コード}.T" for row in items.itertuples(index=False)]
    return yf.download(
        tickers,
        period="10mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
        timeout=30,
    )


def build_discord_message(candidates, summary, threshold, scanned, failed):
    now = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y/%m/%d %H:%M")
    lines = [f"🚀 **Stocknote 東証全市場・翌朝成行候補**  {now}"]
    lines.append(f"東証全市場 {scanned}銘柄を検証 / 株価取得失敗 {failed}銘柄")
    if threshold is None:
        lines.append("⚠️ バックテスト基準未達のため、朝成候補は出しません。監視のみです。")
    else:
        selected = summary[summary["基準点"].eq(threshold)].iloc[0]
        lines.append(
            f"検証合格: {int(selected['件数'])}件 / 5日勝率 {selected['5日勝率%']:.1f}%"
            f" / 平均 {selected['5日平均損益%']:+.2f}%"
            f" / 5日以内+5% {selected['5日以内+5%到達率']:.1f}%"
        )
    morning = candidates[candidates["判定"].eq("朝成候補")].head(10) if not candidates.empty else pd.DataFrame()
    if morning.empty:
        lines.append("本日の検証済み朝成候補はありません。")
    else:
        lines.append("SBIの寄付き気配が前日終値比+3%以上なら成行見送りです。")
        for rank, row in enumerate(morning.to_dict("records"), 1):
            lines.append(
                f"{rank}. **{row['コード']} {row['銘柄名']}**  {row['急騰初動点']:.1f}点"
                f" / {row['業種']} / 前日比 {row['前日比%']:+.2f}%"
                f" / 出来高 {row['出来高倍率']:.1f}倍 / {row['理由']}"
            )
    return "\n".join(lines)[:1950]


def post_discord(message):
    print(message)
    if not WEBHOOK or not SEND_DISCORD:
        print("Discord notification skipped.")
        return
    response = requests.post(WEBHOOK, json={"content": message}, timeout=20)
    response.raise_for_status()


def main():
    universe = load_jpx_universe(refresh=True)
    if UNIVERSE_LIMIT > 0:
        universe = universe.head(UNIVERSE_LIMIT).copy()
    if universe.empty:
        raise SystemExit("JPXの東証全市場銘柄を取得できませんでした")

    snapshots = []
    backtest_events = []
    failed = 0
    total = len(universe)
    for start in range(0, total, BATCH_SIZE):
        batch = universe.iloc[start:start + BATCH_SIZE].copy()
        try:
            downloaded = download_batch(batch)
        except Exception as exc:
            print(f"WARN batch {start + 1}-{start + len(batch)}: {exc}")
            downloaded = pd.DataFrame()
        for row in batch.to_dict("records"):
            code = row["コード"]
            history = batch_frame(downloaded, f"{code}.T")
            if history.empty:
                failed += 1
                continue
            snapshot = surge_snapshot(code, row["銘柄名"], row["市場"], row["業種"], history)
            if snapshot:
                snapshots.append(snapshot)
            backtest_events.extend(historical_market_order_events(code, history))
        print(
            f"scanned {min(start + BATCH_SIZE, total)}/{total}"
            f" / snapshots {len(snapshots)} / backtest events {len(backtest_events)}"
        )
        time.sleep(0.4)

    summary = summarize_market_order_backtest(backtest_events)
    threshold = verified_morning_threshold(summary)
    candidates = rank_surge_candidates(
        snapshots,
        limit=RESULT_LIMIT,
        verified_threshold=threshold,
    )
    save_surge_backtest(summary)
    meta = save_surge_candidates(candidates, scanned=total, failed=failed)
    meta["verified_threshold"] = threshold
    meta["verification_passed"] = threshold is not None
    from stocknote_surge import SURGE_META_PATH
    import json
    with open(SURGE_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)

    message = build_discord_message(candidates, summary, threshold, total, failed)
    post_discord(message)
    print(summary.to_string(index=False))
    print(
        f"all-market surge scan complete: {len(candidates)} candidates"
        f" / verified threshold: {threshold}"
    )


if __name__ == "__main__":
    main()
