"""Lightweight 15-minute updater for candidates already being monitored."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import requests

from stocknote_technicals import download_and_calculate
from stocknote_tracking import filter_new_notifications, load_active, update_active

WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
JST = ZoneInfo("Asia/Tokyo")


def is_market_session(now=None):
    now = now or datetime.now(JST)
    if now.weekday() >= 5:
        return False
    minute = now.hour * 60 + now.minute
    return 9 * 60 <= minute <= 11 * 60 + 30 or 12 * 60 + 30 <= minute <= 15 * 60 + 30


def notify(events):
    if not events:
        print("No important state changes; Discord notification skipped.")
        return
    lines = ["📌 **Stocknote 監視候補の状態変化**"]
    for event in events:
        price = event.get("price")
        price_text = f" / ¥{price:,.0f}" if isinstance(price, (int, float)) else ""
        lines.append(f"• **{event['code']} {event['name']}**: {event.get('from') or '未登録'} → {event['to']}{price_text}")
    text = "\n".join(lines)[:1950]
    print(text)
    if WEBHOOK:
        response = requests.post(WEBHOOK, json={"content": text}, timeout=20)
        response.raise_for_status()


def main(force=False):
    if not force and not is_market_session():
        print("Outside Tokyo Stock Exchange session; update skipped.")
        return
    rows = []
    for item in load_active():
        try:
            result = download_and_calculate(item["code"], item.get("name"), intraday=True)
            if result:
                rows.append(result)
        except Exception as exc:
            print(f"WARN {item.get('code')}: {exc}")
    events = update_active(rows)
    notify(filter_new_notifications(events))
    print(f"Updated {len(rows)}/{len(load_active())} active candidates.")


if __name__ == "__main__":
    main(force=os.getenv("STOCKNOTE_FORCE_UPDATE") == "1")
