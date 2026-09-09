"""Lightweight 15-minute updater for candidates already being monitored."""
import json
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

from stocknote_technicals import download_and_calculate
from stocknote_tracking import filter_new_notifications, load_active, update_active

WEBHOOK = os.getenv("DISCORD_WEBHOOK", "").strip()
JST = ZoneInfo("Asia/Tokyo")
PSAR_NOTICE_PATH = Path(os.getenv("STOCKNOTE_DATA_DIR", "data")) / "psar_notice_state.json"
ENTRY_RULE_VERSION = 4


def is_market_session(now=None):
    now = now or datetime.now(JST)
    if now.weekday() >= 5:
        return False
    minute = now.hour * 60 + now.minute
    return 9 * 60 <= minute <= 11 * 60 + 30 or 12 * 60 + 30 <= minute <= 15 * 60 + 30


def _load_psar_notice_state():
    try:
        with PSAR_NOTICE_PATH.open(encoding="utf-8") as fh:
            value = json.load(fh)
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_psar_notice_state(state):
    PSAR_NOTICE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = PSAR_NOTICE_PATH.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    os.replace(tmp, PSAR_NOTICE_PATH)


def psar_buy_events(rows):
    """Return only never-notified latest-bar Parabolic SAR sell-to-buy turns."""
    state = _load_psar_notice_state()
    events = []
    for row in rows:
        # 買い確認は、新しい逆張りルールで登録され、現在も日足条件を
        # 維持している候補の「最新15分足SAR転換」だけに限定する。
        if row.get("entry_rule_version") != ENTRY_RULE_VERSION:
            continue
        if not row.get("buy_eligible") or not row.get("psar_buy_turn"):
            continue
        code = str(row.get("code", "")).strip()
        bar_time = str(row.get("psar_bar_time", "")).strip()
        if not code or not bar_time or state.get(code) == bar_time:
            continue
        events.append({
            "code": code,
            "name": row.get("name") or code,
            "price": row.get("price"),
            "bar_time": bar_time,
        })
        state[code] = bar_time
    _save_psar_notice_state(state)
    return events


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


def notify_psar(events):
    if not events:
        print("No new Parabolic SAR buy turns.")
        return
    lines = ["✅ **Stocknote 逆張り＋15分足パラボリック買い確認**"]
    for event in events:
        price = event.get("price")
        price_text = f" / ¥{price:,.0f}" if isinstance(price, (int, float)) else ""
        lines.append(f"• **{event['code']} {event['name']}**: 逆張り候補でSAR 売り → 買い{price_text}")
    text = "\n".join(lines)[:1950]
    print(text)
    if WEBHOOK:
        response = requests.post(WEBHOOK, json={"content": text}, timeout=20)
        response.raise_for_status()


def main(force=False):
    if not force and not is_market_session():
        print("Outside Tokyo Stock Exchange session; update skipped.")
        return
    active = load_active()
    rows = []
    for item in active:
        # 旧条件の候補から買い通知を出さない。履歴自体は削除せず保存する。
        if item.get("entry_rule_version") != ENTRY_RULE_VERSION:
            continue
        try:
            result = download_and_calculate(item["code"], item.get("name"), intraday=True)
            if result:
                result["entry_rule_version"] = ENTRY_RULE_VERSION
                rows.append(result)
        except Exception as exc:
            print(f"WARN {item.get('code')}: {exc}")
    # Alert only on a sell-to-buy turn occurring on the latest 15-minute bar.
    notify_psar(psar_buy_events(rows))
    events = update_active(rows)
    # 点数の上昇だけでは買い通知を出さない。条件悪化・監視終了のみ通知する。
    risk_events = [
        event for event in filter_new_notifications(events)
        if event.get("to") in {"条件悪化", "見送り", "監視終了"}
    ]
    notify(risk_events)
    print(f"Updated {len(rows)}/{len(active)} active candidates.")


if __name__ == "__main__":
    main(force=os.getenv("STOCKNOTE_FORCE_UPDATE") == "1")
