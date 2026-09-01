"""Persistent candidate tracking for Stocknote.

The files managed here are deliberately separate from the scanner's current
run.  An empty scan therefore never erases candidates which are still inside
their 14-day observation window.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("STOCKNOTE_DATA_DIR", "data"))
ACTIVE_PATH = DATA_DIR / "active_buy_candidates.json"
HISTORY_PATH = DATA_DIR / "candidate_history.json"
NOTICE_PATH = DATA_DIR / "candidate_notice_state.json"
TRACKING_DAYS = int(os.getenv("STOCKNOTE_TRACKING_DAYS", "14"))


def now_iso(now=None):
    value = now or datetime.now().astimezone()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat(timespec="seconds")


def _read(path, default):
    try:
        with path.open(encoding="utf-8") as fh:
            value = json.load(fh)
        return value
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, allow_nan=False)
        fh.write("\n")
    os.replace(tmp, path)


def load_active():
    value = _read(ACTIVE_PATH, [])
    return value if isinstance(value, list) else []


def load_history():
    value = _read(HISTORY_PATH, [])
    return value if isinstance(value, list) else []


def _as_float(value):
    try:
        number = float(value)
        return number if number == number and abs(number) != float("inf") else None
    except (TypeError, ValueError):
        return None


def classify(metrics):
    """Translate current real indicators into a stable monitoring state."""
    if metrics.get("buy_eligible") is False:
        return "条件悪化" if metrics.get("cloud_position") == "雲の下" else "監視継続"
    score = _as_float(metrics.get("score") or metrics.get("買いスコア"))
    if score is None:
        return "監視継続"
    if score >= 75:
        return "買い条件到達"
    if score >= 65:
        return "買い条件接近"
    if score >= 45:
        return "監視継続"
    if score >= 30:
        return "条件悪化"
    return "見送り"


def _elapsed_days(first_seen, now):
    try:
        start = datetime.fromisoformat(first_seen)
        if start.tzinfo is None:
            start = start.replace(tzinfo=now.tzinfo)
        return max(0, (now.date() - start.astimezone(now.tzinfo).date()).days)
    except (TypeError, ValueError):
        return 0


def _snapshot(candidate):
    keys = (
        "updated_at", "elapsed_days", "current_price", "return_pct", "status",
        "rsi", "bb_position", "ma5", "ma25", "ma75", "ma200", "macd",
        "macd_signal", "volume", "volume_ratio", "psar", "atr", "score",
        "cloud_top", "cloud_bottom", "cloud_position", "tenkan", "kijun",
        "tenkan_above_kijun", "tenkan_cross_up", "chikou_confirmed",
        "ma75_up", "ma200_up", "buy_eligible", "trend_reason",
    )
    return {key: candidate.get(key) for key in keys}


def merge_new_candidates(rows, now=None):
    """Upsert a new scan without replacing candidates absent from that scan."""
    now = now or datetime.now().astimezone()
    stamp = now_iso(now)
    active = {str(r.get("code", "")): r for r in load_active() if r.get("code")}
    events = []
    for row in rows:
        code = str(row.get("code", "")).strip()
        price = _as_float(row.get("price") or row.get("current_price"))
        if not code or price is None:
            continue
        if code in active:
            current = active[code]
            current["name"] = row.get("name") or current.get("name") or code
            current["last_detected_at"] = stamp
            current.update(_metrics(row))
            _refresh_derived(current, now)
        else:
            current = {
                "code": code, "name": row.get("name") or code,
                "first_seen_at": stamp, "first_price": price,
                "last_detected_at": stamp, "updated_at": stamp,
                "elapsed_days": 0, "current_price": price, "return_pct": 0.0,
                "status": "新規候補", "previous_status": None, "snapshots": [],
            }
            current.update(_metrics(row))
            current["snapshots"].append(_snapshot(current))
            active[code] = current
            events.append({"code": code, "name": current["name"], "from": None,
                           "to": "新規候補", "price": price})
    _write(ACTIVE_PATH, sorted(active.values(), key=lambda x: x.get("first_seen_at", ""), reverse=True))
    return events


def _metrics(row):
    aliases = {
        "price": "current_price", "rsi": "rsi", "bb_position": "bb_position",
        "ma5": "ma5", "ma25": "ma25", "ma75": "ma75", "ma200": "ma200",
        "macd": "macd", "macd_signal": "macd_signal", "volume": "volume",
        "volume_ratio": "volume_ratio", "vr": "volume_ratio", "psar": "psar",
        "atr": "atr", "score": "score",
    }
    out = {}
    for source, target in aliases.items():
        if source in row:
            out[target] = _as_float(row[source])
    for key in ("cloud_position", "trend_reason", "tenkan_above_kijun",
                "tenkan_cross_up", "chikou_confirmed", "ma75_up",
                "ma200_up", "buy_eligible"):
        if key in row:
            out[key] = row[key]
    for key in ("cloud_top", "cloud_bottom", "tenkan", "kijun"):
        if key in row:
            out[key] = _as_float(row[key])
    return out


def _refresh_derived(candidate, now):
    candidate["updated_at"] = now_iso(now)
    candidate["elapsed_days"] = _elapsed_days(candidate.get("first_seen_at"), now)
    first = _as_float(candidate.get("first_price"))
    current = _as_float(candidate.get("current_price"))
    candidate["return_pct"] = ((current / first - 1) * 100) if first and current is not None else None


def update_active(metric_rows, now=None):
    """Refresh only active candidates and archive entries after 14 days."""
    now = now or datetime.now().astimezone()
    metrics = {str(r.get("code", "")): r for r in metric_rows}
    kept, finished, events = [], [], []
    for candidate in load_active():
        code = str(candidate.get("code", ""))
        old_status = candidate.get("status")
        if code in metrics:
            candidate.update(_metrics(metrics[code]))
        _refresh_derived(candidate, now)
        if candidate["elapsed_days"] >= TRACKING_DAYS:
            new_status = "監視終了"
        elif code not in metrics:
            new_status = old_status if old_status not in (None, "新規候補") else "監視継続"
            candidate["update_error"] = "市場データを取得できず、前回値を維持"
        else:
            new_status = classify(metrics[code])
            candidate.pop("update_error", None)
        candidate["previous_status"] = old_status
        candidate["status"] = new_status
        candidate.setdefault("snapshots", []).append(_snapshot(candidate))
        candidate["snapshots"] = candidate["snapshots"][-1500:]
        if new_status != old_status and new_status in {
            "買い条件接近", "買い条件到達", "条件悪化", "見送り", "監視終了"
        }:
            events.append({"code": code, "name": candidate.get("name", code),
                           "from": old_status, "to": new_status,
                           "price": candidate.get("current_price")})
        (finished if new_status == "監視終了" else kept).append(candidate)
    if finished:
        history = load_history()
        history.extend(finished)
        _write(HISTORY_PATH, history)
    _write(ACTIVE_PATH, kept)
    return events


def filter_new_notifications(events):
    """Suppress duplicate code/status notifications across workflow runs."""
    state = _read(NOTICE_PATH, {})
    send = []
    for event in events:
        code, status = str(event.get("code", "")), event.get("to")
        if code and state.get(code) != status:
            send.append(event)
            state[code] = status
    _write(NOTICE_PATH, state)
    return send
