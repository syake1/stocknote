"""Persist sector-leader rankings and detect early acceleration patterns."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path


DEFAULT_PATH = Path("data/sector_rotation_history.json")


def load_rotation_history(path=DEFAULT_PATH):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, list) else []
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return []


def save_rotation_snapshot(candidates, asof=None, path=DEFAULT_PATH, keep_days=30):
    snapshot_date = str(asof or date.today())[:10]
    rows = []
    for rank, item in enumerate(candidates, 1):
        rows.append({
            "rank": rank,
            "sector": item["業種・テーマ"],
            "code": item["コード"],
            "name": item["銘柄名"],
            "day1": float(item["1日%"]),
            "day5": float(item["5日%"]),
            "day20": float(item["20日%"]),
            "score": float(item["循環点"]),
        })
    history = [item for item in load_rotation_history(path) if item.get("date") != snapshot_date]
    history.append({"date": snapshot_date, "candidates": rows})
    history.sort(key=lambda item: item.get("date", ""))
    history = history[-keep_days:]
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".tmp")
    temp.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)
    return history


def analyze_rotation_patterns(candidates, history, market_score=50.0, asof=None):
    """Compare today's leaders with earlier daily rankings."""
    today = str(asof or date.today())[:10]
    past = sorted(
        [item for item in history or [] if item.get("date", "") < today],
        key=lambda item: item.get("date", ""),
    )[-5:]
    results = []
    for rank, current in enumerate(candidates, 1):
        code = current["コード"]
        previous = []
        for snapshot in past:
            match = next((row for row in snapshot.get("candidates", []) if row.get("code") == code), None)
            if match:
                previous.append((snapshot.get("date"), match))
        appearances = len(previous) + 1
        latest = previous[-1][1] if previous else None
        previous_rank = int(latest["rank"]) if latest else None
        rank_change = previous_rank - rank if previous_rank is not None else None
        acceleration = float(current["5日%"] - latest["day5"]) if latest else None
        accelerating = bool(
            latest
            and rank <= 5
            and appearances >= 2
            and 0 < float(current["1日%"]) <= 4.0
            and float(current["20日%"]) <= 12.0
            and ((rank_change is not None and rank_change > 0) or acceleration >= 0.75)
        )
        if accelerating:
            handling = (
                "寄り付き確認（地合い逆風）"
                if float(market_score) <= 38.0
                else "翌朝の成行検討（手動確認）"
            )
        else:
            handling = "継続監視"
        results.append({
            **current,
            "本日順位": rank,
            "過去5回登場": appearances,
            "前回順位": previous_rank,
            "順位変化": rank_change,
            "5日加速pt": acceleration,
            "初動加速": accelerating,
            "翌朝の扱い": handling,
        })
    return results
