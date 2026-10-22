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
        day1 = float(current["1日%"])
        day5 = float(current["5日%"])
        day20 = float(current["20日%"])
        ma5_distance = float(current.get("5日線乖離%", 0.0) or 0.0)
        overheated = bool(
            day1 > 4.0 or day5 >= 7.0 or day20 >= 15.0 or ma5_distance > 3.0
        )
        fresh_sector = bool(
            current.get("業種判定") == "循環初動"
            and 0 < day1 <= 4.0
            and -3.5 <= day5 < 4.0
            and day20 <= 12.0
            and -2.0 <= ma5_distance <= 2.0
        )
        accelerating = bool(
            not overheated
            and rank <= 10
            and 0 < day1 <= 4.0
            and day20 <= 12.0
            and (
                bool(current.get("5日線上抜け"))
                or fresh_sector
                or (
                    latest and appearances >= 2
                    and ((rank_change is not None and rank_change > 0) or acceleration >= 0.75)
                )
            )
        )
        if current.get("5日線上抜け"):
            reason = "5日線上抜け"
        elif fresh_sector:
            reason = "業種初動＋5日線付近"
        elif accelerating:
            reason = "順位改善＋上昇加速"
        elif overheated:
            reason = "上昇後・過熱"
        else:
            reason = "条件待ち"
        if accelerating:
            if float(market_score) <= 38.0:
                handling = "寄り付き確認（地合い逆風）"
            elif current.get("5日線上抜け"):
                handling = "翌朝候補（寄り付き確認）"
            else:
                handling = "先回り監視（5日線上抜け待ち）"
        elif overheated:
            handling = "上昇後・追わない"
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
            "初動理由": reason,
            "過熱判定": "過熱" if overheated else "未過熱",
            "初動段階": current.get("初動段階", "条件待ち"),
            "翌朝の扱い": handling,
        })
    results.sort(key=lambda row: (
        0 if row["初動加速"] else 1,
        1 if row["過熱判定"] == "過熱" else 0,
        0 if row.get("業種判定") == "循環初動" else 1,
        -(row.get("順位変化") or 0),
        -float(row.get("循環点", 0.0) or 0.0),
    ))
    return results
