"""Readable price changes shared by both monitored-candidate tables."""
import math
import re
from datetime import datetime
from zoneinfo import ZoneInfo


def _number(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def price_change(candidate):
    """Compare the current per-share price with the first detection price."""
    first = _number(candidate.get("first_price"))
    current = _number(candidate.get("current_price"))
    if first is None or first <= 0 or current is None or current <= 0:
        return None, None
    difference = round(current - first, 6)
    return difference, difference / first * 100


def signed_change(value, percent=False):
    if value is None:
        return "—"
    rounded = round(value, 2)
    if rounded == 0:
        return "0.00%" if percent else "0円"
    if percent:
        return f"{rounded:+.2f}%"
    return f"{rounded:+,.2f}".rstrip("0").rstrip(".") + "円"


def candidate_code(code):
    code = str(code or "").strip().upper().removesuffix(".T")
    if not re.fullmatch(r"[0-9A-Z]{4}", code):
        return None
    return code


def candidate_day(candidate, today=None):
    """Count calendar days inclusively, starting with detection day as day 1."""
    jst = ZoneInfo("Asia/Tokyo")
    today = today or datetime.now(jst).date()
    try:
        detected = datetime.fromisoformat(candidate.get("first_seen_at", ""))
        if detected.tzinfo is not None:
            detected = detected.astimezone(jst)
        return max(0, (today - detected.date()).days) + 1
    except (TypeError, ValueError):
        elapsed = _number(candidate.get("elapsed_days"))
        return max(0, int(elapsed)) + 1 if elapsed is not None else None


def candidate_status(candidate):
    day = candidate_day(candidate)
    status = "買いサイン" if candidate.get("ma5_cross_up") else "5日線上抜け待ち"
    return f"{status}（{day}日目）" if day is not None else status


def candidate_row(candidate):
    amount, percent = price_change(candidate)
    return {
        "コード": candidate_code(candidate.get("code")),
        "銘柄名": candidate.get("name"),
        "状態": candidate_status(candidate),
        "現在値": candidate.get("current_price"),
        "値上がり・値下がり": signed_change(amount),
        "上昇・下落（%）": signed_change(percent, percent=True),
        "初回価格": candidate.get("first_price"),
        "初回検出日時": candidate.get("first_seen_at"),
        "最終更新日時": candidate.get("updated_at"),
    }


def selected_candidate(candidates, selection):
    cells = selection.get("cells", [])
    rows = selection.get("rows", [])
    index = cells[0][0] if cells else rows[0] if rows else None
    if isinstance(index, int) and 0 <= index < len(candidates):
        return candidates[index]
    return None


def render_candidate_table(candidates, key="candidate_table"):
    import pandas as pd
    import streamlit as st

    st.caption("初回検出時からの1株あたりの変化です。＋は値上がり、−は値下がり。状態の日数は検出日を1日目として数えます。表の銘柄名・コードをクリックすると、すぐ下に日足チャートが出ます。")
    event = st.dataframe(
        pd.DataFrame([candidate_row(candidate) for candidate in candidates]),
        hide_index=True,
        width="stretch",
        key=key,
        on_select="rerun",
        selection_mode="single-cell",
        alt="監視中の買い候補の株価、初回検出からの値上がり・値下がり、経過日数と日足チャート",
        column_config={
            "コード": st.column_config.TextColumn(width=70),
            "銘柄名": st.column_config.TextColumn(width=165),
            "状態": st.column_config.TextColumn(width=145),
            "値上がり・値下がり": st.column_config.TextColumn(width=120),
            "上昇・下落（%）": st.column_config.TextColumn(width=110),
        },
    )
    selected = selected_candidate(candidates, event.selection)
    if selected:
        from stocknote_candidate_chart import render_daily_chart
        render_daily_chart(selected)
