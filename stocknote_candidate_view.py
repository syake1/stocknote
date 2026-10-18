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


def daily_chart_url(code):
    code = str(code or "").strip().upper().removesuffix(".T")
    if not re.fullmatch(r"[0-9A-Z]{4}", code):
        return None
    return f"https://finance.yahoo.co.jp/quote/{code}.T/chart?styl=cndl&frm=dly"


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
    status = candidate.get("status") or "監視中"
    return f"{status}（{day}日目）" if day is not None else status


def candidate_row(candidate):
    amount, percent = price_change(candidate)
    return {
        "コード（日足）": daily_chart_url(candidate.get("code")),
        "銘柄名": candidate.get("name"),
        "現在値": candidate.get("current_price"),
        "値上がり・値下がり": signed_change(amount),
        "上昇・下落（%）": signed_change(percent, percent=True),
        "初回価格": candidate.get("first_price"),
        "状態": candidate_status(candidate),
        "初回検出日時": candidate.get("first_seen_at"),
        "最終更新日時": candidate.get("updated_at"),
    }


def render_candidate_table(candidates):
    import pandas as pd
    import streamlit as st

    st.caption("初回検出時からの1株あたりの変化です。＋は値上がり、−は値下がり。状態の日数は検出日を1日目として数えます。銘柄コードを押すと日足チャートが開きます。")
    st.dataframe(
        pd.DataFrame([candidate_row(candidate) for candidate in candidates]),
        hide_index=True,
        width="stretch",
        alt="監視中の買い候補の株価、初回検出からの値上がり・値下がり、経過日数と日足チャート",
        column_config={
            "コード（日足）": st.column_config.LinkColumn(
                "コード（日足）",
                display_text=r"https://finance\.yahoo\.co\.jp/quote/([0-9A-Z]+)\.T/chart.*",
            ),
        },
    )
