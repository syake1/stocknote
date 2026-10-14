"""Yahoo Finance chart API client for daily prices and dividend events."""

from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import Any

from curl_cffi import requests
import pandas as pd


YAHOO_CHART_HOSTS = (
    "https://query1.finance.yahoo.com",
    "https://query2.finance.yahoo.com",
)


def _epoch(value: date) -> int:
    return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())


def parse_chart_payload(payload: dict[str, Any]) -> pd.DataFrame:
    """Convert a Yahoo chart response to unadjusted close and dividend columns."""
    chart = payload.get("chart", {})
    error = chart.get("error")
    if error:
        description = error.get("description") or error.get("code") or "Yahoo API error"
        raise ValueError(str(description))

    results = chart.get("result") or []
    if not results:
        raise ValueError("Yahoo Financeから株価データが返されませんでした。")

    result = results[0]
    timestamps = result.get("timestamp") or []
    quotes = (result.get("indicators", {}).get("quote") or [{}])[0]
    closes = quotes.get("close") or []
    if not timestamps or len(timestamps) != len(closes):
        raise ValueError("Yahoo Financeの株価データ形式を確認できませんでした。")

    index = (
        pd.to_datetime(timestamps, unit="s", utc=True)
        .tz_convert("Asia/Tokyo")
        .tz_localize(None)
        .normalize()
    )
    history = pd.DataFrame({"Close": closes}, index=index)
    history["Close"] = pd.to_numeric(history["Close"], errors="coerce")
    history["Dividends"] = 0.0
    history = history.dropna(subset=["Close"])
    history = history[~history.index.duplicated(keep="last")].sort_index()

    dividend_events = result.get("events", {}).get("dividends", {})
    for event in dividend_events.values():
        amount = pd.to_numeric(event.get("amount"), errors="coerce")
        event_timestamp = event.get("date")
        if pd.isna(amount) or event_timestamp is None:
            continue
        event_date = (
            pd.to_datetime(event_timestamp, unit="s", utc=True)
            .tz_convert("Asia/Tokyo")
            .tz_localize(None)
            .normalize()
        )
        if event_date in history.index:
            history.loc[event_date, "Dividends"] += float(amount)

    return history


def fetch_yahoo_history(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch daily data from Yahoo, retrying a second host when necessary."""
    params = {
        "period1": _epoch(start),
        "period2": _epoch(end),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    errors: list[str] = []

    for host in YAHOO_CHART_HOSTS:
        url = f"{host}/v8/finance/chart/{symbol}"
        verify_modes = (True, False)
        for verify in verify_modes:
            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=20,
                    verify=verify,
                    impersonate="chrome",
                )
                if response.status_code == 429:
                    errors.append(f"{host}: アクセスが一時制限されています")
                    break
                response.raise_for_status()
                return parse_chart_payload(response.json())
            except requests.exceptions.CertificateVerifyError as exc:
                errors.append(f"{host}: 証明書を確認できませんでした")
                if not verify:
                    break
                continue
            except Exception as exc:
                errors.append(f"{host}: {exc}")
                break

    raise RuntimeError(" / ".join(errors) or "Yahoo Financeへ接続できませんでした。")
