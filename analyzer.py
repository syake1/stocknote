"""Core calculations for Japanese ex-dividend recovery analysis."""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


@dataclass(frozen=True)
class AnalysisResult:
    events: pd.DataFrame
    curves: pd.DataFrame


def normalize_tickers(raw: str) -> list[str]:
    """Convert user-entered Japanese stock codes to Yahoo Finance symbols."""
    tokens = re.split(r"[,、，;\s]+", raw)
    symbols: list[str] = []
    for token in tokens:
        value = token.strip().upper()
        if not value:
            continue
        if value.isdigit():
            value = f"{value}.T"
        if value not in symbols:
            symbols.append(value)
    return symbols


def _first_true_offset(mask: pd.Series) -> int | None:
    matches = mask[mask.fillna(False)]
    if matches.empty:
        return None
    return int(mask.index.get_loc(matches.index[0]))


def analyze_history(
    history: pd.DataFrame,
    ticker: str,
    lookback_start: pd.Timestamp,
    horizon: int,
) -> AnalysisResult:
    """Analyze dividend events using trading-session counts.

    Yahoo Finance's Dividends column is treated as the ex-dividend event date.
    Price recovery uses unadjusted close prices so the dividend gap remains visible.
    """
    required = {"Close", "Dividends"}
    missing = required.difference(history.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    data = history.loc[:, ["Close", "Dividends"]].copy()
    data.index = pd.to_datetime(data.index)
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)
    data = data.sort_index()
    data = data[~data.index.duplicated(keep="last")]
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data["Dividends"] = pd.to_numeric(data["Dividends"], errors="coerce").fillna(0.0)
    data = data.dropna(subset=["Close"])

    start = pd.Timestamp(lookback_start).tz_localize(None)
    dividend_dates = data.index[(data["Dividends"] > 0) & (data.index >= start)]
    event_rows: list[dict[str, object]] = []
    curve_rows: list[dict[str, object]] = []

    for ex_date in dividend_dates:
        position = data.index.get_loc(ex_date)
        if not isinstance(position, int) or position == 0:
            continue

        pre_date = data.index[position - 1]
        pre_close = float(data.iloc[position - 1]["Close"])
        dividend = float(data.iloc[position]["Dividends"])
        window = data.iloc[position : position + horizon + 1].copy()
        if window.empty or pre_close <= 0:
            continue

        adjusted_target = pre_close - dividend
        recovery_days = _first_true_offset(window["Close"] >= pre_close)
        adjusted_recovery_days = _first_true_offset(window["Close"] >= adjusted_target)

        rebound_days: int | None = None
        for offset in range(1, len(window)):
            current_close = float(window.iloc[offset]["Close"])
            previous_close = float(window.iloc[offset - 1]["Close"])
            if current_close > previous_close:
                rebound_days = offset
                break

        ex_close = float(window.iloc[0]["Close"])
        rebound_date = (
            window.index[rebound_days] if rebound_days is not None else pd.NaT
        )
        recovery_date = (
            window.index[recovery_days] if recovery_days is not None else pd.NaT
        )
        bottom_window = (
            window.iloc[: recovery_days + 1]
            if recovery_days is not None
            else window
        )
        bottom_days = int(bottom_window["Close"].to_numpy().argmin())
        bottom_date = bottom_window.index[bottom_days]
        event_rows.append(
            {
                "ticker": ticker,
                "ex_date": ex_date,
                "pre_date": pre_date,
                "dividend": dividend,
                "pre_close": pre_close,
                "ex_close": ex_close,
                "gap_pct": ex_close / pre_close - 1,
                "rebound_days": rebound_days,
                "rebound_date": rebound_date,
                "recovery_days": recovery_days,
                "recovery_date": recovery_date,
                "bottom_days": bottom_days,
                "bottom_date": bottom_date,
                "adjusted_recovery_days": adjusted_recovery_days,
                "recovered": recovery_days is not None,
                "adjusted_recovered": adjusted_recovery_days is not None,
                "max_drawdown_pct": float(window["Close"].min() / pre_close - 1),
                "observed_days": len(window) - 1,
            }
        )

        for day, (date, row) in enumerate(window.iterrows()):
            curve_rows.append(
                {
                    "ticker": ticker,
                    "ex_date": ex_date,
                    "day": day,
                    "date": date,
                    "return_pct": float(row["Close"] / pre_close - 1),
                    "close": float(row["Close"]),
                    "pre_close": pre_close,
                    "adjusted_target": adjusted_target,
                }
            )

    events = pd.DataFrame(event_rows)
    curves = pd.DataFrame(curve_rows)
    return AnalysisResult(events=events, curves=curves)
