"""Daily-bar decline counter and backtest helpers for Stocknote."""

from __future__ import annotations

from datetime import time
from typing import Iterable
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


PRICE_LABELS = {
    "終値": "Close",
    "高値": "High",
    "安値": "Low",
}

# 数え始めの起点モード
START_MODE_RESET = "reset"        # 上がったあと最初に出た陰線から数える(従来)
START_MODE_BB_UPPER = "bb_upper"  # BB上限タッチ後の最初の陰線から数える
START_MODE_LABELS = {
    "上がったあと最初の陰線から(従来)": START_MODE_RESET,
    "BB上限タッチ後の最初の陰線から": START_MODE_BB_UPPER,
}


def completed_daily_history(
    history: pd.DataFrame,
    now=None,
    market_close: time = time(15, 30),
) -> pd.DataFrame:
    """Return sorted daily data without a still-forming Tokyo trading-day bar."""
    if history is None or history.empty:
        return pd.DataFrame()

    data = history.copy()
    index = pd.to_datetime(data.index, errors="coerce")
    valid = ~pd.isna(index)
    data = data.loc[valid].copy()
    data.index = index[valid]
    data = data[~data.index.duplicated(keep="last")].sort_index()
    if data.empty:
        return data

    jst = ZoneInfo("Asia/Tokyo")
    current = pd.Timestamp.now(tz=jst) if now is None else pd.Timestamp(now)
    if current.tzinfo is None:
        current = current.tz_localize(jst)
    else:
        current = current.tz_convert(jst)

    if data.index.tz is None:
        bar_dates = pd.Series(data.index.date, index=data.index)
    else:
        bar_dates = pd.Series(data.index.tz_convert(jst).date, index=data.index)

    data = data.loc[bar_dates <= current.date()].copy()
    if data.empty:
        return data
    bar_dates = bar_dates.loc[data.index]
    if current.time() < market_close:
        data = data.loc[bar_dates < current.date()].copy()
    return data


def bollinger_upper_touch(
    data: pd.DataFrame,
    window: int = 20,
    sigma: float = 2.0,
) -> tuple[pd.Series, pd.Series]:
    """Return (upper band, touched flag). Touch means High (or Close) >= upper band."""
    close = pd.to_numeric(data["Close"], errors="coerce")
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std(ddof=0)
    upper = mean + sigma * std
    if "High" in data.columns:
        high = pd.to_numeric(data["High"], errors="coerce")
    else:
        high = close
    touched = (high >= upper) & upper.notna()
    return upper, touched


def calculate_decline_count(
    history: pd.DataFrame,
    signal_count: int = 10,
    add_count: int = 20,
    compare_bars: int = 2,
    price_column: str = "Close",
    start_mode: str = START_MODE_RESET,
    bb_window: int = 20,
    bb_sigma: float = 2.0,
) -> pd.DataFrame:
    """Calculate the reset-aware decline count on completed daily bars.

    A reset (selected price above the selected price ``compare_bars`` ago)
    always takes precedence while counting.

    start_mode == "reset" (default):
        After a reset, the first close below the previous close starts a new
        count at one.

    start_mode == "bb_upper":
        The count starts only after the Bollinger upper band has been touched.
        The first close below the previous close on or after the touch day
        starts the count at one. A reset clears the count and requires a new
        upper-band touch before counting again.

    While counting, every completed bar adds one until the next reset.
    """
    if signal_count < 1:
        raise ValueError("買いシグナルの本数は1以上にしてください")
    if add_count < 1:
        raise ValueError("追加買いの本数は1以上にしてください")
    if compare_bars < 1:
        raise ValueError("比較する足の本数は1以上にしてください")
    if start_mode not in (START_MODE_RESET, START_MODE_BB_UPPER):
        raise ValueError("数え始めのモードが正しくありません")
    if history is None or history.empty:
        return pd.DataFrame()
    if "Close" not in history.columns:
        raise ValueError("終値（Close）列がありません")
    if price_column not in history.columns:
        raise ValueError(f"判定価格の列（{price_column}）がありません")

    data = history.copy()
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data[price_column] = pd.to_numeric(data[price_column], errors="coerce")
    data = data.dropna(subset=["Close", price_column]).sort_index()
    if data.empty:
        return data

    close = data["Close"]
    comparison_price = data[price_column]

    use_bb = start_mode == START_MODE_BB_UPPER
    if use_bb:
        upper, touched_series = bollinger_upper_touch(data, bb_window, bb_sigma)
        data["BB上限"] = upper
        data["BB上限タッチ"] = touched_series
        touched_values = touched_series.to_numpy()
    else:
        touched_values = None

    counts: list[int] = []
    resets: list[bool] = []
    restarts: list[bool] = []
    count = 0
    armed = False  # bb_upper モード: タッチ済みで陰線待ち

    for position in range(len(data)):
        reset = False
        restart = False
        touched = bool(touched_values[position]) if use_bb else False

        if position >= compare_bars:
            reset = bool(
                comparison_price.iloc[position]
                > comparison_price.iloc[position - compare_bars]
            )

            if use_bb:
                if count > 0:
                    if reset:
                        count = 0
                        armed = touched
                    else:
                        count += 1
                else:
                    if touched:
                        armed = True
                    if armed and close.iloc[position] < close.iloc[position - 1]:
                        count = 1
                        restart = True
                        armed = False
            else:
                if reset:
                    count = 0
                elif count > 0:
                    count += 1
                elif close.iloc[position] < close.iloc[position - 1]:
                    count = 1
                    restart = True
        elif use_bb and touched:
            armed = True

        counts.append(count)
        resets.append(reset)
        restarts.append(restart)

    data["下げ日数"] = pd.Series(counts, index=data.index, dtype="int64")
    data["リセット"] = pd.Series(resets, index=data.index, dtype="bool")
    data["数え直し開始"] = pd.Series(restarts, index=data.index, dtype="bool")
    data["買い到達"] = data["下げ日数"].eq(int(signal_count))
    data["追加買い到達"] = data["下げ日数"].eq(int(add_count))
    return data


def _latest_event_date(counted: pd.DataFrame, count: int):
    matches = counted.index[counted["下げ日数"].eq(int(count))]
    return pd.Timestamp(matches[-1]) if len(matches) else None


def decline_summary(counted: pd.DataFrame, signal_count: int = 10, add_count: int = 20) -> dict:
    if counted is None or counted.empty:
        return {
            "current_count": 0,
            "asof": None,
            "signal_date": None,
            "add_date": None,
            "signal_today": False,
            "add_today": False,
        }
    latest = counted.iloc[-1]
    current_count = int(latest["下げ日数"])
    return {
        "current_count": current_count,
        "asof": pd.Timestamp(counted.index[-1]),
        "signal_date": _latest_event_date(counted, signal_count),
        "add_date": _latest_event_date(counted, add_count),
        "signal_today": current_count == int(signal_count),
        "add_today": current_count == int(add_count),
    }


def signal_event_records(
    counted: pd.DataFrame,
    thresholds: Iterable[int] = (8, 10, 12, 15),
    horizon: int = 5,
) -> list[dict]:
    """Return event-level five-day outcomes for each requested threshold."""
    if counted is None or counted.empty or horizon < 1:
        return []
    close = pd.to_numeric(counted["Close"], errors="coerce")
    records = []
    for threshold in thresholds:
        positions = np.flatnonzero(counted["下げ日数"].to_numpy() == int(threshold))
        for position in positions:
            if position + horizon >= len(counted):
                continue
            entry = float(close.iloc[position])
            future = close.iloc[position + 1:position + horizon + 1].dropna()
            if entry <= 0 or len(future) < horizon:
                continue
            path_returns = future / entry - 1.0
            records.append({
                "threshold": int(threshold),
                "signal_date": pd.Timestamp(counted.index[position]),
                "entry_price": entry,
                "within_horizon_up": bool((future > entry).any()),
                "final_return_pct": float(path_returns.iloc[-1] * 100.0),
                "max_drawdown_pct": float(min(0.0, path_returns.min()) * 100.0),
            })
    return records


def summarize_signal_events(records: Iterable[dict], thresholds=(8, 10, 12, 15)) -> pd.DataFrame:
    rows = []
    records = list(records)
    for threshold in thresholds:
        selected = [r for r in records if int(r["threshold"]) == int(threshold)]
        returns = [float(r["final_return_pct"]) for r in selected]
        drawdowns = [float(r["max_drawdown_pct"]) for r in selected]
        rows.append({
            "本数": int(threshold),
            "シグナル数": len(selected),
            "5日以内上昇率%": (
                float(np.mean([r["within_horizon_up"] for r in selected]) * 100.0)
                if selected else np.nan
            ),
            "勝率%": float(np.mean(np.array(returns) > 0) * 100.0) if returns else np.nan,
            "平均損益%": float(np.mean(returns)) if returns else np.nan,
            "最大ドローダウン%": float(np.min(drawdowns)) if drawdowns else np.nan,
        })
    return pd.DataFrame(rows)


def strategy_trade_records(
    counted: pd.DataFrame,
    signal_count: int = 10,
    add_count: int = 20,
    include_add: bool = False,
) -> list[dict]:
    """Buy at the signal count and exit on reset; optionally add equal size."""
    if counted is None or counted.empty:
        return []

    close = pd.to_numeric(counted["Close"], errors="coerce")
    counts = counted["下げ日数"].astype(int)
    trades = []
    position = 0
    while position < len(counted):
        if counts.iloc[position] != int(signal_count) or pd.isna(close.iloc[position]):
            position += 1
            continue

        entry_position = position
        entry_price = float(close.iloc[position])
        costs = [entry_price]
        add_date = None
        path_returns = [0.0]
        position += 1

        while position < len(counted):
            price = float(close.iloc[position])
            count = int(counts.iloc[position])
            if include_add and add_count > signal_count and count == int(add_count):
                costs.append(price)
                add_date = pd.Timestamp(counted.index[position])
            average_cost = float(np.mean(costs))
            path_returns.append(price / average_cost - 1.0)
            if count == 0:
                break
            position += 1

        exit_position = min(position, len(counted) - 1)
        exit_price = float(close.iloc[exit_position])
        average_cost = float(np.mean(costs))
        trades.append({
            "entry_date": pd.Timestamp(counted.index[entry_position]),
            "exit_date": pd.Timestamp(counted.index[exit_position]),
            "add_date": add_date,
            "units": len(costs),
            "return_pct": float((exit_price / average_cost - 1.0) * 100.0),
            "max_drawdown_pct": float(min(path_returns) * 100.0),
        })
        position = max(position + 1, entry_position + 1)
    return trades


def summarize_strategy_trades(records: Iterable[dict], label: str) -> dict:
    records = list(records)
    returns = [float(r["return_pct"]) for r in records]
    drawdowns = [float(r["max_drawdown_pct"]) for r in records]
    return {
        "方式": label,
        "取引数": len(records),
        "勝率%": float(np.mean(np.array(returns) > 0) * 100.0) if returns else np.nan,
        "平均損益%": float(np.mean(returns)) if returns else np.nan,
        "最大ドローダウン%": float(np.min(drawdowns)) if drawdowns else np.nan,
    }
