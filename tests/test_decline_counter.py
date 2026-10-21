from datetime import time

import pandas as pd

from stocknote_decline_counter import (
    calculate_decline_count,
    completed_daily_history,
    decline_summary,
    signal_event_records,
    strategy_trade_records,
    summarize_signal_events,
)


def frame(closes, highs=None, lows=None, start="2026-01-05"):
    index = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame(
        {
            "Close": closes,
            "High": highs if highs is not None else [v + 1 for v in closes],
            "Low": lows if lows is not None else [v - 1 for v in closes],
        },
        index=index,
    )


def test_count_resets_and_waits_for_next_down_close():
    data = frame([100, 99, 98, 99, 100, 99, 98, 97])
    counted = calculate_decline_count(data, compare_bars=2)
    assert counted["下げ日数"].tolist() == [0, 0, 1, 2, 0, 1, 2, 3]
    assert counted["リセット"].tolist() == [False, False, False, False, True, False, False, False]


def test_every_bar_advances_after_count_starts_until_reset():
    data = frame([100, 99, 98, 99, 98, 97])
    counted = calculate_decline_count(data, compare_bars=2)
    assert counted["下げ日数"].tolist() == [0, 0, 1, 2, 3, 4]


def test_selected_high_price_controls_reset_but_close_starts_count():
    data = frame(
        [100, 99, 98, 97, 96],
        highs=[101, 100, 99, 102, 98],
    )
    counted = calculate_decline_count(data, compare_bars=2, price_column="High")
    assert counted["下げ日数"].tolist() == [0, 0, 1, 0, 1]
    assert bool(counted["リセット"].iloc[3])


def test_signal_and_add_dates_are_recorded():
    closes = [100, 99] + list(range(98, 73, -1))
    counted = calculate_decline_count(frame(closes), signal_count=10, add_count=20)
    summary = decline_summary(counted, signal_count=10, add_count=20)
    assert summary["current_count"] == 25
    assert summary["signal_date"] == counted.index[counted["下げ日数"] == 10][-1]
    assert summary["add_date"] == counted.index[counted["下げ日数"] == 20][-1]


def test_incomplete_today_bar_is_removed_before_market_close():
    data = frame([100, 99, 98], start="2026-10-20")
    before = completed_daily_history(
        data,
        now="2026-10-22 14:00:00+09:00",
        market_close=time(15, 30),
    )
    after = completed_daily_history(
        data,
        now="2026-10-22 16:00:00+09:00",
        market_close=time(15, 30),
    )
    assert len(before) == 2
    assert len(after) == 3


def test_five_day_backtest_and_strategy_add_are_available():
    closes = [110, 109] + list(range(108, 88, -1)) + [90, 92, 94, 96, 98, 100]
    counted = calculate_decline_count(frame(closes), signal_count=10, add_count=20)
    events = signal_event_records(counted, thresholds=(10,), horizon=5)
    summary = summarize_signal_events(events, thresholds=(10,))
    assert int(summary.iloc[0]["シグナル数"]) == 1
    assert float(summary.iloc[0]["5日以内上昇率%"]) == 0.0

    single = strategy_trade_records(counted, signal_count=10, add_count=20, include_add=False)
    added = strategy_trade_records(counted, signal_count=10, add_count=20, include_add=True)
    assert len(single) == len(added) == 1
    assert single[0]["units"] == 1
    assert added[0]["units"] == 2
