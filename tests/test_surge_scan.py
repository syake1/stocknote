import numpy as np
import pandas as pd

from stocknote_surge import (
    historical_market_order_events,
    market_order_decision,
    parse_jpx_universe,
    rank_surge_candidates,
    summarize_market_order_backtest,
    surge_snapshot,
    verified_morning_threshold,
)


def history_with_surge(periods=100):
    index = pd.bdate_range("2026-01-05", periods=periods)
    base = np.linspace(900.0, 1000.0, periods)
    close = pd.Series(base, index=index)
    close.iloc[-6:-1] = [982.0, 984.0, 986.0, 987.0, 988.0]
    close.iloc[-1] = 1005.0
    open_ = close.shift(1).fillna(close.iloc[0])
    high = np.maximum(open_.to_numpy(), close.to_numpy()) + 8.0
    low = np.minimum(open_.to_numpy(), close.to_numpy()) - 5.0
    volume = np.full(periods, 100_000.0)
    volume[-1] = 260_000.0
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=index,
    )


def test_parse_jpx_universe_keeps_domestic_operating_markets():
    source = pd.DataFrame(
        {
            "コード": ["130A", "7203", "1306", "94325"],
            "銘柄名": ["成長社", "トヨタ", "ETF", "種類株"],
            "市場・商品区分": [
                "グロース（内国株式）",
                "プライム（内国株式）",
                "ETF・ETN",
                "プライム（内国株式）",
            ],
            "33業種区分": ["情報・通信業", "輸送用機器", "-", "情報・通信業"],
        }
    )
    result = parse_jpx_universe(source)
    assert result["コード"].tolist() == ["130A", "7203"]


def test_surge_snapshot_identifies_liquid_initial_move():
    snapshot = surge_snapshot(
        "130A",
        "成長社",
        "グロース（内国株式）",
        "情報・通信業",
        history_with_surge(),
        now="2026-06-01 18:00+09:00",
    )
    assert snapshot is not None
    assert snapshot["初動条件"] is True
    assert snapshot["出来高倍率"] >= 2.0
    assert "出来高" in snapshot["理由"]


def test_rank_requires_verified_backtest_for_morning_label():
    snapshot = surge_snapshot(
        "130A", "成長社", "グロース（内国株式）", "情報・通信業",
        history_with_surge(), now="2026-06-01 18:00+09:00",
    )
    snapshot["基礎点"] = 80.0
    unverified = rank_surge_candidates([snapshot], verified_threshold=None)
    verified = rank_surge_candidates([snapshot], verified_threshold=70)
    assert unverified.iloc[0]["判定"] == "監視"
    assert verified.iloc[0]["判定"] == "朝成候補"


def test_market_order_gap_guard():
    assert market_order_decision(102.9, 100.0)[0] == "成行条件内"
    assert market_order_decision(103.0, 100.0)[0] == "成行見送り"
    assert market_order_decision(94.0, 100.0)[0] == "急落気配のため成行見送り"


def test_backtest_threshold_needs_sample_and_edge():
    winning = [
        {
            "検証点": 75.0,
            "5日損益%": 2.0 if index % 10 < 6 else -1.0,
            "5日以内最大上昇%": 6.0 if index % 4 == 0 else 3.0,
            "5日最大DD%": -2.0,
        }
        for index in range(120)
    ]
    summary = summarize_market_order_backtest(winning)
    assert verified_morning_threshold(summary) == 75

    weak = pd.DataFrame([{
        "基準点": 75,
        "件数": 120,
        "5日勝率%": 50.0,
        "5日平均損益%": -0.1,
        "5日以内+5%到達率": 10.0,
    }])
    assert verified_morning_threshold(weak) is None


def test_historical_events_use_next_session_open():
    history = history_with_surge(140)
    history.iloc[-20:, history.columns.get_loc("Volume")] = 250_000.0
    history.iloc[-1, history.columns.get_loc("Volume")] = 500_000.0
    events = historical_market_order_events("130A", history, now="2027-01-01 18:00+09:00")
    for event in events:
        signal_index = history.index.get_loc(pd.Timestamp(event["シグナル日"]))
        assert event["成行価格"] == history.iloc[signal_index + 1]["Open"]
