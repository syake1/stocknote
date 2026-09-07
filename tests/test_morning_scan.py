import pandas as pd

import morning_scan


def test_ma75_touch_is_required_within_recent_three_days():
    ma75 = pd.Series([100.0, 100.0, 100.0, 100.0])
    assert morning_scan.touched_ma75_recently(
        pd.Series([110.0, 101.0, 99.0, 102.0]), ma75
    )
    assert not morning_scan.touched_ma75_recently(
        pd.Series([99.0, 106.0, 105.0, 104.0]), ma75
    )


def test_contrarian_score_remains_primary_after_fundamental_check():
    row = {"score": 90.0, "code": "1234"}
    result = morning_scan.add_fundamental_judgement(
        row, {"score": 50.0, "available": 5, "comment": "標準"}
    )
    assert result["technical_score"] == 90.0
    assert result["fundamental_score"] == 50.0
    assert result["score"] == 82.0


def test_materially_weak_fundamentals_are_rejected():
    row = {"score": 95.0, "code": "1234"}
    assert morning_scan.add_fundamental_judgement(
        row, {"score": 30.0, "available": 5, "comment": "弱い"}
    ) is None


def test_missing_fundamentals_do_not_erase_a_valid_technical_signal():
    row = {"score": 80.0, "code": "1234"}
    result = morning_scan.add_fundamental_judgement(
        row, {"score": 50.0, "available": 0, "comment": "取得不可"}
    )
    assert result is not None
    assert result["score"] == 74.0
