from stocknote_holding_analysis import combined_buy_score


def test_combined_buy_score_uses_scanner_weights():
    assert combined_buy_score(80, 70, 60) == 73.5


def test_combined_buy_score_is_bounded():
    assert combined_buy_score(200, 200, 200) == 100.0
    assert combined_buy_score(-10, -10, -10) == 0.0
