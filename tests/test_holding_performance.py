from stocknote_tracking import holding_performance


def test_buy_position_loss_is_negative():
    result = holding_performance(1000, 900, 100, "買い")
    assert result == {
        "per_share": -100.0,
        "pnl": -10000.0,
        "pnl_pct": -10.0,
        "loss": -10000.0,
        "profit": 0.0,
    }


def test_buy_position_profit_is_positive():
    result = holding_performance(1000, 1100, 100, "買い")
    assert result["pnl"] == 10000.0
    assert result["loss"] == 0.0
    assert result["profit"] == 10000.0


def test_short_position_uses_inverse_price_move():
    result = holding_performance(1000, 900, 100, "空売り")
    assert result["per_share"] == 100.0
    assert result["pnl"] == 10000.0
    assert result["pnl_pct"] == 10.0


def test_missing_current_price_is_not_guessed():
    assert holding_performance(1000, None, 100, "買い") is None
