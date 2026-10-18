from datetime import date
from stocknote_candidate_view import candidate_day, candidate_row, daily_chart_url, price_change, signed_change


def test_price_changes_use_detection_price_and_preserve_decimal_yen():
    row = candidate_row({"code": "1234", "first_price": 1000, "current_price": 1052.5, "return_pct": 999})
    assert row["値上がり・値下がり"] == "+52.5円"
    assert row["上昇・下落（%）"] == "+5.25%"
    assert "騰落率%" not in row and "BB位置σ" not in row


def test_falling_and_unchanged_prices():
    amount, percent = price_change({"first_price": 200, "current_price": 190})
    assert signed_change(amount) == "-10円"
    assert signed_change(percent, percent=True) == "-5.00%"
    assert signed_change(0) == "0円"
    assert signed_change(-0.0001, percent=True) == "0.00%"


def test_missing_or_invalid_prices_are_not_reported_as_zero():
    for first, current in [(None, 100), (0, 100), (100, None), (100, float("nan")), (100, -1)]:
        row = candidate_row({"first_price": first, "current_price": current})
        assert row["値上がり・値下がり"] == "—"
        assert row["上昇・下落（%）"] == "—"


def test_chart_links_support_alphanumeric_codes_and_daily_candles():
    assert daily_chart_url("135a.T") == "https://finance.yahoo.co.jp/quote/135A.T/chart?styl=cndl&frm=dly"
    assert daily_chart_url(None) is None
    assert daily_chart_url("bad/url") is None


def test_candidate_day_starts_at_one_and_uses_current_day_not_stale_elapsed():
    candidate = {"first_seen_at": "2026-10-16T17:31:41+09:00", "elapsed_days": 0}
    assert candidate_day(candidate, today=date(2026, 10, 16)) == 1
    assert candidate_day(candidate, today=date(2026, 10, 19)) == 4
    assert candidate_day({"elapsed_days": 2}, today=date(2026, 10, 19)) == 3
    assert candidate_day({}, today=date(2026, 10, 19)) is None
