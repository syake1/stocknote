import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from monitor_candidates import is_market_session
from stocknote_technicals import calculate, daily_trend_context


class MonitorAndTechnicalTests(unittest.TestCase):
    def test_market_sessions_and_break(self):
        jst = ZoneInfo("Asia/Tokyo")
        self.assertTrue(is_market_session(datetime(2026, 8, 27, 9, 15, tzinfo=jst)))
        self.assertFalse(is_market_session(datetime(2026, 8, 27, 12, 0, tzinfo=jst)))
        self.assertTrue(is_market_session(datetime(2026, 8, 27, 14, 45, tzinfo=jst)))
        self.assertFalse(is_market_session(datetime(2026, 8, 29, 10, 0, tzinfo=jst)))

    def test_real_columns_produce_all_requested_metrics(self):
        index = pd.date_range("2026-01-01", periods=220, freq="D")
        close = pd.Series(np.linspace(100, 130, 220) + np.sin(np.arange(220)), index=index)
        frame = pd.DataFrame({
            "Open": close - .2, "High": close + 1, "Low": close - 1,
            "Close": close, "Volume": np.linspace(1000, 1800, 220),
        }, index=index)
        row = calculate("1234", "テスト", frame)
        for key in ("price", "rsi", "bb_position", "ma5", "ma25", "ma75",
                    "ma200", "macd", "macd_signal", "volume", "volume_ratio",
                    "psar", "atr", "score"):
            self.assertIn(key, row)
            self.assertIsNotNone(row[key])

    def test_ichimoku_gate_accepts_uptrend_above_cloud(self):
        index = pd.date_range("2025-01-01", periods=260, freq="D")
        close = pd.Series(np.linspace(100, 180, 260), index=index)
        frame = pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close}, index=index)
        trend = daily_trend_context(frame)
        self.assertEqual(trend["cloud_position"], "雲の上")
        self.assertTrue(trend["buy_eligible"])

    def test_ichimoku_gate_rejects_falling_stock(self):
        index = pd.date_range("2025-01-01", periods=260, freq="D")
        close = pd.Series(np.linspace(180, 90, 260), index=index)
        frame = pd.DataFrame({"High": close + 1, "Low": close - 1, "Close": close}, index=index)
        trend = daily_trend_context(frame)
        self.assertFalse(trend["buy_eligible"])
        self.assertTrue(trend["cloud_position"] == "雲の下" or not trend["ma75_up"])


if __name__ == "__main__":
    unittest.main()
