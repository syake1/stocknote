import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from monitor_candidates import is_market_session
from stocknote_technicals import calculate, daily_trend_context, quarterly_strength_context


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

    def test_clean_quarterly_uptrend_qualifies_even_with_high_rsi(self):
        index = pd.date_range("2021-01-01", "2026-06-30", freq="B")
        base = np.linspace(100, 240, len(index))
        close = pd.Series(base + np.sin(np.arange(len(index)) / 30) * 2, index=index)
        frame = pd.DataFrame({
            "Open": close * 0.98, "High": close * 1.02,
            "Low": close * 0.97, "Close": close,
            "Volume": np.linspace(1000, 2000, len(index)),
        }, index=index)
        result = quarterly_strength_context(frame, now="2026-08-15")
        self.assertTrue(result["quarterly_qualified"])
        self.assertGreaterEqual(result["quarterly_score"], 70)
        self.assertGreaterEqual(result["quarterly_rsi"], 70)
        self.assertIn("RSIは高い", result["quarterly_reason"])

    def test_falling_quarterly_structure_is_rejected(self):
        index = pd.date_range("2021-01-01", "2026-06-30", freq="B")
        close = pd.Series(np.linspace(240, 100, len(index)), index=index)
        frame = pd.DataFrame({
            "Open": close * 1.02, "High": close * 1.03,
            "Low": close * 0.98, "Close": close,
        }, index=index)
        result = quarterly_strength_context(frame, now="2026-08-15")
        self.assertFalse(result["quarterly_qualified"])
        self.assertLess(result["quarterly_score"], 70)

    def test_high_level_quarterly_consolidation_qualifies(self):
        index = pd.date_range("2021-01-01", "2026-06-30", freq="B")
        rise_end = int(len(index) * 0.72)
        rising = np.linspace(100, 205, rise_end)
        sideways = 202 + np.sin(np.arange(len(index) - rise_end) / 25) * 3
        close = pd.Series(np.concatenate([rising, sideways]), index=index)
        frame = pd.DataFrame({
            "Open": close * 0.995, "High": close * 1.012,
            "Low": close * 0.988, "Close": close,
            "Volume": np.linspace(1000, 1800, len(index)),
        }, index=index)
        result = quarterly_strength_context(frame, now="2026-08-15")
        self.assertTrue(result["quarterly_qualified"])
        self.assertTrue(result["quarterly_high_level_consolidation"])
        self.assertEqual(result["quarterly_pattern"], "高値圏持ち合い")
        self.assertIn("上放れ待ち", result["quarterly_reason"])

    def test_low_level_sideways_stock_is_rejected(self):
        index = pd.date_range("2021-01-01", "2026-06-30", freq="B")
        falling = np.linspace(230, 120, int(len(index) * 0.72))
        sideways = 122 + np.sin(np.arange(len(index) - len(falling)) / 25) * 2
        close = pd.Series(np.concatenate([falling, sideways]), index=index)
        frame = pd.DataFrame({
            "Open": close * 1.002, "High": close * 1.012,
            "Low": close * 0.988, "Close": close,
        }, index=index)
        result = quarterly_strength_context(frame, now="2026-08-15")
        self.assertFalse(result["quarterly_qualified"])
        self.assertFalse(result["quarterly_high_level_consolidation"])

    def test_strong_trend_with_large_quarterly_wick_is_watch_only(self):
        index = pd.date_range("2021-01-01", "2026-06-30", freq="B")
        close = pd.Series(np.linspace(100, 240, len(index)), index=index)
        frame = pd.DataFrame({
            "Open": close * 0.98, "High": close * 1.02,
            "Low": close * 0.97, "Close": close,
        }, index=index)
        last_quarter = frame.index.to_period("Q") == pd.Period("2026Q2")
        frame.loc[last_quarter, "High"] = frame.loc[last_quarter, "Close"].max() * 1.25
        result = quarterly_strength_context(frame, now="2026-08-15")
        self.assertTrue(result["quarterly_large_upper_wick"])
        self.assertFalse(result["quarterly_qualified"])
        self.assertIn("上ヒゲ", result["quarterly_reason"])


if __name__ == "__main__":
    unittest.main()
