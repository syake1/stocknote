import unittest

import pandas as pd

from analyzer import analyze_history, normalize_tickers


class AnalyzerTests(unittest.TestCase):
    def test_normalize_tickers(self):
        self.assertEqual(
            normalize_tickers("7203、8306\n7203　9432.T"),
            ["7203.T", "8306.T", "9432.T"],
        )

    def test_rebound_and_recovery_days(self):
        dates = pd.bdate_range("2025-03-26", periods=7)
        history = pd.DataFrame(
            {
                "Close": [1000, 960, 950, 970, 990, 1000, 1010],
                "Dividends": [0, 30, 0, 0, 0, 0, 0],
            },
            index=dates,
        )

        result = analyze_history(history, "TEST.T", pd.Timestamp("2025-01-01"), 10)
        event = result.events.iloc[0]

        self.assertEqual(event["rebound_days"], 2)
        self.assertEqual(event["rebound_date"], dates[3])
        self.assertEqual(event["recovery_days"], 4)
        self.assertEqual(event["recovery_date"], dates[5])
        self.assertEqual(event["bottom_days"], 1)
        self.assertEqual(event["bottom_date"], dates[2])
        self.assertEqual(event["adjusted_recovery_days"], 2)
        self.assertAlmostEqual(event["gap_pct"], -0.04)

    def test_marks_event_unrecovered_with_short_window(self):
        dates = pd.bdate_range("2025-09-25", periods=4)
        history = pd.DataFrame(
            {
                "Close": [1000, 930, 920, 940],
                "Dividends": [0, 20, 0, 0],
            },
            index=dates,
        )

        result = analyze_history(history, "TEST.T", pd.Timestamp("2025-01-01"), 2)
        event = result.events.iloc[0]

        self.assertFalse(event["recovered"])
        self.assertTrue(pd.isna(event["recovery_days"]))
        self.assertTrue(pd.isna(event["recovery_date"]))
        self.assertEqual(event["bottom_days"], 1)
        self.assertEqual(event["observed_days"], 2)


if __name__ == "__main__":
    unittest.main()
