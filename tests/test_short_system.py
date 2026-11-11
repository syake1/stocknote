import unittest

import numpy as np
import pandas as pd

from stocknote_short import (
    assess_short,
    credit_score,
    short_fundamental_score,
    short_technical_snapshot,
    short_watch_reason,
)


class ShortSystemTests(unittest.TestCase):
    def test_unknown_credit_never_becomes_actionable(self):
        result = assess_short(
            {"空売りスコア": 95, "空売りトレンド適合": True},
            {"per": 40, "pbr": 5, "roe": -0.1, "opm": -0.05, "growth": -0.1},
            {}, 20,
        )
        self.assertEqual(result["空売り状態"], "SBI信用データ確認待ち")
        self.assertIsNone(result["信用倍率"])

    def test_confirmed_loanable_candidate_can_reach_signal(self):
        result = assess_short(
            {"空売りスコア": 95, "空売りトレンド適合": True},
            {"per": 40, "pbr": 5, "roe": -0.1, "opm": -0.05, "growth": -0.1},
            {"信用倍率": "5.2倍", "貸借区分": "貸借銘柄"}, 20,
        )
        self.assertEqual(result["空売り状態"], "空売り条件到達")
        self.assertEqual(result["信用倍率"], 5.2)

    def test_crowded_short_is_flagged(self):
        result = assess_short(
            {"空売りスコア": 90, "空売りトレンド適合": True},
            {"per": 40}, {"信用倍率": "0.5", "売建可否": "売建可"}, 25,
        )
        self.assertEqual(result["空売り状態"], "踏み上げ・逆日歩注意")
        self.assertEqual(credit_score(.5), 15.0)

    def test_strong_fundamentals_reduce_short_score(self):
        weak, _, _ = short_fundamental_score({"per": 40, "roe": -.1, "growth": -.1})
        strong, _, _ = short_fundamental_score({"per": 12, "roe": .15, "growth": .1})
        self.assertGreater(weak, strong)

    def test_early_short_watch_does_not_require_full_long_term_downtrend(self):
        reason = short_watch_reason({
            "現在値": 900,
            "MA5": 920,
            "MA25": 950,
            "MA75": 880,
            "RSI14": 58,
            "空売りスコア": 46,
            "空売りトレンド適合": False,
            "転換線下抜け": True,
            "包み陰線": False,
            "上ヒゲ陰線": False,
        })
        self.assertIn("5日・25日線の下", reason)
        self.assertIn("転換線下抜け", reason)

    def test_short_watch_ignores_healthy_price_above_averages(self):
        reason = short_watch_reason({
            "現在値": 1100,
            "MA5": 1050,
            "MA25": 1000,
            "MA75": 950,
            "RSI14": 55,
            "空売りスコア": 30,
            "空売りトレンド適合": False,
        })
        self.assertIsNone(reason)

    def test_daily_snapshot_produces_short_watch_fields(self):
        index = pd.date_range("2025-01-01", periods=260, freq="B")
        close = np.concatenate([
            np.linspace(150, 105, 250),
            np.array([108, 107, 106, 105, 104, 103, 102, 101, 100, 98]),
        ])
        frame = pd.DataFrame({
            "Open": close + 1.0,
            "High": close + 2.0,
            "Low": close - 2.0,
            "Close": close,
            "Volume": np.full(len(close), 100000),
        }, index=index)
        result = short_technical_snapshot("1234", "テスト", frame)
        self.assertIsNotNone(result)
        self.assertIn("空売りスコア", result)
        self.assertIn("空売りトレンド適合", result)
        self.assertLess(result["現在値"], result["MA25"])


if __name__ == "__main__":
    unittest.main()
