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
            {
                "空売りスコア": 95,
                "空売りトレンド適合": True,
                "75日線乖離%": 8.0,
                "空売り反落確認": True,
            },
            {"per": 40, "pbr": 5, "roe": -0.1, "opm": -0.05, "growth": -0.1},
            {}, 20,
        )
        self.assertEqual(result["空売り状態"], "SBI信用データ確認待ち")
        self.assertIsNone(result["信用倍率"])

    def test_confirmed_loanable_candidate_can_reach_signal(self):
        result = assess_short(
            {
                "空売りスコア": 95,
                "空売りトレンド適合": True,
                "75日線乖離%": 8.0,
                "空売り反落確認": True,
            },
            {"per": 40, "pbr": 5, "roe": -0.1, "opm": -0.05, "growth": -0.1},
            {"信用倍率": "5.2倍", "貸借区分": "貸借銘柄"}, 20,
        )
        self.assertEqual(result["空売り状態"], "空売り条件到達")
        self.assertEqual(result["信用倍率"], 5.2)

    def test_crowded_short_is_flagged(self):
        result = assess_short(
            {
                "空売りスコア": 90,
                "空売りトレンド適合": True,
                "75日線乖離%": 8.0,
                "空売り反落確認": True,
            },
            {"per": 40}, {"信用倍率": "0.5", "売建可否": "売建可"}, 25,
        )
        self.assertEqual(result["空売り状態"], "踏み上げ・逆日歩注意")
        self.assertEqual(credit_score(.5), 15.0)

    def test_strong_fundamentals_reduce_short_score(self):
        weak, _, _ = short_fundamental_score({"per": 40, "roe": -.1, "growth": -.1})
        strong, _, _ = short_fundamental_score({"per": 12, "roe": .15, "growth": .1})
        self.assertGreater(weak, strong)

    def test_short_watch_requires_ma75_overextension(self):
        reason = short_watch_reason({
            "現在値": 980,
            "MA5": 1000,
            "MA25": 1010,
            "MA75": 900,
            "RSI14": 65,
            "空売りスコア": 46,
            "空売りトレンド適合": True,
            "転換線下抜け": True,
            "包み陰線": False,
            "上ヒゲ陰線": False,
        })
        self.assertIn("75日線から+8.9%上方乖離", reason)
        self.assertIn("5日・25日線の下", reason)
        self.assertIn("転換線下抜け", reason)

    def test_short_watch_excludes_small_ma75_distance(self):
        reason = short_watch_reason({
            "現在値": 980,
            "MA5": 1000,
            "MA25": 990,
            "MA75": 950,
            "RSI14": 65,
            "空売りスコア": 90,
            "空売りトレンド適合": False,
            "転換線下抜け": True,
        })
        self.assertIsNone(reason)

    def test_overextended_stock_without_reversal_is_watch_only(self):
        technical = {
            "現在値": 1100,
            "MA5": 1050,
            "MA25": 1000,
            "MA75": 950,
            "RSI14": 68,
            "空売りスコア": 90,
            "75日線乖離%": 15.8,
            "空売り反落確認": False,
        }
        reason = short_watch_reason(technical)
        result = assess_short(
            technical,
            {"per": 40},
            {"信用倍率": "5.2倍", "貸借区分": "貸借銘柄"},
            20,
        )
        self.assertIn("反落待ち", reason)
        self.assertEqual(result["空売り状態"], "反落確認待ち")

    def test_high_score_below_ma75_distance_gate_is_not_actionable(self):
        result = assess_short(
            {"空売りスコア": 99, "75日線乖離%": 4.9, "空売り反落確認": True},
            {"per": 40, "pbr": 5, "roe": -0.1},
            {"信用倍率": "5.2倍", "貸借区分": "貸借銘柄"},
            20,
        )
        self.assertEqual(result["空売り状態"], "75日線上方乖離不足")

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
        self.assertIn("75日線乖離%", result)
        self.assertFalse(result["75日線上方乖離条件"])
        self.assertLess(result["現在値"], result["MA25"])

    def test_daily_snapshot_marks_overextended_reversal(self):
        index = pd.date_range("2025-01-01", periods=260, freq="B")
        close = np.linspace(100, 200, 260)
        close[-10:] = [210, 215, 220, 225, 230, 235, 240, 245, 250, 242]
        open_ = close - 1.0
        open_[-1] = 250.0
        frame = pd.DataFrame({
            "Open": open_,
            "High": np.maximum(open_, close) + 2.0,
            "Low": np.minimum(open_, close) - 2.0,
            "Close": close,
            "Volume": np.full(len(close), 100000),
        }, index=index)

        result = short_technical_snapshot("5678", "上方乖離", frame)

        self.assertGreaterEqual(result["75日線乖離%"], 5.0)
        self.assertTrue(result["75日線上方乖離条件"])
        self.assertTrue(result["空売り反落確認"])
        self.assertIn("75日線から+", short_watch_reason(result))


if __name__ == "__main__":
    unittest.main()
