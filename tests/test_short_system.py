import unittest

from stocknote_short import assess_short, credit_score, short_fundamental_score


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


if __name__ == "__main__":
    unittest.main()
