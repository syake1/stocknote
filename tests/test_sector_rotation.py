import unittest

import numpy as np
import pandas as pd

from stocknote_sector_rotation import (
    SECTOR_BASKETS,
    STOCK_NAMES,
    analyze_close_prices,
    classify_rotation,
    rank_rotation_stocks,
)


class SectorRotationTests(unittest.TestCase):
    def test_fresh_strength_is_rotation_buy(self):
        self.assertEqual(
            ("循環買い", "買い候補"),
            classify_rotation(0.8, 2.0, 4.0, 75.0, 75.0),
        )

    def test_weak_sector_is_not_a_buy(self):
        self.assertEqual(
            ("資金流出", "見送り"),
            classify_rotation(-0.5, -2.0, -4.0, 25.0, 25.0),
        )

    def test_analysis_ranks_stronger_sector_first(self):
        index = pd.date_range("2026-01-01", periods=30, freq="B")
        close = pd.DataFrame({
            "A.T": np.linspace(100, 112, len(index)),
            "B.T": np.linspace(100, 110, len(index)),
            "C.T": np.linspace(100, 92, len(index)),
            "D.T": np.linspace(100, 94, len(index)),
        }, index=index)
        rows, latest = analyze_close_prices(
            close,
            {"強い業種": ["A.T", "B.T"], "弱い業種": ["C.T", "D.T"]},
        )
        self.assertEqual("強い業種", rows[0]["業種・テーマ"])
        self.assertIn("1日%", rows[0]["銘柄候補"][0])
        self.assertIn("20日%", rows[0]["銘柄候補"][0])
        self.assertEqual(index[-1], latest)

    def test_construction_sector_covers_shimizu_and_kumagai(self):
        self.assertIn("1803.T", SECTOR_BASKETS["建設"])
        self.assertIn("1861.T", SECTOR_BASKETS["建設"])
        self.assertEqual("清水建設", STOCK_NAMES["1803"])
        self.assertEqual("熊谷組", STOCK_NAMES["1861"])

    def test_rotation_stock_ranking_keeps_signal_separate(self):
        sectors = [{
            "業種・テーマ": "小売・内需",
            "判定": "循環買い",
            "5日%": 2.0,
            "資金流入点": 3.0,
            "銘柄候補": [
                {"コード": "8267", "銘柄名": "イオン", "現在値": 2500,
                 "1日%": 1.0, "5日%": 4.0, "20日%": 6.0, "20日線上": True},
                {"コード": "3099", "銘柄名": "三越伊勢丹HD", "現在値": 2200,
                 "1日%": -3.0, "5日%": 3.0, "20日%": 4.0, "20日線上": True},
            ],
        }]
        ranked = rank_rotation_stocks(sectors, market_score=30)
        self.assertEqual(["8267"], [row["コード"] for row in ranked])
        self.assertEqual("地合い逆風・監視優先", ranked[0]["扱い"])


if __name__ == "__main__":
    unittest.main()
