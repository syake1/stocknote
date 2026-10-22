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

    def test_one_day_breadth_detects_rotation_earlier(self):
        self.assertEqual(
            ("循環初動", "先回り監視"),
            classify_rotation(0.8, 0.4, -1.0, 50.0, 40.0, 75.0),
        )

    def test_modest_acceleration_is_detected_before_large_rise(self):
        self.assertEqual(
            ("循環初動", "先回り監視"),
            classify_rotation(0.25, -0.4, -2.0, 50.0, 50.0, 50.0),
        )

    def test_extended_sector_is_not_a_fresh_rotation_buy(self):
        self.assertEqual(
            ("上昇後・過熱", "追わない"),
            classify_rotation(1.0, 6.5, 11.0, 80.0, 100.0, 75.0),
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
        self.assertIn("1日上昇銘柄%", rows[0])
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
                 "1日%": 1.0, "5日%": 4.0, "20日%": 6.0, "20日線上": True,
                 "5日線上抜け": True, "5日線乖離%": 0.5},
                {"コード": "3099", "銘柄名": "三越伊勢丹HD", "現在値": 2200,
                 "1日%": -3.0, "5日%": 3.0, "20日%": 4.0, "20日線上": True,
                 "5日線上抜け": False, "5日線乖離%": -0.2},
            ],
        }]
        ranked = rank_rotation_stocks(sectors, market_score=30)
        self.assertEqual(["8267"], [row["コード"] for row in ranked])
        self.assertEqual("地合い逆風・監視優先", ranked[0]["扱い"])
        self.assertTrue(ranked[0]["5日線上抜け"])

    def test_early_rotation_accepts_ma5_cross_before_ma20_recovery(self):
        sectors = [{
            "業種・テーマ": "小売・内需",
            "判定": "循環初動",
            "5日%": -0.5,
            "資金流入点": 2.0,
            "銘柄候補": [{
                "コード": "8267", "銘柄名": "イオン", "現在値": 2500,
                "1日%": 1.2, "5日%": -0.3, "20日%": -2.0,
                "20日線上": False, "5日線上抜け": True, "5日線乖離%": 0.4,
            }],
        }]
        ranked = rank_rotation_stocks(sectors, market_score=55)
        self.assertEqual("8267", ranked[0]["コード"])
        self.assertEqual("循環初動", ranked[0]["業種判定"])
        self.assertEqual("循環初動・5日線上抜け", ranked[0]["扱い"])

    def test_early_rotation_keeps_stock_near_ma5_before_cross(self):
        sectors = [{
            "業種・テーマ": "小売・内需",
            "判定": "循環初動",
            "5日%": -0.5,
            "資金流入点": 1.0,
            "銘柄候補": [{
                "コード": "3382", "銘柄名": "セブン＆アイ", "現在値": 2100,
                "1日%": 0.4, "5日%": -0.6, "20日%": -3.0,
                "20日線上": False, "5日線上抜け": False, "5日線乖離%": -0.7,
            }],
        }]
        ranked = rank_rotation_stocks(sectors, market_score=55)
        self.assertEqual("3382", ranked[0]["コード"])
        self.assertEqual("上抜け前", ranked[0]["初動段階"])
        self.assertEqual("先回り監視・5日線付近", ranked[0]["扱い"])

    def test_overheated_stock_is_removed_from_rotation_candidates(self):
        sectors = [{
            "業種・テーマ": "半導体", "判定": "循環買い", "5日%": 3.0,
            "資金流入点": 4.0,
            "銘柄候補": [{
                "コード": "8035", "銘柄名": "東京エレクトロン", "現在値": 30000,
                "1日%": 2.0, "5日%": 8.0, "20日%": 16.0,
                "20日線上": True, "5日線上抜け": False, "5日線乖離%": 4.0,
            }],
        }]
        self.assertEqual([], rank_rotation_stocks(sectors, market_score=55))


if __name__ == "__main__":
    unittest.main()
