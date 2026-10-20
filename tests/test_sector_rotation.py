import unittest

import numpy as np
import pandas as pd

from stocknote_sector_rotation import analyze_close_prices, classify_rotation


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
        self.assertEqual(index[-1], latest)


if __name__ == "__main__":
    unittest.main()
