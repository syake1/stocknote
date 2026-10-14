import unittest

import pandas as pd

from sbi_csv import merge_yahoo_dividends, parse_sbi_csv


class SbiCsvTests(unittest.TestCase):
    def test_accepts_cp932_and_metadata_rows(self) -> None:
        content = (
            "銘柄コード,5256\n"
            "日付,始値,高値,安値,終値,出来高,VWAP\n"
            '2026/09/29,"3,400","3,450","3,300","3,395","20,400",3390\n'
            '2026/09/30,"3,395","3,450","3,060","3,295","66,302",3290\n'
        ).encode("cp932")

        result = parse_sbi_csv(content)

        self.assertEqual(list(result.columns), ["Close", "Dividends"])
        self.assertEqual(result.loc[pd.Timestamp("2026-09-30"), "Close"], 3295)
        self.assertEqual(result["Dividends"].sum(), 0)

    def test_merge_yahoo_dividends_keeps_sbi_close(self) -> None:
        sbi = pd.DataFrame(
            {"Close": [1000, 970], "Dividends": [0.0, 0.0]},
            index=pd.to_datetime(["2026-03-30", "2026-03-31"]),
        )
        yahoo = pd.DataFrame(
            {"Close": [999, 960], "Dividends": [0.0, 30.0]},
            index=pd.to_datetime(["2026-03-30", "2026-03-31"]),
        )

        result = merge_yahoo_dividends(sbi, yahoo)

        self.assertEqual(result.loc[pd.Timestamp("2026-03-31"), "Close"], 970)
        self.assertEqual(result.loc[pd.Timestamp("2026-03-31"), "Dividends"], 30)


if __name__ == "__main__":
    unittest.main()
