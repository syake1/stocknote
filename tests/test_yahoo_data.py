import unittest

from yahoo_data import parse_chart_payload


class YahooDataTests(unittest.TestCase):
    def test_parses_close_and_dividend_event(self):
        payload = {
            "chart": {
                "error": None,
                "result": [
                    {
                        "timestamp": [1743030000, 1743116400],
                        "indicators": {"quote": [{"close": [1000.0, 970.0]}]},
                        "events": {
                            "dividends": {
                                "1743116400": {
                                    "amount": 30.0,
                                    "date": 1743116400,
                                }
                            }
                        },
                    }
                ],
            }
        }

        history = parse_chart_payload(payload)

        self.assertEqual(history["Close"].tolist(), [1000.0, 970.0])
        self.assertEqual(history["Dividends"].tolist(), [0.0, 30.0])

    def test_raises_yahoo_error(self):
        payload = {
            "chart": {
                "result": None,
                "error": {"code": "Not Found", "description": "No data found"},
            }
        }

        with self.assertRaisesRegex(ValueError, "No data found"):
            parse_chart_payload(payload)


if __name__ == "__main__":
    unittest.main()
