"""Daily Japanese sector rotation analysis using representative liquid stocks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import numpy as np
import pandas as pd

from yahoo_data import fetch_yahoo_history


SECTOR_BASKETS = {
    "半導体": ["8035.T", "6857.T", "6920.T", "4063.T", "7735.T"],
    "証券": ["8601.T", "8604.T", "8473.T", "8697.T"],
    "銀行": ["8306.T", "8316.T", "8411.T", "7182.T"],
    "商社": ["8001.T", "8002.T", "8031.T", "8053.T", "8058.T"],
    "自動車": ["7203.T", "7267.T", "7201.T", "7269.T", "7270.T"],
    "防衛・重工": ["7011.T", "7012.T", "7013.T", "6208.T"],
    "電線・データセンター": ["5801.T", "5802.T", "5803.T", "1942.T"],
    "資源・エネルギー": ["1605.T", "1662.T", "5020.T", "5019.T"],
    "不動産": ["8801.T", "8802.T", "8830.T", "3289.T"],
    "建設": ["1801.T", "1802.T", "1803.T", "1812.T", "1861.T"],
    "小売・内需": ["9983.T", "3382.T", "8267.T", "3099.T"],
    "通信": ["9432.T", "9433.T", "9434.T", "9984.T"],
    "医薬品": ["4502.T", "4503.T", "4519.T", "4568.T"],
}

STOCK_NAMES = {
    "8035": "東京エレクトロン", "6857": "アドバンテスト", "6920": "レーザーテック",
    "4063": "信越化学工業", "7735": "SCREEN HD", "8601": "大和証券G",
    "8604": "野村HD", "8473": "SBI HD", "8697": "日本取引所G",
    "8306": "三菱UFJ", "8316": "三井住友FG", "8411": "みずほFG", "7182": "ゆうちょ銀行",
    "8001": "伊藤忠商事", "8002": "丸紅", "8031": "三井物産", "8053": "住友商事", "8058": "三菱商事",
    "7203": "トヨタ自動車", "7267": "ホンダ", "7201": "日産自動車", "7269": "スズキ", "7270": "SUBARU",
    "7011": "三菱重工業", "7012": "川崎重工業", "7013": "IHI", "6208": "石川製作所",
    "5801": "古河電気工業", "5802": "住友電気工業", "5803": "フジクラ", "1942": "関電工",
    "1605": "INPEX", "1662": "石油資源開発", "5020": "ENEOS HD", "5019": "出光興産",
    "8801": "三井不動産", "8802": "三菱地所", "8830": "住友不動産", "3289": "東急不動産HD",
    "1801": "大成建設", "1802": "大林組", "1803": "清水建設", "1812": "鹿島", "1861": "熊谷組",
    "9983": "ファーストリテイリング", "3382": "セブン＆アイ", "8267": "イオン", "3099": "三越伊勢丹HD",
    "9432": "NTT", "9433": "KDDI", "9434": "ソフトバンク", "9984": "ソフトバンクG",
    "4502": "武田薬品工業", "4503": "アステラス製薬", "4519": "中外製薬", "4568": "第一三共",
}


def classify_rotation(day1, day5, day20, breadth5, above_ma20):
    """Classify whether a sector is receiving fresh rotation money."""
    if day1 > 0 and day5 >= 1.0 and breadth5 >= 60 and -3.0 <= day20 <= 8.0:
        return "循環買い", "買い候補"
    if day5 >= 1.0 and day20 > 0 and breadth5 >= 60 and above_ma20 >= 60:
        return "上昇継続", "押し目待ち"
    if day5 > 0 and day20 < 0 and breadth5 >= 50:
        return "反発初動", "監視"
    if day5 <= -1.0 and breadth5 <= 40:
        return "資金流出", "見送り"
    return "様子見", "中立"


def analyze_close_prices(close, baskets=None):
    """Return ranked sector rows from a ticker-by-column close-price frame."""
    baskets = baskets or SECTOR_BASKETS
    if close is None or close.empty:
        return [], None
    rows = []
    latest_dates = []
    for sector, symbols in baskets.items():
        metrics = []
        for symbol in symbols:
            if symbol not in close.columns:
                continue
            series = pd.to_numeric(close[symbol], errors="coerce").dropna()
            if len(series) < 22:
                continue
            current = float(series.iloc[-1])
            code = symbol.removesuffix(".T")
            metrics.append({
                "コード": code,
                "銘柄名": STOCK_NAMES.get(code, code),
                "現在値": current,
                "1日%": (current / float(series.iloc[-2]) - 1) * 100,
                "5日%": (current / float(series.iloc[-6]) - 1) * 100,
                "20日%": (current / float(series.iloc[-21]) - 1) * 100,
                "5日上昇": current > float(series.iloc[-6]),
                "20日線上": current >= float(series.tail(20).mean()),
            })
            latest_dates.append(series.index[-1])
        if not metrics:
            continue
        frame = pd.DataFrame(metrics)
        day1 = float(frame["1日%"].mean())
        day5 = float(frame["5日%"].mean())
        day20 = float(frame["20日%"].mean())
        breadth5 = float(frame["5日上昇"].mean() * 100)
        above_ma20 = float(frame["20日線上"].mean() * 100)
        rotation, action = classify_rotation(day1, day5, day20, breadth5, above_ma20)
        flow_score = day1 * 0.20 + day5 * 0.50 + day20 * 0.20 + (breadth5 - 50) * 0.03
        rows.append({
            "業種・テーマ": sector,
            "判定": rotation,
            "売買目安": action,
            "1日%": day1,
            "5日%": day5,
            "20日%": day20,
            "5日上昇銘柄%": breadth5,
            "20日線上銘柄%": above_ma20,
            "資金流入点": float(flow_score),
            "取得銘柄数": len(metrics),
            "銘柄候補": sorted(
                frame[["コード", "銘柄名", "現在値", "1日%", "5日%", "20日%"]].to_dict("records"),
                key=lambda item: item["5日%"],
                reverse=True,
            ),
        })
    rows.sort(key=lambda row: row["資金流入点"], reverse=True)
    latest = max(latest_dates) if latest_dates else None
    return rows, latest


def download_sector_rotation():
    symbols = list(dict.fromkeys(symbol for values in SECTOR_BASKETS.values() for symbol in values))
    end = date.today() + timedelta(days=1)
    start = end - timedelta(days=150)
    series_by_symbol = {}

    def fetch_close(symbol):
        history = fetch_yahoo_history(symbol, start, end)
        return pd.to_numeric(history["Close"], errors="coerce").dropna()

    # A failed symbol must not hide the remaining sectors from the daily report.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(fetch_close, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                series = future.result()
            except Exception:
                continue
            if not series.empty:
                series_by_symbol[symbol] = series

    if not series_by_symbol:
        return [], None
    close = pd.DataFrame(series_by_symbol).sort_index()
    return analyze_close_prices(close)
