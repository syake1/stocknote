"""Parser for daily price CSV files exported by HYPER SBI 2."""

from __future__ import annotations

from io import StringIO

import pandas as pd


def _decode_csv(content: bytes) -> str:
    errors: list[str] = []
    for encoding in ("cp932", "utf-8-sig", "utf-8"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError as exc:
            errors.append(f"{encoding}: {exc}")
    raise ValueError("CSVの文字コードを判定できませんでした。" + " / ".join(errors))


def parse_sbi_csv(content: bytes) -> pd.DataFrame:
    """Return an analyzer-ready frame from a HYPER SBI 2 daily CSV export."""
    if not content:
        raise ValueError("CSVファイルが空です。")

    text = _decode_csv(content)
    lines = text.splitlines()
    header_row = next(
        (
            position
            for position, line in enumerate(lines[:30])
            if "日付" in line and "終値" in line
        ),
        None,
    )
    if header_row is None:
        raise ValueError("SBI CSVの見出し（日付・終値）を確認できませんでした。")

    frame = pd.read_csv(StringIO(text), skiprows=header_row)
    frame.columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
    date_column = next(
        (column for column in frame.columns if column in {"日付", "年月日"}), None
    )
    close_column = next(
        (column for column in frame.columns if column in {"終値", "終値(円)", "終値（円）"}),
        None,
    )
    if date_column is None or close_column is None:
        raise ValueError("SBI CSVに日付または終値の列がありません。")

    dates = pd.to_datetime(frame[date_column], errors="coerce")
    closes = pd.to_numeric(
        frame[close_column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("--", "", regex=False),
        errors="coerce",
    )
    history = pd.DataFrame(
        {"Close": closes.to_numpy(), "Dividends": 0.0}, index=dates
    )
    history = history[~history.index.isna()].dropna(subset=["Close"])
    history.index = pd.DatetimeIndex(history.index).normalize()
    history = history[~history.index.duplicated(keep="last")].sort_index()
    if history.empty:
        raise ValueError("SBI CSVから有効な日足データを読み取れませんでした。")
    return history


def merge_yahoo_dividends(
    sbi_history: pd.DataFrame, yahoo_history: pd.DataFrame
) -> pd.DataFrame:
    """Use SBI closes and align Yahoo dividend events to the same sessions."""
    history = sbi_history.copy()
    yahoo = yahoo_history.copy()
    yahoo.index = pd.to_datetime(yahoo.index).tz_localize(None).normalize()
    dividends = pd.to_numeric(yahoo["Dividends"], errors="coerce").fillna(0.0)
    history["Dividends"] = dividends.reindex(history.index).fillna(0.0)
    return history
