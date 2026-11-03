"""All-market surge-candidate helpers for Stocknote."""

from __future__ import annotations

import io
import json
import math
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from stocknote_decline_counter import completed_daily_history


JPX_UNIVERSE_URL = (
    "https://www.jpx.co.jp/markets/statistics-equities/misc/"
    "tvdivq0000001vg2-att/data_j.xlsx"
)
DATA_DIR = os.getenv("STOCKNOTE_DATA_DIR", "data")
JPX_CACHE_PATH = os.path.join(DATA_DIR, "jpx_listed_universe.csv")
SURGE_PATH = os.path.join(DATA_DIR, "surge_candidates.csv")
SURGE_META_PATH = os.path.join(DATA_DIR, "surge_candidates_meta.json")
SURGE_BACKTEST_PATH = os.path.join(DATA_DIR, "surge_backtest_summary.csv")
ALLOWED_MARKETS = (
    "プライム（内国株式）",
    "スタンダード（内国株式）",
    "グロース（内国株式）",
)


def normalize_code(value) -> str:
    text = str(value).strip().upper().replace(".T", "")
    text = re.sub(r"\.0$", "", text)
    return text if re.fullmatch(r"[0-9A-Z]{4}", text) else ""


def parse_jpx_universe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return domestic Prime, Standard and Growth operating-company rows."""
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["コード", "銘柄名", "市場", "業種"])
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    required = {"コード", "銘柄名", "市場・商品区分"}
    if not required.issubset(data.columns):
        raise ValueError("JPX上場銘柄一覧の列構成を確認できません")

    out = pd.DataFrame()
    out["コード"] = data["コード"].map(normalize_code)
    out["銘柄名"] = data["銘柄名"].fillna("").astype(str).str.strip()
    out["市場"] = data["市場・商品区分"].fillna("").astype(str).str.strip()
    sector_column = "33業種区分" if "33業種区分" in data.columns else None
    out["業種"] = (
        data[sector_column].fillna("その他").astype(str).str.strip()
        if sector_column
        else "その他"
    )
    out = out[out["市場"].isin(ALLOWED_MARKETS) & out["コード"].ne("")]
    return out.drop_duplicates("コード", keep="first").reset_index(drop=True)


def fetch_jpx_universe(url: str = JPX_UNIVERSE_URL, timeout: int = 30) -> pd.DataFrame:
    """Download and cache the official JPX listed-issue workbook."""
    response = requests.get(
        url,
        headers={"User-Agent": "Stocknote/1.0 (+JPX listed issue scanner)"},
        timeout=timeout,
    )
    response.raise_for_status()
    raw = pd.read_excel(io.BytesIO(response.content), dtype=str, engine="openpyxl")
    universe = parse_jpx_universe(raw)
    if len(universe) < 3000:
        raise ValueError(f"JPX上場銘柄一覧が少なすぎます: {len(universe)}銘柄")
    os.makedirs(DATA_DIR, exist_ok=True)
    universe.to_csv(JPX_CACHE_PATH, index=False, encoding="utf-8-sig")
    return universe


def load_jpx_universe(refresh: bool = True) -> pd.DataFrame:
    """Load JPX universe, falling back to the most recent cached snapshot."""
    if refresh:
        try:
            return fetch_jpx_universe()
        except Exception:
            if not os.path.exists(JPX_CACHE_PATH):
                raise
    if not os.path.exists(JPX_CACHE_PATH):
        return pd.DataFrame(columns=["コード", "銘柄名", "市場", "業種"])
    return pd.read_csv(JPX_CACHE_PATH, dtype=str, encoding="utf-8-sig")


def _rsi14(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    value = 100 - 100 / (1 + rs)
    value = value.mask((loss == 0) & (gain > 0), 100.0)
    value = value.mask((loss == 0) & (gain == 0), 50.0)
    return value.fillna(50.0)


def _weekly_state(close: pd.Series) -> str:
    weekly = close.resample("W-FRI").last().dropna()
    if len(weekly) < 15:
        return "履歴不足"
    ma13 = weekly.rolling(13).mean()
    if pd.isna(ma13.iloc[-1]) or pd.isna(ma13.iloc[-2]):
        return "履歴不足"
    if weekly.iloc[-1] > ma13.iloc[-1] and ma13.iloc[-1] > ma13.iloc[-2]:
        return "上向き"
    if weekly.iloc[-1] < ma13.iloc[-1] and ma13.iloc[-1] < ma13.iloc[-2]:
        return "下向き"
    return "横ばい"


def surge_snapshot(code, name, market, sector, history, now=None):
    """Calculate completed-daily-bar features used by the all-market scan."""
    data = completed_daily_history(history, now=now)
    if data.empty or "Close" not in data or "Volume" not in data:
        return None
    close = pd.to_numeric(data["Close"], errors="coerce")
    volume = pd.to_numeric(data["Volume"], errors="coerce")
    high = pd.to_numeric(data.get("High", close), errors="coerce")
    low = pd.to_numeric(data.get("Low", close), errors="coerce")
    aligned = pd.concat(
        [close.rename("close"), volume.rename("volume"), high.rename("high"), low.rename("low")],
        axis=1,
    ).dropna(subset=["close", "volume"])
    if len(aligned) < 70:
        return None

    close = aligned["close"]
    volume = aligned["volume"]
    high = aligned["high"].fillna(close)
    low = aligned["low"].fillna(close)
    price = float(close.iloc[-1])
    previous_close = float(close.iloc[-2])
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    previous_volume_mean = float(volume.iloc[-21:-1].mean())
    volume_ratio = float(volume.iloc[-1] / previous_volume_mean) if previous_volume_mean > 0 else 0.0
    turnover = float(price * volume.iloc[-1])
    average_turnover = float((close.iloc[-21:-1] * volume.iloc[-21:-1]).mean())
    day_return = float((price / previous_close - 1.0) * 100.0)
    five_return = float((price / close.iloc[-6] - 1.0) * 100.0)
    prior_five_return = float((close.iloc[-6] / close.iloc[-11] - 1.0) * 100.0)
    twenty_return = float((price / close.iloc[-21] - 1.0) * 100.0)
    acceleration = five_return - prior_five_return
    previous_high20 = float(high.iloc[-21:-1].max())
    high_distance = float((price / previous_high20 - 1.0) * 100.0)
    ma5_cross = bool(price > ma5.iloc[-1] and close.iloc[-2] <= ma5.iloc[-2])
    above_ma5 = bool(price > ma5.iloc[-1])
    ma20_rising = bool(ma20.iloc[-1] > ma20.iloc[-6])
    rsi = float(_rsi14(close).iloc[-1])
    daily_range = float(high.iloc[-1] - low.iloc[-1])
    close_position = float((price - low.iloc[-1]) / daily_range) if daily_range > 0 else 0.5
    weekly = _weekly_state(close)

    liquid = average_turnover >= 50_000_000 and turnover >= 80_000_000
    price_ok = 100.0 <= price <= 50_000.0
    move_ok = -1.5 <= day_return <= 7.0 and five_return <= 15.0 and twenty_return <= 30.0
    momentum_ok = bool(
        above_ma5
        and volume_ratio >= 1.5
        and 42.0 <= rsi <= 75.0
        and close_position >= 0.55
        and (ma5_cross or high_distance >= -2.0 or acceleration >= 3.0)
    )
    overheated = bool(day_return > 7.0 or five_return > 15.0 or twenty_return > 30.0 or rsi > 78.0)

    volume_points = float(np.clip((volume_ratio - 1.0) / 2.0 * 25.0, 0.0, 25.0))
    turnover_oku = turnover / 100_000_000.0
    turnover_points = float(np.clip(math.log10(max(turnover_oku, 0.1)) * 4.0 + 4.0, 0.0, 10.0))
    ma_points = 15.0 if ma5_cross else 7.0 if above_ma5 else 0.0
    high_points = 20.0 if high_distance >= 0 else float(np.clip((3.0 + high_distance) / 3.0 * 16.0, 0.0, 16.0))
    acceleration_points = float(np.clip(acceleration / 6.0 * 10.0, 0.0, 10.0))
    rsi_points = 10.0 if 52.0 <= rsi <= 68.0 else 6.0 if 42.0 <= rsi <= 75.0 else 0.0
    close_points = float(np.clip((close_position - 0.5) / 0.5 * 10.0, 0.0, 10.0))
    weekly_points = 5.0 if weekly == "上向き" else 2.0 if weekly == "横ばい" else 0.0
    base_score = volume_points + turnover_points + ma_points + high_points + acceleration_points + rsi_points + close_points + weekly_points
    if not liquid or not price_ok or not move_ok or not momentum_ok or overheated:
        base_score = min(base_score, 59.0)

    reasons = []
    if volume_ratio >= 1.5:
        reasons.append(f"出来高{volume_ratio:.1f}倍")
    if ma5_cross:
        reasons.append("5日線上抜け")
    if high_distance >= 0:
        reasons.append("20日高値更新")
    elif high_distance >= -2.0:
        reasons.append("20日高値接近")
    if acceleration >= 3.0:
        reasons.append("上昇加速")
    if weekly == "上向き":
        reasons.append("週足上向き")

    return {
        "コード": normalize_code(code),
        "銘柄名": str(name or code),
        "市場": str(market or ""),
        "業種": str(sector or "その他"),
        "判定日": pd.Timestamp(close.index[-1]).strftime("%Y/%m/%d"),
        "終値": price,
        "前日終値": previous_close,
        "前日比%": day_return,
        "5日%": five_return,
        "20日%": twenty_return,
        "5日加速pt": acceleration,
        "出来高倍率": volume_ratio,
        "売買代金億円": turnover_oku,
        "平均売買代金億円": average_turnover / 100_000_000.0,
        "20日高値距離%": high_distance,
        "RSI14": rsi,
        "5日線上抜け": ma5_cross,
        "週足": weekly,
        "基礎点": float(np.clip(base_score, 0.0, 100.0)),
        "業種点": 0.0,
        "急騰初動点": float(np.clip(base_score, 0.0, 100.0)),
        "過熱": overheated,
        "初動条件": bool(liquid and price_ok and move_ok and momentum_ok and not overheated),
        "理由": " / ".join(reasons) if reasons else "条件待ち",
    }


def historical_market_order_events(code, history, now=None):
    """Backtest prior-close signals using the following session's opening price."""
    data = completed_daily_history(history, now=now)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if data.empty or not required.issubset(data.columns):
        return []
    numeric = pd.DataFrame(
        {column: pd.to_numeric(data[column], errors="coerce") for column in required},
        index=data.index,
    ).dropna()
    if len(numeric) < 80:
        return []

    close = numeric["Close"]
    open_ = numeric["Open"]
    high = numeric["High"]
    low = numeric["Low"]
    volume = numeric["Volume"]
    ma5 = close.rolling(5).mean()
    volume_mean = volume.shift(1).rolling(20).mean()
    volume_ratio = volume / volume_mean.replace(0, np.nan)
    turnover = close * volume
    average_turnover = turnover.shift(1).rolling(20).mean()
    day_return = close.pct_change() * 100.0
    five_return = close.pct_change(5) * 100.0
    prior_five_return = (close.shift(5) / close.shift(10) - 1.0) * 100.0
    acceleration = five_return - prior_five_return
    twenty_return = close.pct_change(20) * 100.0
    previous_high20 = high.shift(1).rolling(20).max()
    high_distance = (close / previous_high20 - 1.0) * 100.0
    ma5_cross = (close > ma5) & (close.shift(1) <= ma5.shift(1))
    above_ma5 = close > ma5
    rsi = _rsi14(close)
    daily_range = (high - low).replace(0, np.nan)
    close_position = ((close - low) / daily_range).fillna(0.5)

    liquid = (average_turnover >= 50_000_000) & (turnover >= 80_000_000)
    price_ok = close.between(100.0, 50_000.0)
    move_ok = day_return.between(-1.5, 7.0) & (five_return <= 15.0) & (twenty_return <= 30.0)
    momentum_ok = (
        above_ma5
        & (volume_ratio >= 1.5)
        & rsi.between(42.0, 75.0)
        & (close_position >= 0.55)
        & (ma5_cross | (high_distance >= -2.0) | (acceleration >= 3.0))
    )
    overheated = (day_return > 7.0) | (five_return > 15.0) | (twenty_return > 30.0) | (rsi > 78.0)

    volume_points = ((volume_ratio - 1.0) / 2.0 * 25.0).clip(0.0, 25.0)
    turnover_oku = turnover / 100_000_000.0
    turnover_points = turnover_oku.clip(lower=0.1).map(lambda value: math.log10(value) * 4.0 + 4.0).clip(0.0, 10.0)
    ma_points = pd.Series(np.where(ma5_cross, 15.0, np.where(above_ma5, 7.0, 0.0)), index=close.index)
    high_points = pd.Series(
        np.where(high_distance >= 0.0, 20.0, ((3.0 + high_distance) / 3.0 * 16.0).clip(0.0, 16.0)),
        index=close.index,
    )
    acceleration_points = (acceleration / 6.0 * 10.0).clip(0.0, 10.0)
    rsi_points = pd.Series(
        np.where(rsi.between(52.0, 68.0), 10.0, np.where(rsi.between(42.0, 75.0), 6.0, 0.0)),
        index=close.index,
    )
    close_points = ((close_position - 0.5) / 0.5 * 10.0).clip(0.0, 10.0)
    score = volume_points + turnover_points + ma_points + high_points + acceleration_points + rsi_points + close_points
    valid = liquid & price_ok & move_ok & momentum_ok & ~overheated & (score >= 55.0)

    next_open = open_.shift(-1)
    future_close_1 = close.shift(-1)
    future_close_3 = close.shift(-3)
    future_close_5 = close.shift(-5)
    future_high_5 = pd.concat([high.shift(-offset) for offset in range(1, 6)], axis=1).max(axis=1)
    future_low_5 = pd.concat([low.shift(-offset) for offset in range(1, 6)], axis=1).min(axis=1)
    entry_date = pd.Series(numeric.index, index=numeric.index).shift(-1)

    events = []
    last_position = -10
    for position, index in enumerate(numeric.index):
        if not bool(valid.loc[index]) or position <= last_position + 5:
            continue
        entry = float(next_open.loc[index]) if pd.notna(next_open.loc[index]) else np.nan
        if not np.isfinite(entry) or entry <= 0 or pd.isna(future_close_5.loc[index]):
            continue
        events.append({
            "コード": normalize_code(code),
            "シグナル日": pd.Timestamp(index).strftime("%Y/%m/%d"),
            "成行日": pd.Timestamp(entry_date.loc[index]).strftime("%Y/%m/%d"),
            "検証点": float(score.loc[index]),
            "成行価格": entry,
            "1日損益%": float((future_close_1.loc[index] / entry - 1.0) * 100.0),
            "3日損益%": float((future_close_3.loc[index] / entry - 1.0) * 100.0),
            "5日損益%": float((future_close_5.loc[index] / entry - 1.0) * 100.0),
            "5日以内最大上昇%": float((future_high_5.loc[index] / entry - 1.0) * 100.0),
            "5日最大DD%": float((future_low_5.loc[index] / entry - 1.0) * 100.0),
        })
        last_position = position
    return events


def summarize_market_order_backtest(events, thresholds=(60, 65, 70, 75)) -> pd.DataFrame:
    frame = pd.DataFrame(events)
    rows = []
    for threshold in thresholds:
        selected = frame[frame["検証点"] >= threshold] if not frame.empty else pd.DataFrame()
        if selected.empty:
            rows.append({
                "基準点": threshold,
                "件数": 0,
                "5日勝率%": 0.0,
                "5日平均損益%": 0.0,
                "5日中央値%": 0.0,
                "5日以内+5%到達率": 0.0,
                "平均最大DD%": 0.0,
                "最大DD%": 0.0,
            })
            continue
        rows.append({
            "基準点": threshold,
            "件数": int(len(selected)),
            "5日勝率%": float((selected["5日損益%"] > 0).mean() * 100.0),
            "5日平均損益%": float(selected["5日損益%"].mean()),
            "5日中央値%": float(selected["5日損益%"].median()),
            "5日以内+5%到達率": float((selected["5日以内最大上昇%"] >= 5.0).mean() * 100.0),
            "平均最大DD%": float(selected["5日最大DD%"].mean()),
            "最大DD%": float(selected["5日最大DD%"].min()),
        })
    return pd.DataFrame(rows)


def verified_morning_threshold(summary: pd.DataFrame):
    """Require a useful sample, positive expectancy and a clear win-rate edge."""
    if summary is None or summary.empty:
        return None
    for threshold in sorted(summary["基準点"].astype(int).unique(), reverse=True):
        row = summary[summary["基準点"].eq(threshold)].iloc[0]
        if (
            int(row["件数"]) >= 100
            and float(row["5日勝率%"]) >= 55.0
            and float(row["5日平均損益%"]) >= 0.5
            and float(row["5日以内+5%到達率"]) >= 20.0
        ):
            return int(threshold)
    return None


def rank_surge_candidates(rows, limit: int = 100, verified_threshold=None) -> pd.DataFrame:
    """Add sector breadth and rank candidates, keeping overheated names out."""
    frame = pd.DataFrame([row for row in rows if row])
    if frame.empty:
        return frame
    breadth = frame.groupby("業種", dropna=False).agg(
        業種中央値=("前日比%", "median"),
        業種上昇率=("前日比%", lambda values: float((values > 0).mean() * 100.0)),
        業種出来高増加率=("出来高倍率", lambda values: float((values >= 1.2).mean() * 100.0)),
    )
    frame = frame.join(breadth, on="業種")
    frame["業種点"] = (
        np.clip(frame["業種中央値"], 0.0, 2.0) / 2.0 * 6.0
        + np.clip(frame["業種上昇率"] - 45.0, 0.0, 25.0) / 25.0 * 5.0
        + np.clip(frame["業種出来高増加率"] - 20.0, 0.0, 35.0) / 35.0 * 4.0
    )
    frame["急騰初動点"] = np.clip(frame["基礎点"] + frame["業種点"], 0.0, 100.0)
    morning_condition = pd.Series(False, index=frame.index)
    if verified_threshold is not None:
        morning_condition = frame["初動条件"] & (frame["基礎点"] >= float(verified_threshold))
    frame["判定"] = np.select(
        [morning_condition, frame["初動条件"] & (frame["基礎点"] >= 55.0)],
        ["朝成候補", "監視"],
        default="見送り",
    )
    frame["翌朝の扱い"] = np.where(
        frame["判定"].eq("朝成候補"),
        "SBI気配が前日終値比+3%未満なら成行候補",
        np.where(frame["判定"].eq("監視"), "寄付き後の5日線維持を確認", "成行見送り"),
    )
    ranked = frame[frame["判定"].ne("見送り")].sort_values(
        ["急騰初動点", "出来高倍率", "売買代金億円"], ascending=[False, False, False]
    )
    return ranked.head(limit).reset_index(drop=True)


def save_surge_candidates(frame: pd.DataFrame, scanned: int, failed: int = 0) -> dict:
    os.makedirs(DATA_DIR, exist_ok=True)
    frame.to_csv(SURGE_PATH, index=False, encoding="utf-8-sig")
    meta = {
        "updated_at": datetime.now(ZoneInfo("Asia/Tokyo")).isoformat(timespec="seconds"),
        "scanned": int(scanned),
        "failed": int(failed),
        "candidates": int(len(frame)),
        "morning_candidates": int(frame["判定"].eq("朝成候補").sum()) if "判定" in frame else 0,
    }
    with open(SURGE_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, ensure_ascii=False, indent=2)
    return meta


def save_surge_backtest(summary: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    summary.to_csv(SURGE_BACKTEST_PATH, index=False, encoding="utf-8-sig")


def load_surge_backtest() -> pd.DataFrame:
    if not os.path.exists(SURGE_BACKTEST_PATH):
        return pd.DataFrame()
    return pd.read_csv(SURGE_BACKTEST_PATH, encoding="utf-8-sig")


def load_surge_candidates():
    if not os.path.exists(SURGE_PATH):
        return pd.DataFrame(), None
    frame = pd.read_csv(SURGE_PATH, dtype={"コード": str}, encoding="utf-8-sig")
    meta = None
    if os.path.exists(SURGE_META_PATH):
        try:
            with open(SURGE_META_PATH, encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception:
            meta = None
    return frame, meta


def opening_gap_percent(indicative_price, previous_close):
    try:
        indicative = float(indicative_price)
        previous = float(previous_close)
    except (TypeError, ValueError):
        return None
    if indicative <= 0 or previous <= 0:
        return None
    return (indicative / previous - 1.0) * 100.0


def market_order_decision(indicative_price, previous_close, max_gap: float = 3.0):
    gap = opening_gap_percent(indicative_price, previous_close)
    if gap is None:
        return "気配値を入力してください", None
    if gap >= max_gap:
        return "成行見送り", gap
    if gap <= -5.0:
        return "急落気配のため成行見送り", gap
    return "成行条件内", gap
