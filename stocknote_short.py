"""Conservative short-selling assessment using only available real data."""
import math
import re


def number(value):
    if value is None:
        return None
    text = str(value).strip().replace(",", "").replace("倍", "").replace("%", "")
    if not text or text.lower() in {"nan", "none", "-", "—"}:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        result = float(match.group())
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def first_value(row, aliases):
    for key in aliases:
        if key in row:
            value = row.get(key)
            if value is not None and str(value).strip().lower() not in {"", "nan", "none", "—", "-"}:
                return value
    return None


def loanable_value(row):
    raw = first_value(row, ["貸借銘柄", "貸借区分", "信用区分", "売建可否", "空売り可否"])
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if any(x in text for x in ("貸借", "売建可", "空売り可", "○", "可")) and not any(x in text for x in ("不可", "×")):
        return True
    if any(x in text for x in ("不可", "×", "信用銘柄")):
        return False
    return None


def short_fundamental_score(f, forecast_per=None):
    """High means weaker/more expensive fundamentals; never invent missing values."""
    score = 50.0
    per = number(forecast_per)
    if per is None:
        per = number(f.get("per"))
    pbr, roe = number(f.get("pbr")), number(f.get("roe"))
    opm, growth = number(f.get("opm")), number(f.get("growth"))
    reasons = []
    if per is not None:
        if per >= 35: score += 18; reasons.append("PER35倍以上")
        elif per >= 25: score += 10; reasons.append("PER25倍以上")
        elif 0 < per <= 15: score -= 12; reasons.append("PER割安")
    if pbr is not None:
        if pbr >= 4: score += 10; reasons.append("PBR高い")
        elif 0 < pbr <= 1.2: score -= 7; reasons.append("PBR低い")
    if roe is not None:
        if roe < 0: score += 14; reasons.append("ROEマイナス")
        elif roe >= .10: score -= 10; reasons.append("ROE良好")
    if opm is not None:
        if opm < 0: score += 15; reasons.append("営業赤字")
        elif opm >= .10: score -= 8; reasons.append("営業利益率良好")
    if growth is not None:
        if growth < 0: score += 12; reasons.append("減収")
        elif growth >= .05: score -= 8; reasons.append("増収")
    return max(0.0, min(100.0, score)), reasons, per


def credit_score(ratio):
    ratio = number(ratio)
    if ratio is None:
        return None
    if ratio < 1:
        return 15.0  # short crowding / squeeze and reverse-stock-fee caution
    if ratio <= 3:
        return 45.0
    if ratio <= 10:
        return 80.0
    return 65.0


def assess_short(technical, fundamental, universe_row, market_score):
    per_csv = first_value(universe_row, ["PER(株価収益率)(予)(倍)", "予想PER", "PER", "PER(倍)"])
    ratio_raw = first_value(universe_row, ["信用倍率", "信用倍率(倍)", "信用取引倍率", "貸借倍率"])
    ratio = number(ratio_raw)
    loanable = loanable_value(universe_row)
    fscore, reasons, per = short_fundamental_score(fundamental, per_csv)
    tscore = float(technical.get("空売りスコア", 0))
    cscore = credit_score(ratio)
    bearish_market = 100.0 - float(market_score)
    # Unknown credit data is not treated as neutral: it contributes zero and
    # prevents an actionable label until the broker data is confirmed.
    total = tscore * .45 + fscore * .25 + (cscore or 0.0) * .15 + bearish_market * .15
    trend_ok = bool(technical.get("空売りトレンド適合"))
    confirmed = ratio is not None and loanable is True
    if not trend_ok:
        status = "トレンド条件外"
    elif not confirmed:
        status = "SBI信用データ確認待ち"
    elif ratio < 1:
        status = "踏み上げ・逆日歩注意"
    elif total >= 75:
        status = "空売り条件到達"
    elif total >= 65:
        status = "空売り条件接近"
    else:
        status = "見送り"
    return {
        "空売り総合評価": round(total, 1), "空売りファンダ": round(fscore, 1),
        "信用倍率": ratio, "貸借確認": loanable, "予想PER": per,
        "空売り状態": status, "空売り理由": "・".join(reasons) if reasons else "大きな悪化材料なし",
    }
