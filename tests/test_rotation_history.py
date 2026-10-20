from stocknote_rotation_history import analyze_rotation_patterns, load_rotation_history, save_rotation_snapshot


def candidate(rank_return=3.0):
    return {
        "業種・テーマ": "小売・内需", "コード": "8267", "銘柄名": "イオン",
        "現在値": 2500, "1日%": 1.2, "5日%": rank_return, "20日%": 6.0,
        "循環点": 72.0, "扱い": "循環候補・PSAR確認前",
    }


def test_snapshot_is_upserted_by_date(tmp_path):
    path = tmp_path / "history.json"
    save_rotation_snapshot([candidate(2.0)], "2026-10-20", path)
    save_rotation_snapshot([candidate(3.0)], "2026-10-20", path)
    history = load_rotation_history(path)
    assert len(history) == 1
    assert history[0]["candidates"][0]["day5"] == 3.0


def test_accelerating_repeat_is_morning_candidate():
    history = [{
        "date": "2026-10-20",
        "candidates": [{"rank": 4, "code": "8267", "day5": 1.5}],
    }]
    result = analyze_rotation_patterns([candidate(3.0)], history, 55, "2026-10-21")[0]
    assert result["初動加速"] is True
    assert result["翌朝の扱い"] == "翌朝の成行検討（手動確認）"


def test_risk_off_never_labels_market_order_candidate():
    history = [{
        "date": "2026-10-20",
        "candidates": [{"rank": 4, "code": "8267", "day5": 1.5}],
    }]
    result = analyze_rotation_patterns([candidate(3.0)], history, 30, "2026-10-21")[0]
    assert result["初動加速"] is True
    assert result["翌朝の扱い"] == "寄り付き確認（地合い逆風）"
