import pandas as pd

import morning_scan


def test_ma75_touch_is_required_within_recent_three_days():
    ma75 = pd.Series([100.0, 100.0, 100.0, 100.0])
    assert morning_scan.touched_ma75_recently(
        pd.Series([110.0, 101.0, 99.0, 102.0]), ma75
    )
    assert not morning_scan.touched_ma75_recently(
        pd.Series([99.0, 106.0, 105.0, 104.0]), ma75
    )


def test_bb_lower_touch_requires_actual_low_reach():
    lower = pd.Series([100.0, 101.0, 102.0])
    assert morning_scan.touched_bb_lower_recently(
        pd.Series([103.0, 100.5, 104.0]), lower
    )
    assert not morning_scan.touched_bb_lower_recently(
        pd.Series([103.0, 102.0, 104.0]), lower
    )


def test_contrarian_score_remains_primary_after_fundamental_check():
    row = {"score": 90.0, "code": "1234"}
    result = morning_scan.add_fundamental_judgement(
        row, {"score": 50.0, "available": 5, "comment": "標準"}
    )
    assert result["technical_score"] == 90.0
    assert result["fundamental_score"] == 50.0
    assert result["score"] == 82.0


def test_materially_weak_fundamentals_are_rejected():
    row = {"score": 95.0, "code": "1234"}
    assert morning_scan.add_fundamental_judgement(
        row, {"score": 30.0, "available": 5, "comment": "弱い"}
    ) is None


def test_missing_fundamentals_do_not_erase_a_valid_technical_signal():
    row = {"score": 80.0, "code": "1234"}
    result = morning_scan.add_fundamental_judgement(
        row, {"score": 50.0, "available": 0, "comment": "取得不可"}
    )
    assert result is not None
    assert result["score"] == 74.0


def test_morning_and_night_meeting_labels_are_distinct():
    morning = morning_scan.build_discord_message(
        [], 786, meeting_label="朝の買い候補", now="2026/09/17 05:10"
    )
    night = morning_scan.build_discord_message(
        [], 786, meeting_label="夜の会議（翌日候補）", now="2026/09/16 18:10"
    )
    assert "Stocknote 朝の買い候補" in morning
    assert "Stocknote 夜の会議（翌日候補）" in night
    assert "候補はありません" in morning
    assert "候補はありません" in night


def test_rotation_message_always_names_sectors_and_stocks():
    sectors = [{
        "業種・テーマ": "小売・内需", "判定": "循環初動",
        "1日%": 0.8, "5日%": -0.4, "初動点": 78,
    }]
    candidates = [{
        "コード": "8267", "銘柄名": "イオン", "業種・テーマ": "小売・内需",
        "扱い": "先回り監視・5日線付近", "1日%": 0.9,
    }]
    message = morning_scan.build_rotation_discord_message(
        sectors, candidates, now="2026/11/12 05:00"
    )
    assert "本日の循環セクター" in message
    assert "小売・内需" in message
    assert "8267 イオン" in message


def test_short_message_shows_watchlist_without_credit_confirmation():
    rows = [{
        "コード": "1234", "銘柄名": "テスト", "空売り区分": "本命監視",
        "空売り総合評価": 67.5, "RSI14": 62, "出来高倍率": 1.8,
        "空売り監視理由": "5日線割れ", "貸借確認": None,
    }]
    message = morning_scan.build_short_discord_message(
        rows, 786, now="2026/11/12 05:00"
    )
    assert "信用売り候補・監視" in message
    assert "1234 テスト" in message
    assert "SBIの売建可否を未確認" in message
