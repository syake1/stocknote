import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import stocknote_tracking as tracking


class TrackingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.paths = patch.multiple(
            tracking,
            ACTIVE_PATH=base / "active.json",
            HISTORY_PATH=base / "history.json",
            NOTICE_PATH=base / "notice.json",
        )
        self.paths.start()
        self.now = datetime(2026, 8, 1, 9, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.paths.stop()
        self.tmp.cleanup()

    def row(self, code, score=60, price=100):
        return {"code": code, "name": f"N{code}", "price": price, "score": score, "rsi": 40}

    def test_zero_scan_does_not_remove_three_candidates(self):
        tracking.merge_new_candidates([self.row(str(i)) for i in range(1, 4)], self.now)
        tracking.merge_new_candidates([], self.now + timedelta(days=1))
        self.assertEqual(3, len(tracking.load_active()))

    def test_new_candidate_is_added_and_duplicate_is_upserted(self):
        tracking.merge_new_candidates([self.row("1"), self.row("2"), self.row("3")], self.now)
        tracking.merge_new_candidates([self.row("2", price=110), self.row("4")], self.now + timedelta(days=1))
        active = tracking.load_active()
        self.assertEqual(4, len(active))
        self.assertEqual(110, next(x for x in active if x["code"] == "2")["current_price"])

    def test_update_changes_metrics_and_state(self):
        tracking.merge_new_candidates([self.row("1")], self.now)
        events = tracking.update_active([self.row("1", score=80, price=105)], self.now + timedelta(minutes=15))
        item = tracking.load_active()[0]
        self.assertEqual(105, item["current_price"])
        self.assertEqual("買い条件到達", item["status"])
        self.assertTrue(events)

    def test_missing_update_keeps_candidate(self):
        tracking.merge_new_candidates([self.row("1")], self.now)
        tracking.update_active([], self.now + timedelta(minutes=15))
        self.assertEqual(1, len(tracking.load_active()))

    def test_notification_deduplication(self):
        event = [{"code": "1", "to": "買い条件接近"}]
        self.assertEqual(1, len(tracking.filter_new_notifications(event)))
        self.assertEqual([], tracking.filter_new_notifications(event))

    def test_fourteen_days_archives_instead_of_deleting(self):
        tracking.merge_new_candidates([self.row("1")], self.now)
        tracking.update_active([self.row("1")], self.now + timedelta(days=14))
        self.assertEqual([], tracking.load_active())
        self.assertEqual("監視終了", tracking.load_history()[0]["status"])

    def test_states_cover_approaching_and_worsening(self):
        self.assertEqual("買い条件接近", tracking.classify({"score": 68}))
        self.assertEqual("条件悪化", tracking.classify({"score": 35}))
        self.assertEqual("見送り", tracking.classify({"score": 20}))


if __name__ == "__main__":
    unittest.main()
