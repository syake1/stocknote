"""Persist a sector-rotation pick in Stocknote's existing tracking table."""

from __future__ import annotations

import sqlite3
from datetime import datetime


SOURCE = "市場環境社員・循環買い"


def add_sector_candidate(db_path, code, name, sector, signal_date, signal_price, now=None):
    """Add one daily sector pick, returning False when it is already registered."""
    price = float(signal_price)
    if not str(code).strip() or price <= 0:
        raise ValueError("銘柄コードと株価が必要です。")
    created_at = (now or datetime.now()).isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT NOT NULL,
                name TEXT,
                side TEXT NOT NULL DEFAULT '買い',
                signal_date TEXT NOT NULL,
                signal_price REAL NOT NULL,
                source TEXT,
                note TEXT,
                created_at TEXT NOT NULL
            )
        """)
        duplicate = con.execute(
            "SELECT 1 FROM tracking WHERE code=? AND signal_date=? AND source=? LIMIT 1",
            (str(code), str(signal_date), SOURCE),
        ).fetchone()
        if duplicate:
            return False
        con.execute(
            "INSERT INTO tracking(code,name,side,signal_date,signal_price,source,note,created_at) "
            "VALUES(?,?,?,?,?,?,?,?)",
            (str(code), str(name or code), "買い", str(signal_date), price,
             SOURCE, f"{sector}の循環買い", created_at),
        )
        con.commit()
    return True
