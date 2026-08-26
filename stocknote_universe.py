import json
import os
from datetime import datetime

import pandas as pd

DATA_DIR = os.getenv("STOCKNOTE_DATA_DIR", "data")
UNIVERSE_PATH = os.path.join(DATA_DIR, "saved_universe.csv")
META_PATH = os.path.join(DATA_DIR, "saved_universe_meta.json")


def _ensure_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_universe(frame: pd.DataFrame, source_files=None):
    _ensure_dir()
    frame.to_csv(UNIVERSE_PATH, index=False, encoding="utf-8-sig")
    meta = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": list(source_files or []),
        "count": int(len(frame)),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


def load_universe():
    if not os.path.exists(UNIVERSE_PATH):
        return None, None
    frame = pd.read_csv(UNIVERSE_PATH, dtype=str, encoding="utf-8-sig")
    meta = None
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = None
    return frame, meta


def delete_universe():
    removed = False
    for path in (UNIVERSE_PATH, META_PATH):
        if os.path.exists(path):
            os.remove(path)
            removed = True
    return removed
