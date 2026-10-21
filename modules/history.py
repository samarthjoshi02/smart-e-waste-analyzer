"""Local Analysis History Persistence Module (SQLite).

Problem Statement 2 (PS 2): Intelligent E-Waste Circuit Component Classifier & Material Analyzer.
Provides:
  1. Lightweight SQLite persistence for past analysis runs (timestamp, metrics, inventory summary, valuations).
  2. Deduplication to prevent saving duplicates during Streamlit reruns.
  3. Safe history retrieval, reopening, and clearing.
  4. Resilient error handling (never crashes the app if the database is inaccessible).
"""

import json
import os
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional

# Default database location inside app_data directory
DEFAULT_DB_DIR = Path(__file__).resolve().parent.parent / "app_data"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "history.db"


def get_db_path(custom_path: Optional[str] = None) -> Path:
    """Resolves and returns the history SQLite database path."""
    if custom_path:
        p = Path(custom_path)
    else:
        env_val = os.environ.get("HISTORY_DB_PATH", "").strip()
        p = Path(env_val) if env_val else DEFAULT_DB_PATH

    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def init_history_db(db_path: Optional[Path] = None) -> bool:
    """Initializes the SQLite history table if it does not already exist."""
    path = db_path or get_db_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analysis_uuid TEXT UNIQUE,
                    timestamp TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    total_components INTEGER NOT NULL,
                    categories_count INTEGER NOT NULL,
                    top_confidence REAL,
                    top_component TEXT,
                    inference_time_ms REAL,
                    conf_threshold REAL,
                    counts_json TEXT,
                    valuation_typical_usd REAL,
                    valuation_low_usd REAL,
                    valuation_high_usd REAL
                )
                """
            )
            conn.commit()
        return True
    except Exception as e:
        print(f"[Warning] Could not initialize history database at {path}: {e}")
        return False


def save_analysis_record(record: Dict[str, Any], db_path: Optional[Path] = None) -> bool:
    """Saves a single analysis record to the history database.

    Deduplicates using analysis_uuid to prevent duplicate inserts during Streamlit reruns.

    Args:
        record: Dictionary containing analysis metrics, counts, and valuation.
        db_path: Optional custom path to SQLite file.

    Returns:
        True if inserted, False if duplicate or failed.
    """
    path = db_path or get_db_path()
    init_history_db(path)

    uuid_str = record.get("analysis_uuid", "")
    counts_json_str = json.dumps(record.get("counts_by_class", {}))

    try:
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT OR IGNORE INTO analysis_history (
                    analysis_uuid, timestamp, filename, total_components,
                    categories_count, top_confidence, top_component,
                    inference_time_ms, conf_threshold, counts_json,
                    valuation_typical_usd, valuation_low_usd, valuation_high_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid_str,
                    record.get("timestamp", ""),
                    record.get("filename", "unknown.jpg"),
                    int(record.get("total_components", 0)),
                    int(record.get("categories_count", 0)),
                    float(record.get("top_confidence", 0.0)),
                    record.get("top_component", "None"),
                    float(record.get("inference_time_ms", 0.0)),
                    float(record.get("conf_threshold", 0.30)),
                    counts_json_str,
                    float(record.get("valuation_typical_usd", 0.0)),
                    float(record.get("valuation_low_usd", 0.0)),
                    float(record.get("valuation_high_usd", 0.0)),
                ),
            )
            conn.commit()
            return cursor.rowcount > 0
    except Exception as e:
        print(f"[Warning] Could not save history record to {path}: {e}")
        return False


def get_history_records(limit: int = 50, db_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Retrieves recent analysis records sorted by most recent first."""
    path = db_path or get_db_path()
    if not path.exists():
        return []

    try:
        with sqlite3.connect(path, timeout=5.0) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT * FROM analysis_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            results = []
            for r in rows:
                row_dict = dict(r)
                if row_dict.get("counts_json"):
                    try:
                        row_dict["counts_by_class"] = json.loads(row_dict["counts_json"])
                    except Exception:
                        row_dict["counts_by_class"] = {}
                results.append(row_dict)
                row_dict["valuation_typical_inr"] = round(float(row_dict.get("valuation_typical_usd") or 0.0) * 85.0, 2)
                row_dict["valuation_low_inr"] = round(float(row_dict.get("valuation_low_usd") or 0.0) * 85.0, 2)
                row_dict["valuation_high_inr"] = round(float(row_dict.get("valuation_high_usd") or 0.0) * 85.0, 2)
            return results
    except Exception as e:
        print(f"[Warning] Could not read history records from {path}: {e}")
        return []


def clear_all_history(db_path: Optional[Path] = None) -> bool:
    """Clears all records from the analysis history table."""
    path = db_path or get_db_path()
    if not path.exists():
        return True

    try:
        with sqlite3.connect(path, timeout=5.0) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM analysis_history")
            conn.commit()
        return True
    except Exception as e:
        print(f"[Warning] Could not clear history table in {path}: {e}")
        return False
