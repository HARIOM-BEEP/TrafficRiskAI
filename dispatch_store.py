"""
Dispatch Store -- Persistent real-time dispatch log
===================================================
Stores every dispatch event in an SQLite table (separate from event_store.py).
Thread-safe via WAL mode. Supports:
  - insert         : create a new dispatch record
  - update_status  : change PENDING -> DISPATCHED -> ON_SCENE -> RESOLVED
  - recent         : last N records (newest first)
  - get            : single record by dispatch_id
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path("data/dispatch.db")
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

STATUSES = ("PENDING", "DISPATCHED", "ON_SCENE", "RESOLVED", "CANCELLED")


class DispatchStore:
    _lock = threading.Lock()

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self._db = str(db_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dispatches (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    dispatch_id     TEXT    UNIQUE NOT NULL,
                    junction_id     TEXT    NOT NULL,
                    junction_name   TEXT,
                    latitude        REAL,
                    longitude       REAL,
                    risk_score      REAL,
                    risk_level      TEXT,
                    officers_sent   INTEGER DEFAULT 1,
                    status          TEXT    DEFAULT 'PENDING',
                    priority        TEXT    DEFAULT 'HIGH',
                    reason          TEXT,
                    notes           TEXT,
                    triggered_by    TEXT    DEFAULT 'auto',
                    created_at      TEXT    NOT NULL,
                    updated_at      TEXT    NOT NULL,
                    resolved_at     TEXT
                )
            """)
            conn.commit()

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        return dict(row)

    def insert(self, record: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT OR IGNORE INTO dispatches
                  (dispatch_id, junction_id, junction_name, latitude, longitude,
                   risk_score, risk_level, officers_sent, status, priority, reason,
                   notes, triggered_by, created_at, updated_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                record["dispatch_id"],
                record["junction_id"],
                record.get("junction_name", ""),
                record.get("latitude"),
                record.get("longitude"),
                record.get("risk_score"),
                record.get("risk_level", "HIGH"),
                record.get("officers_sent", 1),
                record.get("status", "PENDING"),
                record.get("priority", "HIGH"),
                record.get("reason", ""),
                record.get("notes", ""),
                record.get("triggered_by", "auto"),
                now,
                now,
            ))
            conn.commit()
        return self.get(record["dispatch_id"])

    def update_status(self, dispatch_id: str, new_status: str, notes: str = "") -> Optional[Dict[str, Any]]:
        now = datetime.now(timezone.utc).isoformat()
        resolved_at = now if new_status in ("RESOLVED", "CANCELLED") else None
        with self._lock, self._conn() as conn:
            conn.execute("""
                UPDATE dispatches
                   SET status=?,
                       notes=CASE WHEN ? != '' THEN ? ELSE notes END,
                       updated_at=?,
                       resolved_at=COALESCE(?,resolved_at)
                 WHERE dispatch_id=?
            """, (new_status, notes, notes, now, resolved_at, dispatch_id))
            conn.commit()
        return self.get(dispatch_id)

    def get(self, dispatch_id: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM dispatches WHERE dispatch_id=?", (dispatch_id,)
            ).fetchone()
        return self._row_to_dict(row) if row else None

    def recent(self, limit: int = 100, status: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM dispatches"
        params: list = []
        if status:
            query += " WHERE status=?"
            params.append(status.upper())
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_active(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n FROM dispatches WHERE status NOT IN ('RESOLVED','CANCELLED')"
            ).fetchone()
        return row["n"] if row else 0
