"""Small SQLite event store for the TrafficRiskAI demo pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Any
from contextlib import contextmanager


DEFAULT_DB_PATH = Path("data/trafficrisk_events.db")


class EventStore:
    """Persists versioned pipeline events without tying the app to a cloud service."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS pipeline_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            # Dynamically add columns if they don't exist to handle existing databases cleanly
            for column in ["junction_id", "camera_id", "session_id"]:
                try:
                    connection.execute(f"ALTER TABLE pipeline_events ADD COLUMN {column} TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists

            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pipeline_events_type ON pipeline_events(event_type)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_pipeline_events_session ON pipeline_events(session_id)"
            )

    def append(self, event: dict[str, Any]) -> str:
        """Persist an event once and return its deterministic identifier."""
        event_id, _ = self._append(event)
        return event_id

    def append_with_record(self, event: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """Persist an event and return its API-ready record plus insert status.

        Reading ``recent(limit=1)`` after a write is unsafe when two cameras
        submit events at the same time: it can return the other camera's event.
        This method is used by the API so a WebSocket broadcast always matches
        the event just accepted.
        """
        event_id, created = self._append(event)
        record = self.get(event_id)
        if record is None:  # Defensive guard; the insert transaction succeeded.
            raise RuntimeError(f"Stored event {event_id} could not be read")
        return record, created

    def _append(self, event: dict[str, Any]) -> tuple[str, bool]:
        """Persist an event once and report whether this call inserted it."""
        event_type = str(event.get("event_type", "unknown"))
        source_name = str(event.get("source_name", "Unknown source"))
        junction_id = event.get("junction_id")
        camera_id = event.get("camera_id")
        session_id = event.get("session_id")

        payload = json.dumps(event, sort_keys=True, default=str, separators=(",", ":"))
        event_id = self._event_id(event, payload)
        with self._connection() as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO pipeline_events
                    (event_id, event_type, source_name, junction_id, camera_id, session_id, recorded_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (event_id, event_type, source_name, junction_id, camera_id, session_id, datetime.now(timezone.utc).isoformat(), payload),
            )
        return event_id, cursor.rowcount == 1

    def append_many(self, events: list[dict[str, Any]]) -> list[str]:
        return [self.append(event) for event in events]

    def recent(self, limit: int = 50, event_type: str | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        query = "SELECT id, event_id, event_type, source_name, junction_id, camera_id, session_id, recorded_at, payload_json FROM pipeline_events"
        parameters: list[Any] = []
        if event_type:
            query += " WHERE event_type = ?"
            parameters.append(event_type)
        query += " ORDER BY id DESC LIMIT ?"
        parameters.append(limit)
        with self._connection() as connection:
            rows = connection.execute(query, parameters).fetchall()
        return [self._decode(row) for row in rows]

    def latest(self) -> dict[str, Any] | None:
        events = self.recent(limit=1)
        return events[0] if events else None

    def get(self, event_id: str) -> dict[str, Any] | None:
        """Return one decoded event by its deterministic identifier."""
        with self._connection() as connection:
            row = connection.execute(
                "SELECT id, event_id, event_type, source_name, junction_id, camera_id, session_id, recorded_at, payload_json "
                "FROM pipeline_events WHERE event_id = ?",
                (event_id,),
            ).fetchone()
        return self._decode(row) if row else None

    def after_id(self, last_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """Read events newer than ``last_id`` in chronological order.

        This is intentionally cursor-based instead of repeatedly fetching the
        newest N records, which prevents missed or duplicate events during
        normal WebSocket reconnects and makes SQLite polling inexpensive.
        """
        limit = max(1, min(int(limit), 500))
        with self._connection() as connection:
            rows = connection.execute(
                "SELECT id, event_id, event_type, source_name, junction_id, camera_id, session_id, recorded_at, payload_json "
                "FROM pipeline_events WHERE id > ? ORDER BY id ASC LIMIT ?",
                (max(0, int(last_id)), limit),
            ).fetchall()
        return [self._decode(row) for row in rows]

    def count(self) -> int:
        with self._connection() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM pipeline_events").fetchone()[0])

    def clear(self, session_id: str | None = None) -> None:
        """Clear events from SQLite. If session_id is provided, clear only that session."""
        with self._connection() as connection:
            if session_id:
                connection.execute("DELETE FROM pipeline_events WHERE session_id = ?", (session_id,))
            else:
                connection.execute("DELETE FROM pipeline_events")

    @staticmethod
    def _event_id(event: dict[str, Any], payload: str) -> str:
        response = event.get("response", {})
        if isinstance(response, dict) and response.get("alert_id"):
            return str(response["alert_id"])
        return f"EVT-{sha256(payload.encode('utf-8')).hexdigest()[:16].upper()}"

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        """Deserialise a DB row into a dict the frontend can consume directly.

        Key design rule: the React client reads risk fields (prediction,
        features, latitude, longitude, risk_score …) at the *root* of each
        event dict. The full original payload blob is also kept for
        backward-compat. We hoist the most important scalar and dict fields
        from the payload to root-level so that both REST /events responses
        and WebSocket broadcasts share a consistent, flat schema.
        """
        keys = row.keys()
        payload = json.loads(row["payload_json"])

        # Scalar fields that should be available at root (hoisted from payload)
        HOIST_SCALARS = [
            "latitude", "longitude",
            "risk_score", "risk_level",
            "timestamp", "traffic_volume",
            "average_speed", "congestion_level",
        ]
        # Dict fields that the React buildJunctions / buildRedeployments helpers
        # read directly (e.g.  e.prediction?.risk_score, e.features?.vehicle_flow_per_hour)
        HOIST_DICTS = ["prediction", "features", "response", "redeployment"]

        base: dict[str, Any] = {
            "id":          row["id"],
            "event_id":    row["event_id"],
            "event_type":  row["event_type"],
            "source_name": row["source_name"],
            "recorded_at": row["recorded_at"],
            "junction_id": row["junction_id"] if "junction_id" in keys else None,
            "camera_id":   row["camera_id"]   if "camera_id"   in keys else None,
            "session_id":  row["session_id"]  if "session_id"  in keys else None,
            "payload":     payload,
        }

        # Hoist scalar keys — only if not already present (or is None)
        for key in HOIST_SCALARS:
            if base.get(key) is None:
                val = payload.get(key)
                if val is not None:
                    base[key] = val

        # Hoist dict keys
        for key in HOIST_DICTS:
            if key not in base:
                val = payload.get(key)
                if val is not None:
                    base[key] = val

        return base
