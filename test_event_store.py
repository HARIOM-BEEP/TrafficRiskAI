"""Unit-like checks for local persistence and API route registration."""

from tempfile import TemporaryDirectory
from pathlib import Path

from api_server import app
from event_store import EventStore


def test_event_store_round_trip() -> None:
    event = {"event_type": "traffic.feature_window.v1", "source_name": "Test Junction", "features": {"traffic_state": "light"}}
    with TemporaryDirectory() as directory:
        store = EventStore(Path(directory) / "events.db")
        first_id = store.append(event)
        assert first_id == store.append(event)
        assert store.count() == 1
        stored = store.latest()
        assert stored is not None
        assert stored["payload"] == event


def test_api_routes_are_available() -> None:
    paths = {route.path for route in app.routes}
    assert {"/health", "/events", "/events/latest", "/junctions", "/ws/events"}.issubset(paths)
