"""Local HTTP API and WebSocket access to persisted TrafficRiskAI demo events.

Dispatch Center endpoints (NEW):
  POST  /dispatch               - Manually or auto trigger a dispatch
  GET   /dispatch               - List recent dispatches (?status=PENDING|DISPATCHED|ON_SCENE|RESOLVED)
  PATCH /dispatch/{dispatch_id} - Update dispatch status (body: {status, notes})
  GET   /dispatch/active/count  - Count active (non-resolved) dispatches
"""

from __future__ import annotations

import asyncio
import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route, WebSocketRoute
from starlette.websockets import WebSocket, WebSocketDisconnect

from event_store import EventStore
from junction_registry import JunctionRegistry
from dispatch_store import DispatchStore

registry         = JunctionRegistry()
store            = EventStore()
dispatch_store   = DispatchStore()
active_websockets: set[WebSocket] = set()

# Risk thresholds that auto-trigger a dispatch
# Only HIGH (>=0.70) and CRITICAL (>=0.85) risk levels trigger auto-dispatch.
# NOTE: vehicle_flow_per_hour derived from short YOLOv8 windows is NOT a real
# hourly count — do NOT use it as a dispatch threshold.
AUTO_DISPATCH_SCORE = 0.70   # risk_score 0–1; >= this triggers auto-dispatch


# ─── helpers ────────────────────────────────────────────────────────────────
async def _broadcast(event: dict) -> None:
    for ws in list(active_websockets):
        try:
            await ws.send_json(event)
        except Exception:
            active_websockets.discard(ws)


def _make_dispatch_event(record: dict) -> dict:
    """Wrap a dispatch DB row as a WebSocket-broadcastable event."""
    return {
        "event_type": "traffic.dispatch.v1",
        "dispatch_id": record["dispatch_id"],
        "junction_id": record["junction_id"],
        "junction_name": record.get("junction_name", ""),
        "latitude":      record.get("latitude"),
        "longitude":     record.get("longitude"),
        "risk_score":    record.get("risk_score"),
        "risk_level":    record.get("risk_level"),
        "officers_sent": record.get("officers_sent", 1),
        "status":        record.get("status"),
        "priority":      record.get("priority"),
        "reason":        record.get("reason", ""),
        "notes":         record.get("notes", ""),
        "triggered_by":  record.get("triggered_by", "auto"),
        "created_at":    record.get("created_at"),
        "updated_at":    record.get("updated_at"),
        "resolved_at":   record.get("resolved_at"),
    }


# ─── existing endpoints ──────────────────────────────────────────────────────
async def health(request: Request) -> JSONResponse:
    return JSONResponse({
        "status": "ok",
        "event_count": store.count(),
        "active_dispatches": dispatch_store.count_active(),
        "mode": "local demo",
    })


async def list_events(request: Request) -> JSONResponse:
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            return JSONResponse({"detail": "Request body must be valid JSON."}, status_code=400)
        if not isinstance(body, (dict, list)):
            return JSONResponse({"detail": "Request body must be an event object or a list of events."}, status_code=400)
        events_to_append = body if isinstance(body, list) else [body]
        if not all(isinstance(event, dict) for event in events_to_append):
            return JSONResponse({"detail": "Every submitted event must be a JSON object."}, status_code=400)
        db_events = []
        for event in events_to_append:
            db_event, created = store.append_with_record(event)
            db_events.append(db_event)
            if created:
                await _broadcast(db_event)

            # ── Auto-dispatch check ─────────────────────────────────────────
            # Only fire on risk_prediction events (not redeployment) to avoid
            # double-dispatching the same junction event.
            etype = event.get("event_type", "")
            if etype == "traffic.risk_prediction.v1":
                # Prefer flat root keys (set by build_risk_event); fall back to
                # nested payload.prediction for older events.
                payload_pred = event.get("payload", {}).get("prediction", {})
                risk_score = float(
                    event.get("risk_score")
                    or payload_pred.get("risk_score")
                    or 0
                )
                risk_level = str(
                    event.get("risk_level")
                    or payload_pred.get("risk_level")
                    or "LOW"
                ).upper()
                junction_id = (
                    event.get("junction_id")
                    or event.get("payload", {}).get("junction_id")
                    or "UNKNOWN"
                )

                # Dispatch only on genuinely elevated risk.
                # Flow threshold is intentionally removed: YOLOv8 window-derived
                # flow values cannot be compared to real hourly vehicle counts.
                needs_dispatch = (
                    risk_score >= AUTO_DISPATCH_SCORE
                    or risk_level in ("HIGH", "CRITICAL")
                )
                if needs_dispatch:
                    dispatch_id = f"DSP-{junction_id}-{int(datetime.now(timezone.utc).timestamp())}"
                    existing = dispatch_store.get(dispatch_id)
                    if not existing:
                        junc = registry.get_junction(junction_id)
                        # Scale officers: CRITICAL=4, HIGH=2
                        officers = (
                            4 if risk_level == "CRITICAL"
                            else 2
                        )
                        priority = (
                            "URGENT" if risk_level == "CRITICAL"
                            else "HIGH"
                        )
                        rec = dispatch_store.insert({
                            "dispatch_id":   dispatch_id,
                            "junction_id":   junc["junction_id"] if junc else junction_id,
                            "junction_name": junc["location"]    if junc else junction_id,
                            "latitude":      junc["latitude"]    if junc else None,
                            "longitude":     junc["longitude"]   if junc else None,
                            "risk_score":    risk_score,
                            "risk_level":    risk_level,
                            "officers_sent": officers,
                            "status":        "PENDING",
                            "priority":      priority,
                            "reason":        f"Auto-triggered: risk_score={risk_score:.2f}, level={risk_level}",
                            "triggered_by":  "pipeline_auto",
                        })
                        if rec:
                            await _broadcast(_make_dispatch_event(rec))

        return JSONResponse(db_events if isinstance(body, list) else db_events[0])

    # GET
    limit      = max(1, min(int(request.query_params.get("limit", "50")), 500))
    event_type = request.query_params.get("event_type")
    session_id = request.query_params.get("session_id")
    events     = store.recent(limit=limit, event_type=event_type)
    if session_id:
        events = [e for e in events if
                  e.get("session_id") == session_id
                  or e.get("payload", {}).get("session_id") == session_id]
    return JSONResponse(events)


async def latest_event(request: Request) -> JSONResponse:
    event = store.latest()
    if event is None:
        return JSONResponse({"detail": "No pipeline events have been stored yet"}, status_code=404)
    return JSONResponse(event)


async def clear_events(request: Request) -> JSONResponse:
    session_id = request.query_params.get("session_id")
    store.clear(session_id=session_id)
    clear_event = {
        "id": 0,
        "event_id":   f"CLEAR-{datetime.now(timezone.utc).timestamp()}",
        "event_type": "traffic.session_clear.v1",
        "source_name": "System",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "junction_id": None, "camera_id": None,
        "session_id":  session_id,
        "payload":     {"session_id": session_id},
    }
    await _broadcast(clear_event)
    return JSONResponse({"status": "cleared", "session_id": session_id})


async def list_junctions(request: Request) -> JSONResponse:
    category = request.query_params.get("risk_category", "ALL").upper()
    all_junctions = registry.list_all()
    if category != "ALL":
        all_junctions = [j for j in all_junctions if j.get("risk_category", "").upper() == category]
    return JSONResponse(all_junctions)


async def list_junction_registry(request: Request) -> JSONResponse:
    return JSONResponse(registry.list_all())


# ─── Dispatch Center endpoints ───────────────────────────────────────────────
async def dispatch_list_or_create(request: Request) -> JSONResponse:
    """GET /dispatch  or  POST /dispatch"""
    if request.method == "POST":
        body = await request.json()
        junction_id = body.get("junction_id")
        junc = registry.get_junction(junction_id) if junction_id else None
        
        std_j_id = junc["junction_id"] if junc else (junction_id or "UNK")
        dispatch_id = body.get("dispatch_id") or f"DSP-{std_j_id}-{uuid.uuid4().hex[:6].upper()}"
        
        body["dispatch_id"] = dispatch_id
        body["junction_id"] = std_j_id
        if junc:
            body["junction_name"] = junc["location"]
            body["latitude"] = junc["latitude"]
            body["longitude"] = junc["longitude"]
            
        body.setdefault("triggered_by", "manual")
        rec = dispatch_store.insert(body)
        if rec:
            await _broadcast(_make_dispatch_event(rec))
            return JSONResponse(rec, status_code=201)
        return JSONResponse({"detail": "Dispatch already exists"}, status_code=409)

    # GET
    limit  = max(1, min(int(request.query_params.get("limit", "100")), 500))
    status = request.query_params.get("status")
    rows   = dispatch_store.recent(limit=limit, status=status)
    return JSONResponse(rows)


async def dispatch_update(request: Request) -> JSONResponse:
    """PATCH /dispatch/{dispatch_id}"""
    dispatch_id = request.path_params["dispatch_id"]
    body        = await request.json()
    new_status  = body.get("status", "").upper()
    notes       = body.get("notes", "")
    if new_status not in ("PENDING", "DISPATCHED", "ON_SCENE", "RESOLVED", "CANCELLED"):
        return JSONResponse({"detail": f"Invalid status: {new_status}"}, status_code=400)
    rec = dispatch_store.update_status(dispatch_id, new_status, notes)
    if not rec:
        return JSONResponse({"detail": "Dispatch not found"}, status_code=404)
    await _broadcast(_make_dispatch_event(rec))
    return JSONResponse(rec)


async def dispatch_active_count(request: Request) -> JSONResponse:
    return JSONResponse({"active_dispatches": dispatch_store.count_active()})

# ─── Risk summary endpoint ────────────────────────────────────────────
async def risk_summary(request: Request) -> JSONResponse:
    """GET /events/risk_summary

    Returns a dict keyed by junction_id with the *latest* risk_prediction
    values for each junction.  Useful for the Live Risk Map marker overlay.

    Query params:
      limit   int  Max risk-prediction events to scan (default 500, max 2000)
    """
    limit = max(1, min(int(request.query_params.get("limit", "500")), 2000))
    events = store.recent(limit=limit, event_type="traffic.risk_prediction.v1")

    # events are ordered newest-first; first occurrence per junction_id wins.
    summary: dict[str, dict] = {}
    for ev in events:
        j_id = ev.get("junction_id") or ev.get("payload", {}).get("junction_id")
        if not j_id or j_id in summary:
            continue
        payload = ev.get("payload", {})
        pred = payload.get("prediction", {})
        feats = payload.get("features", {})
        summary[j_id] = {
            "junction_id":     j_id,
            "risk_score":      ev.get("risk_score") or pred.get("risk_score"),
            "risk_level":      ev.get("risk_level") or pred.get("risk_level"),
            "timestamp":       ev.get("timestamp") or ev.get("recorded_at"),
            "traffic_volume":  ev.get("traffic_volume") or feats.get("vehicle_flow_per_hour"),
            "average_speed":   ev.get("average_speed") or feats.get("average_speed_kmh"),
            "congestion_level": ev.get("congestion_level") or feats.get("vehicle_density_per_100m"),
            "camera_id":       ev.get("camera_id"),
            "session_id":      ev.get("session_id"),
            "event_count":     1,
        }

    # Count total events per junction across the scan window
    for ev in events:
        j_id = ev.get("junction_id") or ev.get("payload", {}).get("junction_id")
        if j_id and j_id in summary:
            summary[j_id]["event_count"] = summary[j_id].get("event_count", 0) + 1

    return JSONResponse(list(summary.values()))


# ─── WebSocket ──────────────────────────────────────────────────────────────────────────────
async def event_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        last_id = max(0, int(websocket.query_params.get("after_id", "0")))
    except (TypeError, ValueError):
        last_id = 0

    # The API can receive HTTP events and Streamlit can fall back to writing
    # SQLite directly. Polling the durable cursor means both paths reach every
    # connected dashboard, including after a short network interruption.
    for event in store.after_id(last_id, limit=100):
        await websocket.send_json(event)
        last_id = int(event["id"])
    # Also send recent dispatches on connect
    for rec in dispatch_store.recent(limit=30):
        try:
            await websocket.send_json(_make_dispatch_event(rec))
        except Exception:
            pass
    try:
        while True:
            # receive_text with a timeout lets us notice disconnects while the
            # server independently delivers records written by any process.
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=0.4)
            except TimeoutError:
                for event in store.after_id(last_id, limit=100):
                    await websocket.send_json(event)
                    last_id = int(event["id"])
    except WebSocketDisconnect:
        pass
    finally:
        active_websockets.discard(websocket)


# ─── App ─────────────────────────────────────────────────────────────────────
app = Starlette(
    routes=[
        Route("/health",                  health),
        Route("/events",                  list_events,              methods=["GET", "POST"]),
        Route("/events/risk_summary",   risk_summary),
        Route("/events/clear",            clear_events,             methods=["POST"]),
        Route("/events/latest",           latest_event),
        Route("/junctions",               list_junctions),
        Route("/junctions/registry",      list_junction_registry),
        Route("/dispatch",                dispatch_list_or_create,  methods=["GET", "POST"]),
        Route("/dispatch/active/count",   dispatch_active_count),
        Route("/dispatch/{dispatch_id}",  dispatch_update,          methods=["PATCH"]),
        WebSocketRoute("/ws/events",      event_stream),
    ]
)
app.add_middleware(
    CORSMiddleware,
    # Localhost plus common private-LAN Vite origins. This keeps the hackathon
    # demo usable from another laptop without opening the API to public sites.
    allow_origin_regex=r"^https?://(?:localhost|127\.0\.0\.1|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3}):(5173|4173)$",
    allow_methods=["*"],
    allow_headers=["*"],
)
