"""Contracts reserved for the stages that follow vision processing."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PipelineStage:
    name: str
    status: str
    accepts: str
    produces: str


UPCOMING_PIPELINE: tuple[PipelineStage, ...] = (
    PipelineStage("Feature aggregator", "Ready", "Tracked vehicle observations", "Windowed traffic feature vector"),
    PipelineStage("Risk classifier", "Ready", "Traffic feature vector", "Risk score and level"),
    PipelineStage("Response engine", "Ready", "Risk event", "Recommended alert/action"),
    PipelineStage("Storage & API", "Ready", "Events and predictions", "Dashboard, API, and live updates"),
)


def build_downstream_event(
    features: dict[str, Any],
    source_name: str,
    junction_id: str | None = None,
    camera_id: str | None = None,
    session_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Stable hand-off shape for future queues, APIs, or risk models."""
    event = {
        "event_type": "traffic.feature_window.v1",
        "source_name": source_name,
        "features": features,
    }
    if junction_id is not None:
        event["junction_id"] = junction_id
    if camera_id is not None:
        event["camera_id"] = camera_id
    if session_id is not None:
        event["session_id"] = session_id
    if latitude is not None:
        event["latitude"] = latitude
    if longitude is not None:
        event["longitude"] = longitude
    return event


def build_risk_event(
    features: dict[str, Any],
    prediction: dict[str, Any],
    source_name: str,
    junction_id: str | None = None,
    camera_id: str | None = None,
    session_id: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Stable hand-off from the classifier to the future response engine."""
    from datetime import datetime, timezone
    event = {
        "event_type": "traffic.risk_prediction.v1",
        "source_name": source_name,
        "features": features,
        "prediction": prediction,
        "risk_score": prediction.get("risk_score"),
        "risk_level": prediction.get("risk_level"),
        "timestamp": features.get("generated_at_utc") or datetime.now(timezone.utc).isoformat(),
        "traffic_volume": features.get("vehicle_flow_per_hour"),
        "average_speed": features.get("average_speed_kmh"),
        "congestion_level": features.get("vehicle_density_per_100m"),
        "base_risk_score": prediction.get("base_risk_score"),
        "base_risk_score_percent": prediction.get("base_risk_score_percent"),
        "live_risk_delta": prediction.get("live_risk_delta"),
        "final_risk_score": prediction.get("risk_score"),
        "stopped_vehicle_count": features.get("stopped_vehicle_count"),
        "stopped_vehicle_ratio": features.get("stopped_vehicle_ratio"),
        "queue_indicator": features.get("queue_indicator"),
        "heavy_vehicle_ratio": features.get("heavy_vehicle_ratio"),
        "peak_hour_flag": features.get("peak_hour_flag"),
        "speed_reliable": features.get("speed_reliable"),
        "risk_contributions": prediction.get("risk_contributions"),
        "risk_reasons": prediction.get("risk_reasons"),
    }
    if junction_id is not None:
        event["junction_id"] = junction_id
    if camera_id is not None:
        event["camera_id"] = camera_id
    if session_id is not None:
        event["session_id"] = session_id
    if latitude is not None:
        event["latitude"] = latitude
    if longitude is not None:
        event["longitude"] = longitude
    return event


def build_response_event(
    risk_event: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any]:
    """Stable hand-off from the response engine to future delivery layers."""
    if risk_event.get("event_type") != "traffic.risk_prediction.v1":
        raise ValueError("Response events must be built from traffic.risk_prediction.v1 events")

    event = {
        "event_type": "traffic.response_recommendation.v1",
        "source_name": risk_event.get("source_name", "Unknown source"),
        "risk_event_type": risk_event["event_type"],
        "risk_event": risk_event,
        "response": response,
    }
    for key in ["junction_id", "camera_id", "session_id", "latitude", "longitude", "risk_score", "risk_level", "timestamp", "traffic_volume", "average_speed", "congestion_level", "base_risk_score", "base_risk_score_percent", "live_risk_delta", "final_risk_score", "stopped_vehicle_count", "stopped_vehicle_ratio", "queue_indicator", "heavy_vehicle_ratio", "peak_hour_flag", "speed_reliable", "risk_contributions", "risk_reasons"]:
        if key in risk_event:
            event[key] = risk_event[key]
    return event


def build_redeployment_event(
    risk_event: dict[str, Any],
    response_event: dict[str, Any],
    redeployment: dict[str, Any],
) -> dict[str, Any]:
    """Stable hand-off for the redeployment engine recommendations."""
    event = {
        "event_type": "traffic.redeployment_recommendation.v1",
        "source_name": risk_event.get("source_name", "Unknown source"),
        "risk_event": risk_event,
        "response_event": response_event,
        "redeployment": redeployment,
    }
    for key in ["junction_id", "camera_id", "session_id", "latitude", "longitude", "risk_score", "risk_level", "timestamp", "traffic_volume", "average_speed", "congestion_level", "base_risk_score", "base_risk_score_percent", "live_risk_delta", "final_risk_score", "stopped_vehicle_count", "stopped_vehicle_ratio", "queue_indicator", "heavy_vehicle_ratio", "peak_hour_flag", "speed_reliable", "risk_contributions", "risk_reasons"]:
        if key in risk_event:
            event[key] = risk_event[key]
    for key in ["junction_id", "camera_id", "session_id", "latitude", "longitude", "risk_score", "risk_level", "recommended_additional_officers", "priority", "action", "reason", "human_approval_required", "demo_notice", "decision_trace", "nearby_redeployments", "timestamp"]:
        if key in redeployment:
            event[key] = redeployment[key]
    return event


def build_manual_override_event(
    junction_id: str,
    camera_id: str,
    session_id: str,
    original_risk_level: str,
    override_action: str,
    override_officers: int,
    override_reason: str,
    operator: str = "demo_operator",
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Build a manual override event for deployment decisions."""
    from datetime import datetime, timezone
    return {
        "event_type": "traffic.manual_override.v1",
        "source_name": "Manual Override",
        "junction_id": junction_id,
        "camera_id": camera_id,
        "session_id": session_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "original_risk_level": original_risk_level,
        "override_action": override_action,
        "override_officers": override_officers,
        "override_reason": override_reason,
        "operator": operator,
        "latitude": latitude,
        "longitude": longitude,
    }


