"""Tests for extended pipeline contracts and backward compatibility."""

from __future__ import annotations

from pipeline_contracts import build_downstream_event, build_risk_event, build_response_event


def test_build_risk_event_contains_new_fields() -> None:
    features = {
        "vehicle_flow_per_hour": 3600.0,
        "vehicle_density_per_100m": 12.0,
        "stopped_vehicle_count": 5,
        "stopped_vehicle_ratio": 0.3,
        "queue_indicator": 0.2,
        "heavy_vehicle_ratio": 0.15,
        "peak_hour_flag": 1,
        "speed_reliable": True,
        "generated_at_utc": "2026-08-17T10:00:00+00:00",
    }
    prediction = {
        "risk_score": 0.535,
        "risk_score_percent": 53.5,
        "risk_level": "MEDIUM",
        "base_risk_score": 0.466,
        "base_risk_score_percent": 46.6,
        "live_risk_delta": 0.069,
        "speed_reliable": True,
        "risk_contributions": {
            "stopped_vehicle_ratio": 2.4,
            "queue_indicator": 3.0,
            "heavy_vehicle_ratio": 1.5,
            "peak_hour_flag": 0.0,
        },
        "risk_reasons": ["Historical base risk: 46.6%. Live adjustment: +6.9%."],
    }
    event = build_risk_event(
        features=features,
        prediction=prediction,
        source_name="Test Junction",
        junction_id="J001",
        camera_id="CAM-01",
        session_id="TEST",
        latitude=21.1458,
        longitude=79.0882,
    )
    assert event["event_type"] == "traffic.risk_prediction.v1"
    assert event["base_risk_score"] == 0.466
    assert event["base_risk_score_percent"] == 46.6
    assert event["live_risk_delta"] == 0.069
    assert event["final_risk_score"] == 0.535
    assert event["stopped_vehicle_count"] == 5
    assert event["stopped_vehicle_ratio"] == 0.3
    assert event["queue_indicator"] == 0.2
    assert event["heavy_vehicle_ratio"] == 0.15
    assert event["peak_hour_flag"] == 1
    assert event["speed_reliable"] is True
    assert event["risk_contributions"] == prediction["risk_contributions"]
    assert event["risk_reasons"] == prediction["risk_reasons"]


def test_build_risk_event_backward_compatible_without_new_fields() -> None:
    features = {"vehicle_flow_per_hour": 1000.0, "vehicle_density_per_100m": 5.0}
    prediction = {"risk_score": 0.3, "risk_level": "LOW"}
    event = build_risk_event(features=features, prediction=prediction, source_name="Test")
    assert event["event_type"] == "traffic.risk_prediction.v1"
    assert event["risk_score"] == 0.3
    assert event["risk_level"] == "LOW"
    assert event.get("base_risk_score") is None
    assert event.get("live_risk_delta") is None


def test_build_downstream_event_preserves_features() -> None:
    features = {"vehicle_count": 10, "heavy_vehicle_ratio": 0.2}
    event = build_downstream_event(features=features, source_name="Test", junction_id="J001")
    assert event["features"] == features
    assert event["junction_id"] == "J001"


def test_build_response_event_inherits_new_fields() -> None:
    risk_event = {
        "event_type": "traffic.risk_prediction.v1",
        "junction_id": "J001",
        "risk_score": 0.535,
        "risk_level": "MEDIUM",
        "base_risk_score": 0.466,
        "live_risk_delta": 0.069,
        "stopped_vehicle_count": 5,
    }
    response = {"alert_status": "MONITORING", "alert_priority": "MEDIUM", "recommended_officers": 1}
    event = build_response_event(risk_event, response)
    assert event["junction_id"] == "J001"
    assert event["risk_score"] == 0.535
    assert event["base_risk_score"] == 0.466
    assert event["live_risk_delta"] == 0.069
    assert event["stopped_vehicle_count"] == 5
