"""Smoke checks for all rule-policy bands in the demo response engine."""

from response_engine import ResponseEngine


def _event(level: str, score: float) -> dict:
    return {
        "event_type": "traffic.risk_prediction.v1",
        "source_name": "Junction smoke test",
        "features": {"traffic_state": "high", "vehicle_density_per_100m": 16, "vehicle_flow_per_hour": 3600},
        "prediction": {"risk_level": level, "risk_score": score, "explanation": ["Representative test input"]},
    }


def test_response_policy_bands() -> None:
    engine = ResponseEngine()
    expected = {"LOW": ("MONITORING", 0), "MEDIUM": ("MONITORING", 1), "HIGH": ("OPEN", 2), "CRITICAL": ("URGENT", 4)}
    for level, score in [("LOW", 0.20), ("MEDIUM", 0.50), ("HIGH", 0.75), ("CRITICAL", 0.91)]:
        response_event = engine.process(_event(level, score))
        assert response_event["event_type"] == "traffic.response_recommendation.v1"
        response = response_event["response"]
        assert response["risk_level"] == level
        assert (response["alert_status"], response["recommended_officers"]) == expected[level]
        assert "No real police dispatch" in response["demo_notice"]
