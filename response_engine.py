"""Transparent, demo-only response recommendations for traffic-risk events."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from pipeline_contracts import build_response_event


@dataclass(frozen=True)
class ResponseRecommendation:
    alert_id: str
    timestamp: str
    source_name: str
    risk_score: float
    risk_level: str
    alert_priority: str
    alert_status: str
    recommended_action: str
    recommended_officers: int
    message: str
    reasons: list[str]
    decision_trace: list[str]
    demo_notice: str = "Demo-only, rule-based recommendation. No real police dispatch is occurring."

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResponseEngine:
    """Converts traffic.risk_prediction.v1 events into explainable recommendations."""

    _RULES = {
        "LOW": ("LOW", "MONITORING", "Continue normal monitoring; no new deployment is recommended.", 0),
        "MEDIUM": ("MEDIUM", "MONITORING", "Increase patrol visibility and continue monitoring traffic conditions.", 1),
        "HIGH": ("HIGH", "OPEN", "Create a demo alert, increase monitoring, and recommend traffic officer deployment.", 2),
        "CRITICAL": ("CRITICAL", "URGENT", "Create an urgent demo alert; recommend immediate traffic police deployment and escalation.", 4),
    }

    def recommend(self, risk_event: dict[str, Any]) -> ResponseRecommendation:
        if risk_event.get("event_type") != "traffic.risk_prediction.v1":
            raise ValueError("ResponseEngine only accepts traffic.risk_prediction.v1 events")

        prediction = risk_event.get("prediction", {})
        features = risk_event.get("features", {})
        source_name = str(risk_event.get("source_name", "Unknown source"))
        risk_score = min(max(float(prediction.get("risk_score", 0.0)), 0.0), 1.0)
        risk_level = str(prediction.get("risk_level", "LOW")).upper()
        if risk_level not in self._RULES:
            risk_level = self._level_from_score(risk_score)

        priority, status, action, officers = self._RULES[risk_level]
        density = float(features.get("vehicle_density_per_100m", 0) or 0)
        flow = float(features.get("vehicle_flow_per_hour", 0) or 0)
        traffic_state = str(features.get("traffic_state", "unknown")).lower()
        reasons = list(prediction.get("explanation", []))
        trace = [
            f"Risk score {risk_score:.2f} maps to {risk_level} response policy.",
            f"Observed traffic state: {traffic_state}; density: {density:.1f} vehicles / 100 m; flow: {flow:.0f} vehicles / hour.",
        ]
        if traffic_state == "high" or density >= 15:
            trace.append("High density reinforces active monitoring and deployment recommendations.")
        if flow >= 3500:
            trace.append("High vehicle flow reinforces the recommendation to protect junction capacity.")
        if not reasons:
            reasons.append("Rule-based recommendation based on the current risk score and traffic features")

        timestamp = datetime.now(timezone.utc).isoformat()
        stable_key = f"{source_name}|{features.get('window_started_s', '')}|{features.get('window_ended_s', '')}|{risk_score:.3f}"
        alert_id = f"TRA-{sha256(stable_key.encode('utf-8')).hexdigest()[:10].upper()}"
        message = f"{risk_level} demo traffic-risk recommendation for {source_name}: {action}"
        return ResponseRecommendation(
            alert_id=alert_id, timestamp=timestamp, source_name=source_name,
            risk_score=round(risk_score, 3), risk_level=risk_level,
            alert_priority=priority, alert_status=status, recommended_action=action,
            recommended_officers=officers, message=message, reasons=reasons,
            decision_trace=trace,
        )

    def process(self, risk_event: dict[str, Any]) -> dict[str, Any]:
        """Produce the versioned response event expected by downstream delivery layers."""
        return build_response_event(risk_event, self.recommend(risk_event).as_dict())

    @staticmethod
    def _level_from_score(score: float) -> str:
        """Unified thresholds matching classifier, registry, and frontend."""
        if score >= 0.85:
            return "CRITICAL"
        if score >= 0.70:
            return "HIGH"
        if score >= 0.40:
            return "MEDIUM"
        return "LOW"
