"""Trained risk scoring for aggregated traffic feature windows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


MODEL_FEATURES = [
    "traffic_volume",
    "average_speed",
    "pedestrian_density",
    "congestion_level",
    "accident_count",
    "weather_risk",
    "night_time_risk",
    "speed_risk",
]


@dataclass(frozen=True)
class RiskPrediction:
    risk_score: float
    risk_score_percent: float
    risk_level: str
    model_confidence: float
    explanation: list[str]
    imputed_context: list[str]
    model_inputs: dict[str, float]
    base_risk_score: float | None = None
    base_risk_score_percent: float | None = None
    live_risk_delta: float | None = None
    speed_reliable: bool | None = None
    risk_contributions: dict[str, float] | None = None
    risk_reasons: list[str] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_score_percent": self.risk_score_percent,
            "risk_level": self.risk_level,
            "model_confidence": self.model_confidence,
            "explanation": self.explanation,
            "imputed_context": self.imputed_context,
            "model_inputs": self.model_inputs,
            "base_risk_score": self.base_risk_score,
            "base_risk_score_percent": self.base_risk_score_percent,
            "live_risk_delta": self.live_risk_delta,
            "speed_reliable": self.speed_reliable,
            "risk_contributions": self.risk_contributions,
            "risk_reasons": self.risk_reasons,
            "model": "RandomForestRegressor / historical junction data",
        }


class LiveRiskAdjustment:
    """Rule-based live delta computed from real-time video features."""

    def compute(self, feature_window: dict[str, Any]) -> tuple[float, dict[str, float]]:
        contributions: dict[str, float] = {}
        delta = 0.0

        stopped_ratio = feature_window.get("stopped_vehicle_ratio")
        if stopped_ratio is not None and np.isfinite(stopped_ratio):
            contribution = 0.08 * float(stopped_ratio)
            contributions["stopped_vehicle_ratio"] = round(contribution, 4)
            delta += contribution

        queue = feature_window.get("queue_indicator")
        if queue is not None and np.isfinite(queue):
            contribution = 0.10 * float(queue)
            contributions["queue_indicator"] = round(contribution, 4)
            delta += contribution

        heavy_ratio = feature_window.get("heavy_vehicle_ratio")
        if heavy_ratio is not None and np.isfinite(heavy_ratio):
            contribution = 0.05 * float(heavy_ratio)
            contributions["heavy_vehicle_ratio"] = round(contribution, 4)
            delta += contribution

        if feature_window.get("peak_hour_flag") == 1:
            contributions["peak_hour_flag"] = 0.03
            delta += 0.03
        else:
            contributions["peak_hour_flag"] = 0.0

        delta = max(-0.15, min(delta, 0.20))
        return round(delta, 4), contributions


class TrafficRiskClassifier:
    """Maps vision feature windows to a score trained on project risk history."""

    def __init__(self, training_data_path: str | Path = "data/traffic_risk_data.csv") -> None:
        import joblib
        model_path = Path("data/traffic_risk_model.joblib")
        if model_path.exists():
            try:
                model_data = joblib.load(model_path)
                self.model = model_data["model"]
                self.defaults = model_data["defaults"]
                print("[TrafficRiskClassifier] Loaded pre-trained model from data/traffic_risk_model.joblib")
                return
            except Exception as e:
                print(f"[TrafficRiskClassifier] Error loading joblib model: {e}. Training on the fly...")
                
        training_data = pd.read_csv(training_data_path)
        self.defaults = training_data[MODEL_FEATURES].median().to_dict()
        self.model = RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(training_data[MODEL_FEATURES], training_data["risk_score"])

    @staticmethod
    def _risk_level(score_percent: float) -> str:
        """Unified thresholds matching registry, frontend, and allocation."""
        if score_percent >= 85:
            return "CRITICAL"
        if score_percent >= 70:
            return "HIGH"
        if score_percent >= 40:
            return "MEDIUM"
        return "LOW"

    def _map_feature_window(self, feature_window: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
        state_to_congestion = {"light": 3.0, "moderate": 6.0, "high": 9.0}
        speed = feature_window.get("average_speed_kmh")
        incoming = {
            "traffic_volume": feature_window.get("vehicle_flow_per_hour"),
            "average_speed": speed,
            "pedestrian_density": None,
            "congestion_level": state_to_congestion.get(feature_window.get("traffic_state")),
            "accident_count": None,
            "weather_risk": None,
            "night_time_risk": None,
            "speed_risk": max(float(speed) - 40, 0) if speed is not None else None,
        }
        imputed: list[str] = []
        values: dict[str, float] = {}
        for feature in MODEL_FEATURES:
            value = incoming.get(feature)
            if value is None or not np.isfinite(value):
                values[feature] = float(self.defaults[feature])
                imputed.append(feature)
            else:
                values[feature] = float(value)
        return values, imputed

    def predict(self, feature_window: dict[str, Any]) -> RiskPrediction:
        inputs, imputed = self._map_feature_window(feature_window)
        speed = feature_window.get("average_speed_kmh")
        row = pd.DataFrame([inputs], columns=MODEL_FEATURES)
        tree_predictions = np.array([tree.predict(row)[0] for tree in self.model.estimators_])
        score_percent = float(np.clip(tree_predictions.mean(), 0, 100))
        confidence = float(np.clip(1 - tree_predictions.std() / 25, 0, 1))

        base_score = score_percent / 100.0

        live_delta, contributions = LiveRiskAdjustment().compute(feature_window)
        final_score = round(max(0.0, min(base_score + live_delta, 1.0)), 3)
        final_percent = round(final_score * 100, 2)

        stopped_ratio = feature_window.get("stopped_vehicle_ratio", 0.0) or 0.0
        queue_ind = feature_window.get("queue_indicator", 0.0) or 0.0
        heavy_ratio = feature_window.get("heavy_vehicle_ratio", 0.0) or 0.0
        peak = feature_window.get("peak_hour_flag", 0)

        reasons: list[str] = []
        if live_delta > 0:
            reasons.append(
                f"Historical base risk: {score_percent:.1f}%. Live adjustment: +{live_delta * 100:.1f}%. "
                f"Stopped vehicles: {stopped_ratio * 100:.1f}%, queue indicator: {queue_ind * 100:.1f}%, "
                f"heavy vehicles: {heavy_ratio * 100:.1f}%, peak hour: {'YES' if peak == 1 else 'NO'}. "
                f"Final risk: {final_percent:.1f}%."
            )
        else:
            reasons.append(
                f"Historical base risk: {score_percent:.1f}%. No live adjustment. Final risk: {final_percent:.1f}%."
            )

        if float(feature_window.get("vehicle_density_per_100m", 0)) >= 15:
            reasons.append("High observed vehicle density")
        if float(feature_window.get("vehicle_flow_per_hour", 0)) >= 3500:
            reasons.append("High estimated vehicle flow")
        if speed is not None and float(speed) >= 60:
            reasons.append("Elevated calibrated average speed")
        if not reasons:
            reasons.append("Current observed flow and density are within the learned baseline range")
        if imputed:
            reasons.append("Some non-vision context uses historical median defaults")

        return RiskPrediction(
            risk_score=final_score,
            risk_score_percent=final_percent,
            risk_level=self._risk_level(final_percent),
            model_confidence=round(confidence, 3),
            explanation=reasons,
            imputed_context=imputed,
            model_inputs={key: round(float(value), 3) for key, value in inputs.items()},
            base_risk_score=round(base_score, 3),
            base_risk_score_percent=round(score_percent, 2),
            live_risk_delta=round(live_delta, 3),
            speed_reliable=bool(feature_window.get("speed_reliable", False)),
            risk_contributions={k: round(v * 100, 1) for k, v in contributions.items()},
            risk_reasons=reasons,
        )
