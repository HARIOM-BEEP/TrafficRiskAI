"""Tests for the hybrid live risk engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from risk_classifier import LiveRiskAdjustment, MODEL_FEATURES, TrafficRiskClassifier


def _build_classifier(df: pd.DataFrame | None = None) -> TrafficRiskClassifier:
    if df is None:
        df = pd.read_csv("data/traffic_risk_data.csv")
    classifier = TrafficRiskClassifier.__new__(TrafficRiskClassifier)
    classifier.defaults = df[MODEL_FEATURES].median().to_dict()
    classifier.model = RandomForestRegressor(
        n_estimators=10,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    classifier.model.fit(df[MODEL_FEATURES], df["risk_score"])
    return classifier


def test_live_risk_adjustment_case_a_normal_traffic() -> None:
    adj = LiveRiskAdjustment()
    delta, _ = adj.compute({
        "stopped_vehicle_ratio": 0.0,
        "queue_indicator": 0.0,
        "heavy_vehicle_ratio": 0.0,
        "peak_hour_flag": 0,
    })
    assert delta == 0.0


def test_live_risk_adjustment_case_b_heavy_queue() -> None:
    adj = LiveRiskAdjustment()
    delta, _ = adj.compute({
        "stopped_vehicle_ratio": 0.8,
        "queue_indicator": 0.9,
        "heavy_vehicle_ratio": 0.2,
        "peak_hour_flag": 1,
    })
    expected = 0.08 * 0.8 + 0.10 * 0.9 + 0.05 * 0.2 + 0.03
    assert abs(delta - expected) < 1e-9
    assert delta <= 0.20


def test_live_risk_adjustment_case_c_missing_fields() -> None:
    adj = LiveRiskAdjustment()
    delta, _ = adj.compute({})
    assert delta == 0.0


def test_live_risk_adjustment_clamp_upper() -> None:
    adj = LiveRiskAdjustment()
    delta, _ = adj.compute({
        "stopped_vehicle_ratio": 10.0,
        "queue_indicator": 10.0,
        "heavy_vehicle_ratio": 10.0,
        "peak_hour_flag": 1,
    })
    assert delta <= 0.20


def test_live_risk_adjustment_clamp_lower() -> None:
    adj = LiveRiskAdjustment()
    delta, _ = adj.compute({})
    assert delta >= -0.15


def test_predict_score_boundary_upper() -> None:
    df = pd.read_csv("data/traffic_risk_data.csv")
    classifier = _build_classifier(df)
    window = {
        "vehicle_flow_per_hour": 4994.0,
        "average_speed_kmh": 69.0,
        "traffic_state": "high",
        "vehicle_density_per_100m": 20.0,
        "stopped_vehicle_ratio": 1.0,
        "queue_indicator": 1.0,
        "heavy_vehicle_ratio": 1.0,
        "peak_hour_flag": 1,
        "speed_reliable": True,
    }
    prediction = classifier.predict(window)
    assert prediction.risk_score <= 1.0
    assert prediction.risk_score_percent <= 100.0


def test_predict_score_boundary_lower() -> None:
    df = pd.read_csv("data/traffic_risk_data.csv")
    classifier = _build_classifier(df)
    window = {
        "vehicle_flow_per_hour": 509.0,
        "average_speed_kmh": 15.0,
        "traffic_state": "light",
        "vehicle_density_per_100m": 1.0,
        "stopped_vehicle_ratio": 0.0,
        "queue_indicator": 0.0,
        "heavy_vehicle_ratio": 0.0,
        "peak_hour_flag": 0,
        "speed_reliable": True,
    }
    prediction = classifier.predict(window)
    assert prediction.risk_score >= 0.0
    assert prediction.risk_score_percent >= 0.0


def test_predict_speed_unreliable_does_not_decrease_score() -> None:
    df = pd.read_csv("data/traffic_risk_data.csv")
    classifier = _build_classifier(df)
    window_reliable = {
        "vehicle_flow_per_hour": 2000.0,
        "average_speed_kmh": 40.0,
        "traffic_state": "moderate",
        "vehicle_density_per_100m": 8.0,
        "stopped_vehicle_ratio": 0.0,
        "queue_indicator": 0.0,
        "heavy_vehicle_ratio": 0.0,
        "peak_hour_flag": 0,
        "speed_reliable": True,
    }
    window_unreliable = dict(window_reliable, speed_reliable=False)
    pred_reliable = classifier.predict(window_reliable)
    pred_unreliable = classifier.predict(window_unreliable)
    assert pred_unreliable.risk_score >= pred_reliable.risk_score


def test_predict_contains_base_and_delta_fields() -> None:
    df = pd.read_csv("data/traffic_risk_data.csv")
    classifier = _build_classifier(df)
    window = {
        "vehicle_flow_per_hour": 2000.0,
        "average_speed_kmh": 40.0,
        "traffic_state": "moderate",
        "vehicle_density_per_100m": 8.0,
        "stopped_vehicle_ratio": 0.2,
        "queue_indicator": 0.1,
        "heavy_vehicle_ratio": 0.1,
        "peak_hour_flag": 1,
        "speed_reliable": True,
    }
    prediction = classifier.predict(window)
    assert prediction.base_risk_score is not None
    assert prediction.base_risk_score_percent is not None
    assert prediction.live_risk_delta is not None
    assert prediction.speed_reliable is True
    assert prediction.risk_score == prediction.base_risk_score + prediction.live_risk_delta


def test_predict_delta_is_positive_for_busy_traffic() -> None:
    df = pd.read_csv("data/traffic_risk_data.csv")
    classifier = _build_classifier(df)
    window = {
        "vehicle_flow_per_hour": 4000.0,
        "average_speed_kmh": 50.0,
        "traffic_state": "high",
        "vehicle_density_per_100m": 18.0,
        "stopped_vehicle_ratio": 0.6,
        "queue_indicator": 0.7,
        "heavy_vehicle_ratio": 0.3,
        "peak_hour_flag": 1,
        "speed_reliable": True,
    }
    prediction = classifier.predict(window)
    assert prediction.live_risk_delta > 0.0
    assert prediction.risk_score > prediction.base_risk_score


def test_predict_explanation_mentions_adjustment() -> None:
    df = pd.read_csv("data/traffic_risk_data.csv")
    classifier = _build_classifier(df)
    window = {
        "vehicle_flow_per_hour": 2000.0,
        "average_speed_kmh": 40.0,
        "traffic_state": "moderate",
        "vehicle_density_per_100m": 8.0,
        "stopped_vehicle_ratio": 0.2,
        "queue_indicator": 0.1,
        "heavy_vehicle_ratio": 0.1,
        "peak_hour_flag": 1,
        "speed_reliable": True,
    }
    prediction = classifier.predict(window)
    assert any("Live adjustment" in str(item) for item in prediction.explanation)
    assert any("Final risk" in str(item) for item in prediction.explanation)
