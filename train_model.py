"""
TrafficRisk AI — Model Training & Evaluation
============================================
Trains and compares three models:
  1. RandomForest (baseline)
  2. GradientBoosting (improved accuracy)
  3. Stacked Ensemble (best of both)

Saves the best model to data/traffic_risk_model.joblib.
The classifier uses this joblib artifact on startup for fast, reproducible predictions.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Features & output target ────────────────────────────────────────────────
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
TARGET = "risk_score"

# ── Risk helpers ─────────────────────────────────────────────────────────────
def score_to_level(score: float) -> str:
    if score >= 70:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def print_metrics(name: str, y_true, y_pred) -> dict:
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    # Level accuracy
    true_levels = [score_to_level(float(s)) for s in y_true]
    pred_levels = [score_to_level(float(s)) for s in y_pred]
    level_acc = sum(t == p for t, p in zip(true_levels, pred_levels)) / len(true_levels)
    print(f"\n  ──── {name} ────")
    print(f"  MSE:              {mse:.4f}")
    print(f"  MAE:              {mae:.4f}")
    print(f"  R² Score:         {r2:.4f}")
    print(f"  Risk-level accuracy: {level_acc:.1%}  (HIGH/MEDIUM/LOW classification)")
    return {"name": name, "r2": r2, "mae": mae, "level_acc": level_acc}


def train_risk_model():
    print("=" * 58)
    print("  TRAFFICRISK AI — MODEL TRAINING & COMPARISON")
    print("=" * 58)

    data_path = Path("data/traffic_risk_data.csv")
    if not data_path.exists():
        print("❌ data/traffic_risk_data.csv not found. Run generate_data.py first.")
        return

    df = pd.read_csv(data_path)
    X  = df[MODEL_FEATURES].fillna(df[MODEL_FEATURES].median())
    y  = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\n  Dataset: {len(df)} junctions · Train: {len(X_train)} · Test: {len(X_test)}")
    print(f"  Risk distribution: {df[TARGET].apply(score_to_level).value_counts().to_dict()}")

    results = []

    # ── 1. RandomForest ─────────────────────────────────────────────────────
    rf = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_leaf=2,
        min_samples_split=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    results.append(print_metrics("Random Forest", y_test, rf.predict(X_test)))

    # ── 2. GradientBoosting ─────────────────────────────────────────────────
    gb = GradientBoostingRegressor(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.06,
        subsample=0.85,
        min_samples_leaf=3,
        random_state=42,
    )
    gb.fit(X_train, y_train)
    results.append(print_metrics("Gradient Boosting", y_test, gb.predict(X_test)))

    # ── 3. Stacked Ensemble (RF + GB → Ridge meta-learner) ──────────────────
    stacked = StackingRegressor(
        estimators=[("rf", rf), ("gb", gb)],
        final_estimator=Pipeline([("sc", StandardScaler()), ("ridge", Ridge(alpha=1.0))]),
        cv=5,
        n_jobs=-1,
        passthrough=False,
    )
    stacked.fit(X_train, y_train)
    stacked_metrics = print_metrics("Stacked Ensemble (RF+GB)", y_test, stacked.predict(X_test))
    results.append(stacked_metrics)

    # ── Cross-validation on test-independent splits ──────────────────────────
    print("\n  ──── 5-Fold Cross-Validation (Stacked Ensemble) ────")
    cv_r2  = cross_val_score(stacked, X, y, cv=5, scoring="r2",                  n_jobs=-1)
    cv_mae = cross_val_score(stacked, X, y, cv=5, scoring="neg_mean_absolute_error", n_jobs=-1)
    print(f"  CV R²  scores: {cv_r2.round(3)}  → mean={cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    print(f"  CV MAE scores: {(-cv_mae).round(3)}  → mean={(-cv_mae).mean():.3f} ± {(-cv_mae).std():.3f}")

    # ── Feature importances (weighted average of both base learners) ─────────
    rf_imp = rf.feature_importances_
    gb_imp = gb.feature_importances_
    importance = (rf_imp * 0.5 + gb_imp * 0.5)
    order = np.argsort(importance)[::-1]
    print("\n  ──── Feature Importances (RF + GB average) ────")
    for rank, i in enumerate(order):
        bar = "█" * int(importance[i] * 50)
        print(f"  {rank+1}. {MODEL_FEATURES[i]:<22} {importance[i]:.4f}  {bar}")

    # ── Pick best model ──────────────────────────────────────────────────────
    best = max(results, key=lambda r: r["r2"])
    model_map = {
        "Random Forest":            rf,
        "Gradient Boosting":        gb,
        "Stacked Ensemble (RF+GB)": stacked,
    }
    best_model = model_map[best["name"]]
    defaults   = X.median().to_dict()

    print(f"\n  ✅ Best model: {best['name']}  (R²={best['r2']:.4f}, level-accuracy={best['level_acc']:.1%})")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = Path("data/traffic_risk_model.joblib")
    joblib.dump(
        {
            "model":      best_model,
            "model_name": best["name"],
            "features":   MODEL_FEATURES,
            "defaults":   defaults,
            "r2_score":   best["r2"],
            "level_acc":  best["level_acc"],
        },
        save_path,
        compress=3,
    )
    print(f"  Saved → {save_path}")
    print("=" * 58)


if __name__ == "__main__":
    train_risk_model()
