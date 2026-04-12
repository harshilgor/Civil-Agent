"""
ML-based section recommendation helpers.

The training path is optional. Prediction gracefully falls back to Civil
Agent's deterministic beam scan when a trained model is not available.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from beams_data import BEAMS_DF


FEATURES = [
    "span_ft",
    "dead_load_kip_ft",
    "live_load_kip_ft",
    "beam_spacing_ft",
    "composite_ratio",
    "unbraced_length_ft",
    "floor_number",
    "num_floors_total",
    "occupancy_encoded",
    "building_height_ft",
    "min_passing_weight",
    "min_passing_index",
]

OCCUPANCY_CODES = {
    "office": 0.0,
    "retail": 1.0,
    "warehouse": 2.0,
    "residential": 3.0,
    "hospital": 4.0,
}


def _greedy_recommendation(
    span_ft: float,
    dead_load: float,
    live_load: float,
    beam_spacing: float,
) -> list[tuple[str, float]]:
    try:
        from composite_beam import design_composite_beam

        result = design_composite_beam(
            span_ft=span_ft,
            dead_load=dead_load,
            live_load=live_load,
            beam_spacing_ft=beam_spacing,
        )
        if result and result.get("passes"):
            return [(result["name"], 1.0)]
    except Exception:
        pass
    return []


def _occupancy_encoded(value: str) -> float:
    return OCCUPANCY_CODES.get((value or "office").strip().lower(), 0.0)


def _prepare_feature_row(
    span_ft: float,
    dead_load: float,
    live_load: float,
    beam_spacing: float,
    *,
    composite_ratio: float = 0.5,
    unbraced_length_ft: float | None = None,
    floor_number: float = 0.5,
    num_floors_total: float = 1.0,
    occupancy: str = "office",
    building_height_ft: float | None = None,
    min_passing_weight: float = 0.0,
    min_passing_index: float = 0.0,
) -> pd.DataFrame:
    if unbraced_length_ft is None:
        unbraced_length_ft = span_ft
    if building_height_ft is None:
        building_height_ft = num_floors_total * 14.0
    return pd.DataFrame(
        [[
            span_ft,
            dead_load,
            live_load,
            beam_spacing,
            composite_ratio,
            unbraced_length_ft,
            floor_number,
            num_floors_total,
            _occupancy_encoded(occupancy),
            building_height_ft,
            min_passing_weight,
            min_passing_index,
        ]],
        columns=FEATURES,
    )


def train_model(
    training_data_path: str,
    model_save_path: str = "models/section_recommender.pkl",
):
    try:
        import joblib
        import xgboost as xgb
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
    except ImportError as exc:
        raise ImportError("pip install xgboost scikit-learn joblib") from exc

    df = pd.read_csv(training_data_path)
    X = df[FEATURES]
    y = df["section_index"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        eval_metric="mlogloss",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    y_proba = model.predict_proba(X_val)
    top3_correct = sum(
        y_val.iloc[i] in np.argsort(y_proba[i])[-3:]
        for i in range(len(y_val))
    ) / max(len(y_val), 1)

    model_path = Path(model_save_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"Validation accuracy: {acc:.3f}")
    print(f"Top-3 accuracy: {top3_correct:.3f}")
    print(f"Model saved to {model_path}")
    return model


def predict_section(
    span_ft: float,
    dead_load: float,
    live_load: float,
    beam_spacing: float,
    model_path: str = "models/section_recommender.pkl",
    top_k: int = 3,
) -> list[tuple[str, float]]:
    try:
        import joblib
    except ImportError:
        return _greedy_recommendation(span_ft, dead_load, live_load, beam_spacing)

    if not os.path.exists(model_path):
        return _greedy_recommendation(span_ft, dead_load, live_load, beam_spacing)

    try:
        model = joblib.load(model_path)
        baseline = _greedy_recommendation(span_ft, dead_load, live_load, beam_spacing)
        min_name = baseline[0][0] if baseline else BEAMS_DF.iloc[0]["name"]
        min_row = BEAMS_DF[BEAMS_DF["name"] == min_name]
        min_weight = float(min_row.iloc[0]["weight"]) if not min_row.empty else 0.0
        min_index = float(min_row.index[0]) if not min_row.empty else 0.0
        X = _prepare_feature_row(
            span_ft,
            dead_load,
            live_load,
            beam_spacing,
            min_passing_weight=min_weight,
            min_passing_index=min_index,
        )
        probabilities = model.predict_proba(X)[0]
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        predictions = []
        for idx in top_indices:
            if idx >= len(BEAMS_DF):
                continue
            predictions.append((str(BEAMS_DF.iloc[idx]["name"]), round(float(probabilities[idx]), 3)))
        return predictions or baseline
    except Exception:
        return _greedy_recommendation(span_ft, dead_load, live_load, beam_spacing)
