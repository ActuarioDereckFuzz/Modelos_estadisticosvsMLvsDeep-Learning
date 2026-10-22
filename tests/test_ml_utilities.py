"""Tests ligeros de utilidades ML, sin entrenar Random Forest ni XGBoost."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.evaluation.ml_calibration import calibration_by_decile, calibration_by_segments
from src.models.ml_persistence import MODEL_TARGETS, load_ml_models, predict_ml_pure_premium, save_ml_models


FEATURES = ["edad", "anios_vehiculo", "historial_siniestros", "zona", "cobertura", "tipo_uso"]
NUMERIC = FEATURES[:3]
CATEGORICAL = FEATURES[3:]


def _portfolio(n=20):
    i = np.arange(n)
    return pd.DataFrame({
        "policy_id": i + 1,
        "edad": 20 + i,
        "anios_vehiculo": i % 7,
        "historial_siniestros": i % 3,
        "zona": i % 5 + 1,
        "cobertura": i % 3,
        "tipo_uso": i % 2,
        "costo_siniestros": 900.0 + 140.0 * i,
        "prima_pura_real": 950.0 + 135.0 * i,
        "pp_rf_fs": 960.0 + 130.0 * i,
        "pp_xgb_fs": 940.0 + 140.0 * i,
        "pp_rf_direct": 970.0 + 129.0 * i,
        "pp_xgb_direct": 980.0 + 127.0 * i,
    })


def test_calibration_preserves_rows_and_segments():
    data = _portfolio()
    byseg = calibration_by_segments(data, ["zona", "cobertura"], {"XGBoost directo": "pp_xgb_direct"})
    assert set(byseg["segment_type"]) == {"zona", "cobertura"}
    grouped, dec = calibration_by_decile(data)
    assert len(grouped) == len(data) and int(dec["n_policy"].sum()) == len(data)
    assert len(dec) == 10
    assert (dec["n_policy"] == 2).all()


def test_save_load_predict_roundtrip(tmp_path):
    X = _portfolio()[FEATURES]
    pre = ColumnTransformer([("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL)], remainder="passthrough")
    # Stub pipelines fitted, solo para probar IO sin búsquedas ni modelos pesados.
    fitted = {}
    for model_name in MODEL_TARGETS:
        pipe = Pipeline([("preprocessor", pre), ("model", DummyRegressor(strategy="constant", constant=2.0))])
        pipe.fit(X, np.full(len(X), 2.0))
        fitted[model_name] = pipe
    input_file = tmp_path / "input.csv"
    X.to_csv(input_file, index=False)
    saved, manifest_path = save_ml_models(
        fitted, tmp_path / "models", tmp_path,
        FEATURES, NUMERIC, CATEGORICAL, {c: str(X[c].dtype) for c in FEATURES},
        {"synthetic": input_file},
    )
    assert manifest_path.is_file()
    restored, metadata = load_ml_models(tmp_path / "models")
    assert set(restored) == set(MODEL_TARGETS)
    assert saved["feature_order"] == metadata["feature_order"]
    preds = predict_ml_pure_premium(X, restored, FEATURES)
    assert len(preds) == len(X)
    assert (preds["pp_xgb_fs"].to_numpy() == 4.0).all()
    with pytest.raises(ValueError, match="Faltan variables"):
        predict_ml_pure_premium(X.drop(columns=["zona"]), restored, FEATURES)
    (tmp_path / "models" / "cost_xgb.joblib").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA256"):
        load_ml_models(tmp_path / "models")
