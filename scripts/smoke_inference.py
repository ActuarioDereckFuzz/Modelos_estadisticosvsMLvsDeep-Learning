"""Smoke test de carga e inferencia sobre perfiles nuevos, SIN entrenamiento.

Ejecutar desde la raíz, tras instalar environment.yml:
    python scripts/smoke_inference.py
    python scripts/smoke_inference.py --skip-dl  # diagnóstico parcial sin TF

Las primas son técnicas (costos esperados) y no tarifas comerciales.
Solo cargar artefactos .joblib del repositorio propio/confiable.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def validate(frame: pd.DataFrame, name: str, n: int) -> None:
    if len(frame) != n:
        raise AssertionError(f"{name}: longitud inesperada")
    numerical = frame.select_dtypes(include="number")
    numerical = numerical.drop(columns=["policy_id"], errors="ignore")
    if numerical.empty or not np.isfinite(numerical.to_numpy(dtype=float)).all():
        raise AssertionError(f"{name}: predicciones ausentes o no finitas")
    if (numerical < 0).any().any():
        raise AssertionError(f"{name}: predicciones negativas")
    print(f"{name}: OK, {len(numerical)} pólizas; columnas: {', '.join(numerical.columns)}")
    print(numerical.round(3).to_string(index=False))


def main(skip_dl: bool = False) -> int:
    from src.models.statistical_predictor import load_statistical_bundle, predict_statistical_models
    from src.models.ml_persistence import load_ml_models, predict_ml_pure_premium

    # Tres perfiles nuevos sin targets ni oracle; valores válidos del DGP.
    policies = pd.DataFrame([
        {"policy_id": "P-A", "edad": 26, "zona": 1, "anios_vehiculo": 2, "cobertura": 0, "historial_siniestros": 0, "tipo_uso": 0},
        {"policy_id": "P-B", "edad": 42, "zona": 3, "anios_vehiculo": 6, "cobertura": 2, "historial_siniestros": 1, "tipo_uso": 0},
        {"policy_id": "P-C", "edad": 58, "zona": 5, "anios_vehiculo": 12, "cobertura": 2, "historial_siniestros": 3, "tipo_uso": 2},
    ])
    print("\n--- Estadísticos ---")
    bundle = load_statistical_bundle(ROOT / "models/statistical/statistical_models_v1.joblib")
    stat = predict_statistical_models(policies, bundle, model="all")
    validate(stat, "GLM / Log+Duan", len(policies))

    print("\n--- Machine Learning ---")
    ml_models, metadata = load_ml_models(ROOT / "models/machine_learning")
    pred_ml = predict_ml_pure_premium(policies, ml_models, metadata["feature_order"])
    validate(pred_ml, "RF / XGBoost", len(policies))

    if skip_dl:
        print("\nDeep Learning omitido a petición (--skip-dl).")
    else:
        print("\n--- Deep Learning ---")
        from src.models.dl_inference import FrozenDLPricing

        predictor = FrozenDLPricing(project_root=ROOT, version="v1")
        pred_dl = predictor.predict(policies)
        validate(pred_dl, "MLP Poisson / Gamma / Tweedie", len(policies))

    print("\nSmoke test de inferencia: PASS")
    return 0


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--skip-dl", action="store_true", help="Probar GLM y ML sin TensorFlow")
    arguments = p.parse_args()
    try:
        raise SystemExit(main(skip_dl=arguments.skip_dl))
    except Exception as exc:
        print(f"\nSmoke test: FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
