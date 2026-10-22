"""Registro, persistencia e inferencia de los seis pipelines ML de prima pura.

Los seis modelos son estimadores YA ENTRENADOS. El módulo no contiene fit ni
búsqueda de hiperparámetros. No cargar joblib desde fuentes no confiables.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost


MODEL_TARGETS = {
    "frequency_rf": "numero_siniestros",
    "frequency_xgb": "numero_siniestros",
    "severity_rf": "severidad",
    "severity_xgb": "severidad",
    "cost_rf": "costo_siniestros",
    "cost_xgb": "costo_siniestros",
}


def _sha256(path: str | Path) -> str:
    """Hash del contenido binario, no del nombre."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_ml_models(
    models: dict,
    model_directory: str | Path,
    project_root: str | Path,
    feature_order: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    feature_dtypes_train: dict[str, str],
    input_paths: dict[str, str | Path],
    seed: int = 42,
    notebook: str = "04_machine_learning.ipynb",
) -> tuple[dict, Path]:
    """Guarda seis pipelines preajustados y sus checksums.

    Se mantiene el contrato de nombres para que la fase 07 utilice siempre las
    mismas cuatro variantes de prima pura: RF/XGB × (F×S/directa).
    """
    if set(models) != set(MODEL_TARGETS):
        raise ValueError(f"Se necesitan exactamente estos modelos: {sorted(MODEL_TARGETS)}")
    if set(feature_order) != set(numeric_features + categorical_features):
        raise ValueError("feature_order no coincide con numéricas + categóricas")
    for key, model in models.items():
        if not hasattr(model, "predict") or not hasattr(model, "named_steps"):
            raise ValueError(f"{key} no es un Pipeline predictivo")
        if "model" not in model.named_steps or "preprocessor" not in model.named_steps:
            raise ValueError(f"{key} carece de model o preprocessor")
        if not hasattr(model.named_steps["preprocessor"], "transformers_"):
            raise ValueError(f"{key} no tiene preprocesador ajustado")
    directory = Path(model_directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(project_root)
    manifest = {
        "notebook": notebook,
        "seed": seed,
        "input_sha256": {name: _sha256(path) for name, path in input_paths.items()},
        "feature_order": list(feature_order),
        "numeric_features": list(numeric_features),
        "categorical_features": list(categorical_features),
        "feature_dtypes_train": dict(feature_dtypes_train),
        "target_by_model": MODEL_TARGETS.copy(),
        "library_versions": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "joblib": joblib.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "note": "Pipelines ajustados sobre train. No se reentrenan al persistir ni en inferencia.",
        "models": {},
    }
    for key, model in models.items():
        file_path = directory / f"{key}.joblib"
        joblib.dump(model, file_path, compress=3)
        try:
            relpath = file_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            relpath = file_path.name
        manifest["models"][key] = {
            "path": relpath,
            "sha256": _sha256(file_path),
            "parameters": model.named_steps["model"].get_params(),
        }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return manifest, manifest_path


def load_ml_models(model_directory: str | Path) -> tuple[dict, dict]:
    """Verifica SHA256 y carga pipelines propios/confiables sin reentrenar."""
    directory = Path(model_directory)
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if set(metadata["models"]) != set(MODEL_TARGETS):
        raise ValueError("El manifiesto no contiene los seis modelos esperados")
    for package, running in (("scikit_learn", sklearn.__version__), ("xgboost", xgboost.__version__)):
        trained = metadata["library_versions"].get(package)
        if trained != running:
            warnings.warn(f"Versión distinta de {package}: entrenamiento {trained}, actual {running}", stacklevel=2)
    models = {}
    for key, info in metadata["models"].items():
        file_path = directory / Path(info["path"]).name
        if _sha256(file_path) != info["sha256"]:
            raise ValueError(f"SHA256 no coincide: {key}")
        # joblib.load implica deserialización de pickle; NO usar con archivos no confiables.
        models[key] = joblib.load(file_path)
    return models, metadata


def predict_ml_pure_premium(
    new_portfolio: pd.DataFrame, models: dict, features: list[str]
) -> pd.DataFrame:
    """Predice lambda, severidad por claim y prima pura para cada póliza nueva.

    Las predicciones F×S suponen que la severidad por claim modelada puede
    aplicarse a la distribución de riesgo por póliza. No incluyen gastos,
    márgenes, cargas comerciales ni disposición a pagar.
    """
    if set(models) != set(MODEL_TARGETS):
        raise ValueError("Falta alguno de los seis modelos")
    missing = [column for column in features if column not in new_portfolio.columns]
    if missing:
        raise ValueError(f"Faltan variables para tarificación: {missing}")
    X = new_portfolio.loc[:, features].copy()
    if X.isna().any().any():
        raise ValueError("La nueva cartera contiene predictores nulos")
    if any(pd.api.types.is_object_dtype(X[column]) for column in features):
        # Esta cartera usa categorías codificadas numéricamente; evita que un
        # archivo con códigos en strings cree categorías desconocidas sin aviso.
        raise ValueError("Se encontraron variables object; verificar tipos de las seis features")
    preds = pd.DataFrame(index=X.index)
    preds["freq_rf"] = models["frequency_rf"].predict(X)
    preds["freq_xgb"] = models["frequency_xgb"].predict(X)
    preds["sev_rf"] = models["severity_rf"].predict(X)
    preds["sev_xgb"] = models["severity_xgb"].predict(X)
    preds["pp_rf_fs"] = preds["freq_rf"] * preds["sev_rf"]
    preds["pp_xgb_fs"] = preds["freq_xgb"] * preds["sev_xgb"]
    preds["pp_rf_direct"] = models["cost_rf"].predict(X)
    preds["pp_xgb_direct"] = models["cost_xgb"].predict(X)
    matrix = preds.to_numpy(dtype=float)
    if not np.isfinite(matrix).all() or (matrix < 0).any():
        raise ValueError("Predicciones no finitas o negativas")
    return preds
