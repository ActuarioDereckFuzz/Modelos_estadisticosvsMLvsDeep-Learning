"""Inferencia reproducible de los GLM estadísticos de prima pura.

Los modelos ajustados se entrenan en notebooks/03_statistical_models.ipynb.
Este módulo NO vuelve a entrenar ni selecciona modelos: valida datos nuevos,
reproduce transformaciones aprendidas en train y combina predicciones.

No cargar archivos .joblib de fuentes desconocidas (joblib usa pickle).
"""
from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = (
    "edad",
    "zona",
    "anios_vehiculo",
    "cobertura",
    "historial_siniestros",
    "tipo_uso",
)

CATEGORY_MAPPINGS = {
    "zona": {1: "Muy bajo", 2: "Bajo", 3: "Medio", 4: "Alto", 5: "Muy alto"},
    "cobertura": {0: "Limitada", 1: "RC", 2: "Amplia"},
    "tipo_uso": {0: "Particular", 1: "Transporte privado", 2: "Carga comercial"},
}

MODEL_KEYS = ("poisson", "gamma", "log_severity", "tweedie")
# Opciones públicas de inferencia ("poisson_gamma" combina dos estimadores).
PREDICTION_CHOICES = ("all", "poisson", "gamma", "log_severity", "poisson_gamma", "tweedie")
BUNDLE_FORMAT = "statistical_pure_premium_bundle"
BUNDLE_VERSION = 1


def prepare_policy_features(
    raw: pd.DataFrame,
    edad_media_train: float | None = None,
    *,
    category_mappings: Mapping[str, Mapping[int, str]] | None = None,
) -> pd.DataFrame:
    """Transforma variables crudas con mapas fijos sin aprender usando test.

    - Valida esquema, faltantes, categorías conocidas y valores numéricos.
    - Conserva índice y variables originales, incluyendo `policy_id` si existe.
    - Si se proporciona la media de edad de train, agrega `edad_c` y `edad_c2`.
    """
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("Se requiere un DataFrame de pandas con variables crudas.")
    missing = sorted(set(FEATURE_COLUMNS).difference(raw.columns))
    if missing:
        raise ValueError(f"Faltan variables predictoras: {missing}")

    data = raw.copy()
    for col in FEATURE_COLUMNS:
        data[col] = pd.to_numeric(data[col], errors="raise")
        if not np.isfinite(data[col].to_numpy(dtype=float)).all():
            raise ValueError(f"La variable {col!r} contiene NaN o valores infinitos.")

    for col in ("edad", "anios_vehiculo", "historial_siniestros"):
        if (data[col] < 0).any():
            raise ValueError(f"La variable {col!r} no puede ser negativa.")
    if (data["historial_siniestros"] % 1 != 0).any():
        raise ValueError("historial_siniestros debe ser un conteo entero.")

    maps = CATEGORY_MAPPINGS if category_mappings is None else category_mappings
    for feature, labels in maps.items():
        invalid = ~data[feature].isin(labels.keys())
        if invalid.any():
            values = data.loc[invalid, feature].unique().tolist()
            raise ValueError(
                f"Valores no contemplados en {feature!r}: {values}. "
                "No se extrapolan categorías fuera del entrenamiento."
            )
        data[f"{feature}_cat"] = data[feature].map(labels)

    if edad_media_train is not None:
        center = float(edad_media_train)
        if not np.isfinite(center):
            raise ValueError("edad_media_train debe ser finita.")
        data["edad_c"] = data["edad"] - center
        data["edad_c2"] = data["edad_c"] ** 2

    return data


def save_statistical_bundle(
    path: str | Path,
    *,
    poisson_model: Any,
    gamma_model: Any,
    log_severity_model: Any,
    tweedie_model: Any,
    edad_media_train: float,
    smearing_factor: float,
    tweedie_power: float,
    random_seed: int = 42,
) -> Path:
    """Persiste estimadores finales y transformaciones necesarias para inferencia."""
    if not np.isfinite(float(edad_media_train)):
        raise ValueError("La edad media del entrenamiento debe ser finita.")
    if not np.isfinite(float(smearing_factor)) or smearing_factor <= 0:
        raise ValueError("El factor de Duan debe ser finito y positivo.")
    if not 1.0 < float(tweedie_power) < 2.0:
        raise ValueError("La potencia del Tweedie compuesto debe estar en (1, 2).")

    bundle = {
        "bundle_format": BUNDLE_FORMAT,
        "bundle_version": BUNDLE_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "preprocessing": {
            "feature_columns": list(FEATURE_COLUMNS),
            "category_mappings": {k: dict(v) for k, v in CATEGORY_MAPPINGS.items()},
            "edad_media_train": float(edad_media_train),
            "smearing_factor": float(smearing_factor),
            "tweedie_power": float(tweedie_power),
        },
        "training": {
            "split_seed": int(random_seed),
            "source": "03_statistical_models.ipynb",
            "note": "Modelos ajustados sin predictores oracle; no incluye CSV ni datos de train/test.",
        },
        "dependencies": {
            name: version(name)
            for name in ("numpy", "pandas", "statsmodels", "patsy", "joblib")
        },
        "models": {
            "poisson": poisson_model,
            "gamma": gamma_model,
            "log_severity": log_severity_model,
            "tweedie": tweedie_model,
        },
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_statistical_bundle(path: str | Path) -> dict:
    """Carga SOLO artefactos confiables creados por el propio proyecto."""
    bundle = joblib.load(Path(path))
    if not isinstance(bundle, dict):
        raise ValueError("El artefacto no contiene un diccionario de modelos.")
    if bundle.get("bundle_format") != BUNDLE_FORMAT or bundle.get("bundle_version") != BUNDLE_VERSION:
        raise ValueError("Formato o versión de artefacto no soportados.")
    if not set(MODEL_KEYS).issubset(bundle.get("models", {})):
        raise ValueError("El artefacto está incompleto: faltan estimadores.")
    for key in ("edad_media_train", "smearing_factor", "category_mappings"):
        if key not in bundle.get("preprocessing", {}):
            raise ValueError(f"El artefacto no guarda el preprocesamiento: {key}.")
    return bundle


def predict_statistical_models(
    raw: pd.DataFrame,
    bundle: Mapping[str, Any],
    *,
    model: str = "all",
) -> pd.DataFrame:
    """Predice sobre pólizas nuevas con el estimador o estrategia solicitada.

    Parámetros
    ----------
    raw : pd.DataFrame
        Contiene las seis variables en ``FEATURE_COLUMNS`` y opcionalmente
        ``policy_id``. No requiere targets ni datos oracle.
    bundle : Mapping
        Artefacto recuperado mediante ``load_statistical_bundle``.
    model : str, default="all"
        - ``"poisson"``: frecuencia esperada (``freq_glm_poisson``).
        - ``"gamma"``: severidad condicional (``sev_glm_gamma``).
        - ``"log_severity"``: severidad con retransformación de Duan
          (``sev_log_duan``).
        - ``"poisson_gamma"``: prima pura frecuencia × severidad Gamma
          (``pp_poisson_gamma``; ejecuta solo Poisson y Gamma).
        - ``"tweedie"``: prima pura directa (``pp_tweedie``).
        - ``"all"``: las cinco salidas, compatible con la versión anterior.

    Devuelve
    --------
    pd.DataFrame
        Conserva el índice y ``policy_id`` si existe, más **solo** las
        columnas solicitadas. ``sev_*`` es severidad esperada condicionada
        al siniestro; ``pp_*`` es prima pura técnica (sin gastos, margen,
        impuestos ni ajustes comerciales).

    Notas
    -----
    Se carga el bundle completo, pero únicamente se ejecutan los modelos
    necesarios para la opción elegida; no se vuelve a entrenar ninguno.
    """
    if not isinstance(model, str) or model not in PREDICTION_CHOICES:
        raise ValueError(
            f"Selección de modelo {model!r} no válida. "
            f"Opciones: {', '.join(PREDICTION_CHOICES)}."
        )

    prep = bundle["preprocessing"]
    features = prepare_policy_features(
        raw,
        edad_media_train=prep["edad_media_train"],
        category_mappings=prep["category_mappings"],
    )
    models = bundle["models"]

    def predict_positive(key: str) -> np.ndarray:
        values = np.asarray(models[key].predict(features), dtype=float).reshape(-1)
        if len(values) != len(features) or not np.isfinite(values).all() or (values <= 0).any():
            raise ValueError(f"El modelo {key!r} produjo predicciones inválidas.")
        return values

    result = pd.DataFrame(index=raw.index)
    if "policy_id" in raw.columns:
        result["policy_id"] = raw["policy_id"]

    # Ejecutamos solo los estimadores requeridos. En particular, la opción
    # "poisson_gamma" no ejecuta log-severidad ni Tweedie.
    frequency = (
        predict_positive("poisson")
        if model in ("all", "poisson", "poisson_gamma") else None
    )
    severity_gamma = (
        predict_positive("gamma")
        if model in ("all", "gamma", "poisson_gamma") else None
    )

    if model in ("all", "poisson"):
        result["freq_glm_poisson"] = frequency
    if model in ("all", "gamma"):
        result["sev_glm_gamma"] = severity_gamma
    if model in ("all", "log_severity"):
        predicted_log_severity = np.asarray(
            models["log_severity"].predict(features), dtype=float
        ).reshape(-1)
        if len(predicted_log_severity) != len(features) or not np.isfinite(predicted_log_severity).all():
            raise ValueError("La predicción de log-severidad contiene valores inválidos.")
        severity_log_duan = np.exp(predicted_log_severity) * float(prep["smearing_factor"])
        if not np.isfinite(severity_log_duan).all() or (severity_log_duan <= 0).any():
            raise ValueError("La severidad retransformada contiene valores inválidos.")
        result["sev_log_duan"] = severity_log_duan
    if model in ("all", "poisson_gamma"):
        result["pp_poisson_gamma"] = frequency * severity_gamma
    if model in ("all", "tweedie"):
        result["pp_tweedie"] = predict_positive("tweedie")

    return result
