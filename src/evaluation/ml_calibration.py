"""Diagnósticos descriptivos de calibración para modelos de prima pura.

Todas las funciones usan predicciones congeladas. Nunca entrenan, recalibran
ni seleccionan modelos sobre el conjunto de prueba.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def segment_calibration_table(
    data: pd.DataFrame, segment_name: str, model_name: str, prediction_col: str
) -> pd.DataFrame:
    """Agrega sumas y sesgos por segmento; oracle solo para datos sintéticos.

    Sesgo oracle = suma(predicción)/suma(prima_pura_real) - 1.
    Loss ratio observado = suma(costo_siniestros)/suma(predicción).
    """
    needed = {"policy_id", "costo_siniestros", "prima_pura_real", segment_name, prediction_col}
    missing = needed - set(data.columns)
    if missing:
        raise ValueError(f"Columnas faltantes para calibración: {sorted(missing)}")
    summary = (
        data.groupby(segment_name, observed=True, dropna=False)
        .agg(
            n_policy=("policy_id", "size"),
            costo_observado=("costo_siniestros", "sum"),
            prima_oracle=("prima_pura_real", "sum"),
            prima_predicha=(prediction_col, "sum"),
        )
        .reset_index()
        .rename(columns={segment_name: "segment_value"})
    )
    summary.insert(0, "segment_type", segment_name)
    summary.insert(0, "model", model_name)
    summary["sesgo_rel_oracle"] = (
        summary["prima_predicha"] / summary["prima_oracle"].replace(0, np.nan) - 1
    )
    summary["loss_ratio_observado"] = (
        summary["costo_observado"] / summary["prima_predicha"].replace(0, np.nan)
    )
    summary["n_menor_30"] = summary["n_policy"] < 30
    return summary


def calibration_by_segments(
    data: pd.DataFrame,
    segment_columns: list[str],
    premium_models: dict[str, str],
) -> pd.DataFrame:
    """Concatenación de métricas por cada cruce (segmento, modelo)."""
    if data["policy_id"].duplicated().any():
        raise ValueError("Una misma póliza aparece varias veces en calibration_data")
    if not segment_columns or not premium_models:
        raise ValueError("Se requieren al menos un segmento y un modelo")
    tables = [
        segment_calibration_table(data, segment, model_name, pred_col)
        for segment in segment_columns
        for model_name, pred_col in premium_models.items()
    ]
    result = pd.concat(tables, ignore_index=True)
    if not np.isfinite(result["prima_predicha"].to_numpy(dtype=float)).all():
        raise ValueError("Primas predichas no finitas")
    return result


def calibration_by_decile(
    data: pd.DataFrame,
    reference_prediction_col: str = "pp_xgb_direct",
    prediction_cols: tuple[str, ...] = (
        "pp_rf_fs", "pp_xgb_fs", "pp_rf_direct", "pp_xgb_direct"
    ),
    n_deciles: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Asigna grupos por score predicho; devuelve (datos con grupo, tabla).

    Los cortes se obtienen SOLO del score predicho; las respuestas no influyen.
    En caso de empate, rank(method='first') genera grupos de igual tamaño
    (con desempate dependiente del orden de las filas).
    """
    required = {"policy_id", "costo_siniestros", "prima_pura_real", reference_prediction_col, *prediction_cols}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {sorted(missing)}")
    if len(data) < n_deciles or n_deciles < 2:
        raise ValueError("Número insuficiente de pólizas o de grupos")
    if data[reference_prediction_col].isna().any():
        raise ValueError("El score de referencia tiene valores faltantes")
    result_data = data.copy()
    group_col = "decil_riesgo_xgb_direct" if reference_prediction_col == "pp_xgb_direct" else "decil_riesgo"
    result_data[group_col] = (
        pd.qcut(result_data[reference_prediction_col].rank(method="first"), q=n_deciles, labels=False) + 1
    )
    agg_args = {
        "n_policy": ("policy_id", "size"),
        "costo_observado_medio": ("costo_siniestros", "mean"),
        "prima_oracle_media": ("prima_pura_real", "mean"),
    }
    agg_args.update({f"{col}_media": (col, "mean") for col in prediction_cols})
    result_table = result_data.groupby(group_col).agg(**agg_args).reset_index()
    return result_data, result_table
