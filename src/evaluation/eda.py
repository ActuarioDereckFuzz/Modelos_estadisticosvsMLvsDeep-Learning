"""Resúmenes descriptivos reutilizables para el EDA de prima pura.

No realiza transformación de datos ni entrenamiento. Las tablas reproducen
las agregaciones del notebook 02 y respetan el orden de categorías indicado.
"""

from collections.abc import Mapping, Sequence

import pandas as pd


def _order_categories(
    summary: pd.DataFrame,
    variable: str,
    orders: Mapping[str, Sequence] | None,
) -> pd.DataFrame:
    """Ordena la columna categórica si el EDA proporcionó categorías."""
    if orders is not None and variable in orders:
        summary[variable] = pd.Categorical(
            summary[variable],
            categories=orders[variable],
            ordered=True,
        )
        summary = summary.sort_values(variable)
    return summary


def frequency_summary(
    df: pd.DataFrame,
    variable: str,
    *,
    orders: Mapping[str, Sequence] | None = None,
) -> pd.DataFrame:
    """Conteo, siniestros, tasa de frecuencia y proporción por segmento."""
    summary = (
        df.groupby(variable, observed=True)
        .agg(
            polizas=("numero_siniestros", "size"),
            siniestros=("numero_siniestros", "sum"),
            frecuencia=("numero_siniestros", "mean"),
            proporcion_con_siniestro=("tuvo_siniestro", "mean"),
        )
        .reset_index()
    )
    return _order_categories(summary, variable, orders)


def severity_summary(
    df: pd.DataFrame,
    variable: str,
    *,
    orders: Mapping[str, Sequence] | None = None,
) -> pd.DataFrame:
    """Severidad promedio por póliza siniestrada: media, mediana y desvío.

    Requiere una cartera previamente filtrada a numero_siniestros > 0,
    como se realiza en la sección 4 del notebook 02.
    """
    summary = (
        df.groupby(variable, observed=True)
        .agg(
            polizas=("severidad_promedio", "size"),
            severidad_media=("severidad_promedio", "mean"),
            severidad_mediana=("severidad_promedio", "median"),
            desviacion=("severidad_promedio", "std"),
        )
        .reset_index()
    )
    return _order_categories(summary, variable, orders)


def cost_summary(
    df: pd.DataFrame,
    variable: str,
    *,
    orders: Mapping[str, Sequence] | None = None,
) -> pd.DataFrame:
    """Costo agregado medio y proporción de pólizas siniestradas."""
    summary = (
        df.groupby(variable, observed=True)
        .agg(
            polizas=("costo_siniestros", "size"),
            costo_medio=("costo_siniestros", "mean"),
            proporcion_con_costo=("tuvo_siniestro", "mean"),
        )
        .reset_index()
    )
    return _order_categories(summary, variable, orders)
