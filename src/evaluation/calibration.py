import numpy as np
import pandas as pd


def calibration_by_quantile(
    y_true,
    y_pred,
    n_quantiles=10
):
    calibration_df = pd.DataFrame({
        "observado": np.asarray(y_true),
        "predicho": np.asarray(y_pred)
    })

    calibration_df["grupo_riesgo"] = pd.qcut(
        calibration_df["predicho"],
        q=n_quantiles,
        duplicates="drop"
    )

    calibration_table = (
        calibration_df
        .groupby(
            "grupo_riesgo",
            observed=True
        )
        .agg(
            observado_medio=(
                "observado",
                "mean"
            ),
            predicho_medio=(
                "predicho",
                "mean"
            ),
            observaciones=(
                "observado",
                "size"
            )
        )
        .reset_index()
    )

    return calibration_table