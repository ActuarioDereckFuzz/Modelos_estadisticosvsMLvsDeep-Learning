import numpy as np

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_poisson_deviance
)


def regression_metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(
            mean_squared_error(y_true, y_pred)
        )
    }


def poisson_metrics(y_true, y_pred):
    metrics = regression_metrics(
        y_true,
        y_pred
    )

    metrics["Poisson Deviance"] = (
        mean_poisson_deviance(
            y_true,
            y_pred
        )
    )

    return metrics


def relative_improvement(
    baseline_value,
    model_value
):
    return (
        1
        - model_value / baseline_value
    )