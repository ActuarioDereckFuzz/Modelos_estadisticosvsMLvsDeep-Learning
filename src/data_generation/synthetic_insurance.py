"""Simulación de pólizas y siniestros para comparar modelos de prima pura.

El proceso generador de datos (DGP) es completamente sintético. Conserva
los parámetros, el orden de muestreo y la semilla del notebook de referencia.
"""

import numpy as np
import pandas as pd

FEATURES = [
    "edad", "zona", "anios_vehiculo", "cobertura",
    "historial_siniestros", "tipo_uso",
]
TARGETS = ["numero_siniestros", "severidad_promedio", "costo_siniestros"]
ORACLE = [
    "lambda_real", "prob_siniestro_real",
    "severidad_esperada_real", "prima_pura_real",
]
CLAIM_FEATURES = FEATURES.copy()
CLAIM_TARGET = "severidad"
CLAIM_ORACLE = ["severidad_esperada_real"]


def _generate_features(rng: np.random.Generator, n: int) -> dict:
    """Simula las seis características observables de las pólizas.

    Se asumen independientes al muestrearlas; las dependencias del riesgo
    aparecen después, mediante las funciones de frecuencia y severidad.
    """
    edad = rng.integers(18, 75, n)
    zona = rng.integers(1, 6, n)
    anios_vehiculo = rng.integers(0, 20, n)
    cobertura = rng.choice([0, 1, 2], n, p=[0.50, 0.30, 0.20])
    historial = rng.choice([0, 1, 2, 3, 4], n, p=[0.50, 0.28, 0.14, 0.05, 0.03])
    uso = rng.choice([0, 1, 2], n, p=[0.50, 0.42, 0.08])
    return {
        "edad": edad,
        "zona": zona,
        "anios_vehiculo": anios_vehiculo,
        "cobertura": cobertura,
        "historial_siniestros": historial,
        "tipo_uso": uso,
    }


def _simulate_frequency(rng: np.random.Generator, x: dict) -> tuple:
    """Genera el número de siniestros mediante una Poisson con enlace log."""
    eta_freq = (
        -2.8
        + 0.18 * x["zona"]
        + 0.30 * x["historial_siniestros"]
        + 0.08 * x["cobertura"]
        + 0.015 * x["anios_vehiculo"]
        - 0.0005 * (x["edad"] - 40) ** 2
        + 0.10 * (x["tipo_uso"] == 2)
        + 0.05 * x["cobertura"] * x["tipo_uso"]
    )
    lambda_freq = np.exp(eta_freq)
    numero_siniestros = rng.poisson(lambda_freq)
    tuvo_siniestro = (numero_siniestros > 0).astype(int)
    prob_siniestro = 1 - np.exp(-lambda_freq)
    return lambda_freq, numero_siniestros, tuvo_siniestro, prob_siniestro


def _simulate_severity(
    rng: np.random.Generator, x: dict, numero_siniestros: np.ndarray
) -> tuple:
    """Genera severidades individuales y agrega los costos a nivel póliza."""
    mu_sev = (
        8.5
        + 0.10 * x["zona"]
        + 0.20 * x["cobertura"]
        + 0.012 * x["anios_vehiculo"]
        + 0.15 * (x["tipo_uso"] == 2)
        + 0.04 * x["cobertura"] * x["zona"]
    )
    sigma_sev = 0.55
    expected_severity = np.exp(mu_sev + sigma_sev ** 2 / 2)

    n = len(numero_siniestros)
    siniestro_total = np.zeros(n)
    severidad_promedio = np.full(n, np.nan)
    claims = []

    # El orden secuencial de las simulaciones mantiene los resultados del DGP.
    for i, n_claims in enumerate(numero_siniestros):
        if n_claims > 0:
            severidades = rng.lognormal(
                mean=mu_sev[i], sigma=sigma_sev, size=n_claims
            )
            siniestro_total[i] = severidades.sum()
            severidad_promedio[i] = severidades.mean()

            for j, severidad in enumerate(severidades, start=1):
                claims.append({
                    "policy_id": i + 1,
                    "claim_number": j,
                    "edad": x["edad"][i],
                    "zona": x["zona"][i],
                    "anios_vehiculo": x["anios_vehiculo"][i],
                    "cobertura": x["cobertura"][i],
                    "historial_siniestros": x["historial_siniestros"][i],
                    "tipo_uso": x["tipo_uso"][i],
                    "severidad": severidad,
                    "severidad_esperada_real": expected_severity[i],
                })
    return siniestro_total, severidad_promedio, expected_severity, claims


def generate_synthetic_insurance_data(
    n_policies: int = 10_000, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve un portafolio por póliza y una tabla de siniestros individuales.

    Args:
        n_policies: Número de pólizas con exposición homogénea.
        seed: Semilla de NumPy para repetir la simulación.

    Returns:
        portfolio: Variables observables, resultados y valores *oracle*.
        claims: Una fila por siniestro individual, con su `policy_id`.

    Advertencia:
        Las variables *oracle* se usan para evaluar el experimento, nunca
        como predictores de los modelos que se comparan posteriormente.
    """
    if not isinstance(n_policies, (int, np.integer)) or n_policies < 1:
        raise ValueError("n_policies debe ser un entero positivo.")

    rng = np.random.default_rng(seed)
    x = _generate_features(rng, n_policies)
    lambda_freq, numero_siniestros, tuvo_siniestro, prob_siniestro = (
        _simulate_frequency(rng, x)
    )
    siniestro_total, severidad_promedio, expected_severity, claims = (
        _simulate_severity(rng, x, numero_siniestros)
    )
    prima_pura_real = lambda_freq * expected_severity

    # Se conservan los nombres y el orden de columnas del notebook original.
    df = pd.DataFrame({
        "policy_id": np.arange(1, n_policies + 1),
        "edad": x["edad"],
        "zona": x["zona"],
        "anios_vehiculo": x["anios_vehiculo"],
        "cobertura": x["cobertura"],
        "historial_siniestros": x["historial_siniestros"],
        "tipo_uso": x["tipo_uso"],
        "numero_siniestros": numero_siniestros,
        "tuvo_siniestro": tuvo_siniestro,
        "severidad_promedio": severidad_promedio,
        "costo_siniestros": siniestro_total,
        "lambda_real": lambda_freq,
        "prob_siniestro_real": prob_siniestro,
        "severidad_esperada_real": expected_severity,
        "prima_pura_real": prima_pura_real,
    })

    df_claims = pd.DataFrame(claims, columns=[
        "policy_id", "claim_number", *FEATURES,
        "severidad", "severidad_esperada_real",
    ])
    # Si se simula un portafolio pequeño sin siniestros, conservar el esquema.
    if df_claims.empty:
        for col in ["policy_id", "claim_number", *FEATURES]:
            df_claims[col] = df_claims[col].astype("int64")
        for col in ["severidad", "severidad_esperada_real"]:
            df_claims[col] = df_claims[col].astype("float64")
    df_claims.insert(0, "claim_id", np.arange(1, len(df_claims) + 1))
    return df, df_claims
