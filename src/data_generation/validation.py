"""Comprobaciones actuariales y de consistencia del DGP sintético."""

import numpy as np
import pandas as pd

from .synthetic_insurance import FEATURES, ORACLE


def validate_synthetic_insurance_data(
    portfolio: pd.DataFrame, claims: pd.DataFrame
) -> None:
    """Comprueba dominios, identidades actuariales y conciliación de las bases.

    Genera AssertionError cuando se incumple una condición esperada.
    """
    df, df_claims = portfolio, claims
    assert df[FEATURES].notna().all().all(), "Faltan datos en predictores."
    assert df["policy_id"].is_unique, "Las pólizas deben tener ID único."
    assert df["policy_id"].notna().all()
    assert df_claims["claim_id"].is_unique
    assert not df_claims.duplicated(["policy_id", "claim_number"]).any()

    # Restricciones de las variables simuladas.
    assert df["edad"].between(18, 74).all()
    assert df["zona"].between(1, 5).all()
    assert df["anios_vehiculo"].between(0, 19).all()
    assert df["cobertura"].isin([0, 1, 2]).all()
    assert df["historial_siniestros"].isin([0, 1, 2, 3, 4]).all()
    assert df["tipo_uso"].isin([0, 1, 2]).all()

    assert (df["numero_siniestros"] >= 0).all()
    assert (df["costo_siniestros"] >= 0).all()
    assert (df[ORACLE] > 0).all().all()
    assert df["prob_siniestro_real"].between(0, 1).all()
    assert df["tuvo_siniestro"].equals(
        (df["numero_siniestros"] > 0).astype(int)
    )

    # Identidades teóricas: Poisson, prima pura y costos agregados.
    np.testing.assert_allclose(
        df["prob_siniestro_real"].to_numpy(),
        1 - np.exp(-df["lambda_real"].to_numpy()),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        df["prima_pura_real"].to_numpy(),
        df["lambda_real"].to_numpy()
        * df["severidad_esperada_real"].to_numpy(),
        rtol=1e-12,
    )

    no_claims = df["numero_siniestros"] == 0
    assert (df.loc[no_claims, "costo_siniestros"] == 0).all()
    assert df.loc[no_claims, "severidad_promedio"].isna().all()
    assert df.loc[~no_claims, "severidad_promedio"].notna().all()
    assert (df.loc[~no_claims, "costo_siniestros"] > 0).all()
    assert (df_claims["severidad"] > 0).all()
    assert len(df_claims) == int(df["numero_siniestros"].sum())
    assert df_claims["policy_id"].isin(df["policy_id"]).all()

    # Alineación por policy_id: no dependemos del orden de las filas.
    grouped = df_claims.groupby("policy_id").agg(
        numero_siniestros_claims=("severidad", "size"),
        costo_siniestros_claims=("severidad", "sum"),
        severidad_promedio_claims=("severidad", "mean"),
    )
    with_claims = df.set_index("policy_id").loc[~no_claims.to_numpy()]
    assert set(grouped.index) == set(with_claims.index)
    checked = with_claims.loc[grouped.index]

    np.testing.assert_array_equal(
        grouped["numero_siniestros_claims"].to_numpy(),
        checked["numero_siniestros"].to_numpy(),
    )
    np.testing.assert_allclose(
        grouped["costo_siniestros_claims"].to_numpy(),
        checked["costo_siniestros"].to_numpy(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        grouped["severidad_promedio_claims"].to_numpy(),
        checked["severidad_promedio"].to_numpy(),
        rtol=1e-12,
    )

    # Las características individuales deben concordar con su póliza.
    link = df_claims[["policy_id", *FEATURES, "severidad_esperada_real"]].merge(
        df[["policy_id", *FEATURES, "severidad_esperada_real"]],
        on="policy_id",
        how="left",
        validate="many_to_one",
        suffixes=("_claim", "_policy"),
    )
    for col in FEATURES:
        np.testing.assert_array_equal(
            link[f"{col}_claim"].to_numpy(),
            link[f"{col}_policy"].to_numpy(),
        )
    np.testing.assert_allclose(
        link["severidad_esperada_real_claim"].to_numpy(),
        link["severidad_esperada_real_policy"].to_numpy(),
        rtol=1e-12,
    )
