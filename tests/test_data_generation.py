"""Pruebas esenciales del generador sintético de prima pura."""

import pandas as pd

from src.data_generation.synthetic_insurance import generate_synthetic_insurance_data
from src.data_generation.validation import validate_synthetic_insurance_data


def test_seed_42_reproduce_y_valida():
    a, ca = generate_synthetic_insurance_data(n_policies=10_000, seed=42)
    b, cb = generate_synthetic_insurance_data(n_policies=10_000, seed=42)

    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_frame_equal(ca, cb)
    validate_synthetic_insurance_data(a, ca)

    assert len(a) == 10_000
    assert len(ca) == 1_554
    assert int(a["tuvo_siniestro"].sum()) == 1_423
    assert round(a["prima_pura_real"].mean(), 2) == 2_007.02
