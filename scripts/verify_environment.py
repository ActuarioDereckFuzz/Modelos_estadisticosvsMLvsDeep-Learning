"""Verifica versiones del entorno, sin cargar modelos ni reentrenar.

Ejecutar desde la raíz del repositorio:
    python scripts/verify_environment.py

Código 0: versiones fijadas y dependencias disponibles.
Código 1: falta alguna o una versión no coincide con los artefactos.
"""
from __future__ import annotations

from importlib import metadata
import platform
import sys


# Extraídas de models/machine_learning/manifest.json y el bundle GLM.
EXACT = {
    "numpy": "2.4.6",
    "pandas": "3.0.3",
    "scikit-learn": "1.9.1",
    "xgboost": "3.3.0",
    "joblib": "1.6.0",
    "statsmodels": "0.15.0",
    "patsy": "1.0.3",
    "tensorflow": "2.21.0",
    "keras-tuner": "1.4.8",
    "tensorboard": "2.21.0",
}
OTHER_REQUIRED = ["scipy", "matplotlib", "seaborn", "pyarrow", "jupyterlab", "ipykernel", "pytest"]


def main() -> int:
    print(f"Python: {sys.version.split()[0]} | {platform.system()} {platform.machine()}")
    errors = []
    if sys.version_info[:3] != (3, 11, 16):
        errors.append("Usar Python 3.11.16 para el entorno de inferencia congelada")

    for name, expected in EXACT.items():
        try:
            current = metadata.version(name)
        except metadata.PackageNotFoundError:
            current = "NO INSTALADO"
        status = "OK" if current == expected else "REVISAR"
        print(f"[{status:7}] {name:15}: {current} (esperado {expected})")
        if status != "OK":
            errors.append(f"{name}: {current} != {expected}")

    for name in OTHER_REQUIRED:
        try:
            current = metadata.version(name)
        except metadata.PackageNotFoundError:
            current = "NO INSTALADO"
            errors.append(f"Falta {name}")
        print(f"[{'OK' if current != 'NO INSTALADO' else 'REVISAR':7}] {name:15}: {current}")

    if errors:
        print("\nEntorno INCOMPATIBLE con la referencia congelada:")
        for error in errors:
            print(" -", error)
        return 1
    print("\nEntorno consistente con las versiones declaradas (aún falta comprobar inferencia).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
