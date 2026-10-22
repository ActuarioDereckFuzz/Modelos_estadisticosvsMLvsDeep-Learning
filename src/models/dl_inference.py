"""Inferencia de prima pura con redes Deep Learning congeladas (notebook 05).

Uso desde la raíz del repositorio:

    from src.models.dl_inference import FrozenDLPricing
    predictor = FrozenDLPricing(project_root='.', version='v1')
    estimated_risk = predictor.predict(nuevas_polizas)

Este módulo NO entrena, modifica ni guarda artefactos. Necesita los modelos
previamente sellados por el notebook 05; admite manifiestos v1 generados en
results/models/deep_learning/ después de mover la carpeta completa a
models/deep_learning/, sin modificar pesos, registros ni hashes.

Entradas: columnas explicativas usadas durante el entrenamiento (nada de targets).
Salidas: estimaciones técnicas, NO tarifas comerciales.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def _normalize_manifest_paths(file_hashes: dict[str, str]) -> dict[str, str]:
    """Traduce rutas del manifiesto heredado sin modificar sus hashes.

    Compatible con claves grabadas en Windows (\\) o POSIX (/). La traducción
    solo ocurre en memoria y coincide con la del notebook 05 revisado.
    """
    old_prefix = 'results/models/deep_learning/'
    new_prefix = 'models/deep_learning/'
    normalized = {}
    for original_path, expected_hash in file_hashes.items():
        relative = original_path.replace('\\', '/')
        if relative.startswith(old_prefix):
            relative = new_prefix + relative[len(old_prefix):]
        if relative in normalized:
            raise RuntimeError('Ruta duplicada en el manifiesto: ' + relative)
        normalized[relative] = expected_hash
    return normalized


class FrozenDLPricing:
    """Carga 3 MLPs finales y sus 3 preprocesadores, con control de hashes."""

    def __init__(self, project_root: str | Path, version: str = 'v1') -> None:
        # La versión solo puede ser un nombre de subdirectorio simple.
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', version):
            raise ValueError('Versión no válida')
        self.root = Path(project_root).resolve()
        self.version = version
        self.base = self.root / 'models' / 'deep_learning' / version
        manifest_path = self.base / 'freeze_manifest.json'
        if not manifest_path.is_file():
            raise FileNotFoundError('El experimento no está sellado: ' + str(manifest_path))
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        self._manifest_signature = manifest['signature']
        self._hashes = _normalize_manifest_paths(manifest['files'])

        # Valida los artefactos antes de cargarlos. joblib/pickle debe provenir
        # exclusivamente del repositorio propio o una ubicación confiable.
        def safe_path(relative: str) -> Path:
            key = relative.replace('\\', '/')
            # El manifiesto se normaliza al cargarlo. Sus hashes quedan intactos.
            expected = self._hashes.get(key)
            if expected is None:
                raise RuntimeError('Artefacto fuera del manifiesto: ' + relative)
            full = (self.root / key).resolve()
            if not full.is_relative_to(self.root):
                raise RuntimeError('Ruta fuera de la raíz del proyecto')
            if not full.is_file() or _sha256(full) != expected:
                raise RuntimeError('Hash inválido / artefacto ausente: ' + str(full))
            return full

        def get_meta(label: str) -> dict:
            item = self.base.relative_to(self.root).as_posix() + '/records/' + label + '.json'
            meta = json.loads(safe_path(item).read_text(encoding='utf-8'))
            if meta['signature'] != self._manifest_signature:
                raise RuntimeError('Firma de entrada incompatible: ' + label)
            return meta['value']

        prefix = self.base.relative_to(self.root).as_posix()
        self.prep_frequency = joblib.load(safe_path(prefix + '/preprocessors/frequency_refit_train.joblib'))
        self.prep_severity = joblib.load(safe_path(prefix + '/preprocessors/severity_claim_train.joblib'))
        self.prep_tweedie = joblib.load(safe_path(prefix + '/preprocessors/dl_train.joblib'))
        self.features = list(self.prep_frequency.feature_names_in_)
        if (list(self.prep_severity.feature_names_in_) != self.features
                or list(self.prep_tweedie.feature_names_in_) != self.features):
            raise RuntimeError('El orden de las variables difiere entre transformadores')

        self.severity_scale = float(get_meta('severity_scale')['value'])
        self.pure_premium_scale = float(get_meta('pure_premium_scale')['value'])
        if self.severity_scale <= 0 or self.pure_premium_scale <= 0:
            raise RuntimeError('Escalas objetivo incorrectas')

        # Import tardío: pueden importarse helpers sin cargar TensorFlow/GPU.
        from tensorflow import keras

        def network(name: str):
            item = prefix + '/networks/' + name + '.keras'
            path = safe_path(item)
            metadata = get_meta(name + '_training')
            if _sha256(path) != metadata['model_sha256']:
                raise RuntimeError('Modelo corrupto o cambiado: ' + name)
            # Los losses custom son necesarios al entrenar, NO al predecir.
            fitted_model = keras.models.load_model(path, compile=False)
            if fitted_model.count_params() != metadata['parameter_count']:
                raise RuntimeError('Parámetros del modelo cambiaron: ' + name)
            return fitted_model

        self.frequency_model = network('frequency_final')
        self.severity_model = network('severity_final')
        self.tweedie_model = network('pure_premium_final')

    def predict(self, policies: pd.DataFrame) -> pd.DataFrame:
        """Predice frecuencia, severidad y primas puras a partir de nuevas pólizas."""
        if not isinstance(policies, pd.DataFrame):
            raise TypeError('Se espera un pandas.DataFrame')
        missing = sorted(set(self.features) - set(policies.columns))
        if missing:
            raise ValueError('Variables faltantes: ' + ', '.join(missing))
        if policies[self.features].isna().any().any():
            raise ValueError('Hay variables explicativas nulas')
        if policies.empty:
            raise ValueError('La cartera está vacía')

        x_frequency = self.prep_frequency.transform(policies[self.features]).astype(np.float32)
        x_severity = self.prep_severity.transform(policies[self.features]).astype(np.float32)
        x_tweedie = self.prep_tweedie.transform(policies[self.features]).astype(np.float32)

        frequency = self.frequency_model.predict(x_frequency, verbose=0).reshape(-1)
        severity = (self.severity_model.predict(x_severity, verbose=0).reshape(-1)
                    * self.severity_scale)
        direct = (self.tweedie_model.predict(x_tweedie, verbose=0).reshape(-1)
                  * self.pure_premium_scale)
        result = pd.DataFrame({
            'freq_mlp': frequency,
            'sev_mlp': severity,
            'pp_mlp_fs': frequency * severity,
            'pp_mlp_tweedie': direct,
        }, index=policies.index)
        if 'policy_id' in policies:
            result.insert(0, 'policy_id', policies['policy_id'].to_numpy())
        predictions = result[['freq_mlp', 'sev_mlp', 'pp_mlp_fs', 'pp_mlp_tweedie']]
        if not np.isfinite(predictions.to_numpy()).all() or (predictions <= 0).any().any():
            raise RuntimeError('Predicciones no finitas o no positivas')
        return result
