# Statistical vs. Machine Learning vs. Deep Learning for Pure Premium Estimation

**Comparación de modelos estadísticos, Machine Learning y Deep Learning para la estimación de prima pura en seguros de automóviles.**

## Descripción

Este proyecto desarrolla un experimento actuarial reproducible para comparar modelos estadísticos clásicos, algoritmos de Machine Learning y redes neuronales en tres problemas fundamentales:

- Frecuencia de siniestros.
- Severidad de siniestros.
- Prima pura o costo esperado de siniestralidad.

La pregunta central es:

**¿Cuánto valor adicional aporta la complejidad de Machine Learning y Deep Learning frente a modelos estadísticos correctamente especificados?**

La evaluación no se limita a minimizar errores predictivos. También considera calibración, recuperación del riesgo verdadero, interpretabilidad, estabilidad y complejidad computacional.

## 1. Diseño experimental

Se construyó una cartera sintética de **10,000 pólizas** de seguros de automóviles mediante un proceso generador de datos conocido (*Data Generating Process*, DGP).

Las variables explicativas incluyen características del conductor, vehículo, cobertura, zona geográfica, historial de siniestralidad y tipo de uso.

### Frecuencia

El número de siniestros se genera mediante:

$$
N_i\mid X_i\sim\operatorname{Poisson}(\lambda_i)
$$

donde $\lambda_i$ representa la frecuencia esperada verdadera de la póliza.

### Severidad

Los montos individuales de siniestros siguen una distribución lognormal:

$$
\log(S_{ij})\mid X_i\sim\mathcal N(\mu_{S,i},\sigma^2)
$$

La esperanza condicional verdadera considera la corrección de la transformación logarítmica:

$$
E[S\mid X_i]=\exp(\mu_{S,i}+\sigma^2/2)
$$

### Prima pura

La prima pura teórica se define como:

$$
PP_i=\lambda_i E[S\mid X_i]
$$

El conocimiento del proceso generador permite comparar las predicciones con las realizaciones observadas y con el valor esperado verdadero (*oracle*).

## 2. Modelos implementados

| Familia | Frecuencia | Severidad | Prima pura |
|---|---|---|---|
| Estadísticos | GLM Poisson | GLM Gamma y Log + Duan | Poisson × Gamma y GLM Tweedie |
| Machine Learning | Random Forest y XGBoost | Random Forest y XGBoost | Frecuencia × Severidad y modelos directos |
| Deep Learning | MLP Poisson | MLP Gamma | MLP Frecuencia × Severidad y MLP Tweedie |

Se estudian dos estrategias para la prima pura:

**Modelado descompuesto:** estimar independientemente frecuencia y severidad, y multiplicar sus esperanzas condicionales.

**Modelado directo:** predecir el costo agregado esperado mediante una única función.

## 3. Entrenamiento y validación

Se utiliza una partición maestra común:

| Conjunto | Pólizas |
|---|---:|
| Entrenamiento | 8,000 |
| Prueba | 2,000 |

La división se realiza por póliza. Los registros individuales de siniestros heredan la partición correspondiente a su `policy_id`, evitando que siniestros de una misma póliza aparezcan simultáneamente en entrenamiento y prueba.

Los modelos utilizan estrategias de validación y optimización apropiadas para cada familia:

- GLM: selección y diagnóstico estadístico.
- Machine Learning: validación cruzada y búsqueda de hiperparámetros.
- Deep Learning: validación interna, optimización bayesiana y control del entrenamiento mediante early stopping cuando corresponde.

Los modelos finales no utilizan el test para ajustar parámetros ni seleccionar hiperparámetros. Sin embargo, sus resultados se inspeccionaron durante el desarrollo; esta limitación se considera en las interpretaciones.

## 4. Evaluación

Se consideran dos perspectivas complementarias.

### Error frente a realizaciones observadas

Se utilizan MAE, RMSE y deviances apropiadas para cada distribución.

Estas métricas describen la capacidad predictiva frente al conteo, severidad o costo efectivamente observado, sujetos a variabilidad aleatoria.

### Recuperación del riesgo verdadero

Gracias al DGP conocido, es posible calcular:

- Oracle MAE.
- Oracle RMSE.
- Correlaciones con la esperanza condicional verdadera.
- Calibración y discriminación del riesgo.

De esta forma, se distingue el error de estimación del modelo del ruido aleatorio de la siniestralidad observada.

Para las comparaciones de prima pura se utiliza un parámetro Tweedie común de $p=1.25$, determinado utilizando información de entrenamiento.

## 5. Principales hallazgos

Los experimentos muestran que una mayor complejidad no garantiza mejores estimaciones del riesgo.

En el escenario simulado, los modelos estadísticos presentan una recuperación especialmente precisa de las esperanzas condicionales verdaderas, favorecidos por una especificación cercana al proceso generador de datos.

Random Forest, XGBoost y las redes neuronales ofrecen alternativas competitivas para predecir la siniestralidad observada, aunque sus mejoras no necesariamente se traducen en una mejor recuperación del oracle.

El análisis también muestra que diferencias reducidas entre métricas de test deben interpretarse considerando la incertidumbre de las realizaciones observadas y las características del diseño experimental.

Estas conclusiones corresponden al DGP estudiado y no establecen una superioridad universal de los GLM frente a Machine Learning o Deep Learning.

## 6. Modelos reutilizables

Los modelos finales y sus transformaciones de entrada se conservaron para permitir inferencia sobre nuevas carteras sin reentrenamiento.

Se incluyen los modelos estadísticos, seis pipelines finales de Random Forest/XGBoost y tres redes neuronales MLP con sus preprocesadores.

La reutilización de artefactos permite separar el proceso de entrenamiento del proceso de predicción, manteniendo la consistencia de las transformaciones.

## 7. Archivos de resultados

Los notebooks de modelado exportan tablas de métricas y predicciones a `results/tables/`.

Las predicciones deben relacionarse mediante sus identificadores:

- `policy_id` para predicciones a nivel póliza.
- `claim_id` para predicciones a nivel de siniestro.

No se presupone que dos archivos conserven el mismo orden de filas.

## 8. Limitaciones

Este es un estudio experimental con datos sintéticos y un proceso generador conocido.

Las principales limitaciones incluyen:

- Los resultados dependen del DGP y de los supuestos de simulación.
- El tamaño de muestra disponible limita la precisión de algunos modelos.
- La especificación estadística guarda una relación estrecha con el proceso generador.
- Los protocolos de entrenamiento difieren entre familias, aunque comparten conjuntos de evaluación.
- El desempeño predictivo no equivale directamente a rentabilidad comercial.
- Las conclusiones no sustituyen una validación externa con datos reales.

## 9. Extensión: estrategias de tarificación

**En desarrollo.**

La siguiente fase contempla simular nuevas carteras, generar una disposición máxima a pagar por asegurado y evaluar estrategias de precios conservadoras, competitivas y agresivas.

Se estudiarán conversión, selección de riesgos, siniestralidad y rentabilidad esperada y realizada, manteniendo separado el modelo de demanda del oracle actuarial.

## 10. Resultados cuantitativos seleccionados

Los siguientes valores proceden de las predicciones congeladas en `results/tables/` y se han recalculado independientemente durante la auditoría. Todos corresponden al mismo conjunto test y a sus respectivas unidades de evaluación.

| Problema | Mejor deviance observada | Modelo | Mejor Oracle RMSE | Modelo |
|---|---:|---|---:|---|
| Frecuencia (por póliza) | Poisson: 0.587152 | GLM Poisson | 0.017865 | GLM Poisson |
| Severidad (por claim) | Gamma: 0.315653 | Log-severidad + Duan | 1,043.530 | GLM Gamma |
| Prima pura (por póliza) | Tweedie: 1,159.819 | XGBoost directo | 280.871 | GLM Tweedie |

El menor MAE observado para prima pura fue **3,170.16 (MLP Tweedie)**, mientras su Oracle RMSE fue **636.74**, frente a **280.87** del GLM Tweedie. El notebook 06 incorpora bootstrap exploratorio pareado, calibración y discusión de incertidumbre. Estas diferencias numéricas no equivalen a significancia universal ni validación externa.

## 11. Estructura del repositorio

```text
.
├── data/
│   ├── raw/                   # Cartera sintética y siniestros
│   └── processed/             # Splits reproducibles
├── notebooks/                 # 01 a 06, resultados ejecutados
├── src/
│   ├── data_generation/       # Proceso generador y validaciones
│   ├── evaluation/            # Métricas y diagnósticos
│   └── models/                # Predicción y carga de modelos persistidos
├── models/
│   ├── statistical/           # Bundle GLM/OLS
│   ├── machine_learning/      # 6 pipelines y hashes SHA256
│   └── deep_learning/v1/     # MLP finales, preprocesadores y manifiesto
├── results/
│   ├── tables/                # Métricas, predicciones y comparación
│   └── tuning/                # Evidencia histórica de HPO
├── scripts/
│   ├── verify_environment.py  # Comprueba versiones de entrenamiento
│   └── smoke_inference.py     # Carga e inferencia sin entrenamiento
├── tests/                     # Suite de integridad y validación
├── README.md                  # Única documentación central del proyecto
├── requirements.txt           # Dependencias compatibles con los artefactos
├── environment.yml           # Entorno conda reproducible
└── .gitignore
```

## 12. Instalación del entorno

**Requisitos:** Windows 11, Linux o WSL2 con [Conda/Miniconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) instalado, conexión a internet para descargar dependencias y espacio suficiente para TensorFlow. Se recomienda **Python 3.11.16** (el entorno lo instala). La ejecución con CPU es suficiente para comparar resultados y cargar modelos; CUDA/GPU no es obligatoria.

Desde una terminal **ubicada en la raíz del repositorio**, ejecutar:

```bash
conda env create -f environment.yml
conda activate GLMvsML
python -m pip check
python scripts/verify_environment.py
```

Si el entorno ya existe y cambian las dependencias:

```bash
conda env update -f environment.yml --prune
conda activate GLMvsML
```

### Versiones que deben respetarse

Las siguientes versiones fueron extraídas directamente de los manifiestos de los modelos serializados (no son sugerencias arbitrarias):

| Paquete | Versión de entrenamiento |
|---|---|
| Python | 3.11.16 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| scikit-learn | 1.9.1 |
| XGBoost | 3.3.0 |
| joblib | 1.6.0 |
| statsmodels | 0.15.0 |
| patsy | 1.0.3 |
| TensorFlow | 2.21.0 |
| Keras Tuner | 1.4.8 |
| TensorBoard | 2.21.0 |

`pyarrow` también se instala: el bundle estadístico guarda arrays de texto de pandas respaldados por Arrow, por lo que puede fallar al deserializar si falta esta dependencia. Para paquetes no registrados en el entrenamiento se permiten rangos acotados en `requirements.txt`. Este proyecto **no ofrece todavía un lockfile transitivo con hashes de instalación**. La verificación de versiones es necesaria, pero solo la prueba de inferencia comprueba que los artefactos realmente pueden usarse.

### Pruebas y carga de modelos, sin reentrenar

```bash
# Controles sobre resultados, datasets y hashes congelados
python -m pytest -q

# Comprobar carga, transformaciones y predicciones sobre 3 pólizas NUEVAS
python scripts/smoke_inference.py

# Si solo se desea probar GLM y ML:
python scripts/smoke_inference.py --skip-dl
```

El smoke test produce estimaciones para tres perfiles con las seis variables explicativas, sin targets, valores oracle ni acceso a información futura. Si un paso falla, **no reentrenar automáticamente**: revisar versiones, artefactos y los mensajes de error.

### Abrir los notebooks

Registrar opcionalmente un kernel de Jupyter y lanzar el servidor desde la raíz:

```bash
python -m ipykernel install --user --name GLMvsML --display-name "Python (GLMvsML)"
jupyter lab
```

Abrir los notebooks de `notebooks/` en orden **01 → 06** y seleccionar el kernel `Python (GLMvsML)`. Para inspeccionar resultados existentes no es necesario ejecutar todo otra vez: los CSV están en `results/tables/`. **Volver a ejecutar 01, 03, 04 o 05 puede sobrescribir datasets, modelos, métricas o registros congelados**; usar una copia de trabajo si se pretende reconstruir el experimento. En el notebook 05 preferir `ARTIFACT_MODE = "reuse"` al inspeccionar resultados previamente almacenados.

### TensorFlow y GPU

TensorFlow 2.21.0 puede utilizarse en **CPU**, suficiente para inferencia. Para GPU NVIDIA en Windows, emplear **WSL2 con Linux** y configurar el conjunto de controladores CUDA/cuDNN requerido por TensorFlow. No se afirma que el archivo `environment.yml` instale o detecte CUDA automáticamente. Los resultados congelados no requieren una GPU para ser consultados.

## 13. Inferencia sobre una cartera nueva

Las tres familias tienen cargadores ya implementados en `src/models/`. Este ejemplo usa los seis predictores de la simulación, **sin siniestralidad observada ni información oracle**:

```python
import pandas as pd
from src.models.statistical_predictor import load_statistical_bundle, predict_statistical_models
from src.models.ml_persistence import load_ml_models, predict_ml_pure_premium
from src.models.dl_inference import FrozenDLPricing

nuevas_polizas = pd.DataFrame([{
    "edad": 42,
    "zona": 3,
    "anios_vehiculo": 6,
    "cobertura": 2,
    "historial_siniestros": 1,
    "tipo_uso": 0,
}])

bundle = load_statistical_bundle("models/statistical/statistical_models_v1.joblib")
pred_glm = predict_statistical_models(nuevas_polizas, bundle, model="all")

modelos_ml, manifiesto = load_ml_models("models/machine_learning")
pred_ml = predict_ml_pure_premium(nuevas_polizas, modelos_ml, manifiesto["feature_order"])

predictor_dl = FrozenDLPricing(project_root=".", version="v1")
pred_dl = predictor_dl.predict(nuevas_polizas)

print(pred_glm)
print(pred_ml)
print(pred_dl)
```

**Interpretación:** la frecuencia es el número esperado de siniestros, la severidad es el costo esperado *por siniestro* y la prima pura es el costo agregado esperado *por póliza*. No se consideran gastos, margen, impuestos, elasticidad de demanda ni estrategias comerciales. Cargar archivos `joblib` únicamente desde el repositorio propio y fuentes confiables: pickle puede ejecutar código durante la deserialización. Los SHA256 ayudan a detectar modificaciones de archivos respecto de sus manifiestos, pero no autentican a terceros.

## 15. Estado y trabajo futuro

- **01–06:** generación, EDA, modelos estadísticos, ML, DL y comparación global realizados.
- **07 — Estrategias de tarificación:** pendiente. Incluirá nuevas carteras, disposición máxima a pagar, conversión, selección de riesgos y rentabilidad.
- **08 — Cierre:** README unificado, manifiestos y comprobaciones de integridad preparados. **Pendiente**: validar instalación e inferencia completa en un Python 3.11.16 limpio y luego incorporar el alcance de la fase 07.

---

**Tecnologías principales:** Python, pandas, NumPy, SciPy, statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, Keras Tuner, Matplotlib, Seaborn, Jupyter.
