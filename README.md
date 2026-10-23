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

## 10. Estado del proyecto

- Fases 01–05: simulación, exploración y modelos completados.
- Fase 06: comparación global finalizada.
- Fase 07: estrategias de tarificación pendientes.
- Fase 08: documentación y cierre técnico en curso.

La documentación de instalación, estructura exacta del repositorio y comandos de reproducción se completará después de verificar el entorno y las dependencias finales.

---

**Tecnologías principales:** Python, NumPy, pandas, SciPy, statsmodels, scikit-learn, XGBoost, TensorFlow/Keras, Keras Tuner y Jupyter Notebook.