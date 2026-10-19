# CHANGELOG

Historial de versiones de **datos**, **código** y **modelos** (formato basado en *Keep a Changelog*; versiones semánticas por artefacto).

## [modelo-v1.0.0] — 2026-10-18 · Reporte final del experimento

### Resultados
- Experimento MLflow `seguros-costos-regresion`: **109 corridas** (3 padres de Grid Search, 104 candidatos, 1 comparación estadística, 1 ablación).
- Hiperparámetros seleccionados por `GridSearchCV` (K-Fold k=5, refit por RMSE):
  - **Ridge:** `alpha=1.0`, interacciones de grado 1, objetivo sin transformar.
  - **Random Forest:** `n_estimators=400`, `max_depth=6`, `min_samples_leaf=5`, `max_features=0.5`.
  - **XGBoost:** `n_estimators=200`, `max_depth=2`, `learning_rate=0.03`, `subsample=0.8`, `min_child_weight=5`.
- Prueba (20 %, 268 filas): RMSE 3 437 / 3 411 / 3 422 USD y R² 0.918 / 0.919 / 0.919 (Ridge / RF / XGBoost).
- K-Fold repetido 5×5: RMSE 4 722 / 4 751 / 4 719 USD. Friedman p = 0.003; ningún par significativo con t de Nadeau-Bengio + Holm (p ≥ 0.21).
- Ablación: `smoker_obese` redujo el RMSE de la regresión lineal de 6 297 a 4 715 USD (−25 %).
- **Model Registry:** `seguros-costos-regresor` **v1** = XGBoost, alias `champion` (criterio: menor RMSE medio en K-Fold repetido).

### Agregado
- `evidencias/mlflow/`: capturas del panel (corridas anidadas, métricas, parámetros y etiquetas, coordenadas paralelas del grid, artefactos de la comparación y Model Registry).
- `resultados/`: métricas, `cv_results` de cada grid, pruebas estadísticas, importancia por permutación y ablación.
- `requirements-lock.txt` con las versiones exactas de la corrida (Python 3.12.3, scikit-learn 1.9.1, XGBoost 3.4.1, MLflow 3.17.1).
- Conclusiones en el cuaderno, README y este CHANGELOG.

## [código 1.1.1] — 2026-10-18
### Corregido
- `train.py`: MLflow 3.17 serializa scikit-learn con **skops**, que rechazó una función identidad definida en el script (`__main__._identidad`). Se reemplazó por `FunctionTransformer()` estándar y se calcula la lista de tipos de confianza (con respaldo `cloudpickle`). Los tres modelos quedaron guardados con skops.
### Agregado
- `ejecutar.sh --limpio` para borrar corridas fallidas o previas antes de entrenar.

## [código 1.1.0] — 2026-10-18
### Agregado
- `train.py`: Grid Search con validación cruzada para Ridge, Random Forest y XGBoost; corridas hijas por candidato; métricas de CV, entrenamiento y prueba; gráficas; importancia por permutación; comparación estadística (Friedman, Nadeau-Bengio, Wilcoxon, Holm); ablación de variables; registro del campeón en el Model Registry.
- `entrena.ipynb`: cuaderno compatible con Colab que orquesta el flujo y consulta MLflow.
- `ejecutar.sh` y `requirements.txt`.
- Verificación de `libomp` (XGBoost en macOS) y detección de la raíz del repositorio en el cuaderno.

## [código 1.0.0] — 2026-10-18
### Agregado
- `eda.py`: 7 visualizaciones (calidad, distribución del objetivo, costo vs edad, costo vs IMC, boxplots, correlación de Spearman, atípicos de IMC) y `resumen_eda.json`.

## [datos-v1.1.0] — 2026-10-18 · Datos limpios
- `datos/datos_limp/insurance_limpio.csv` — **1 337 filas × 9 columnas**, SHA-256 `456e2ef0…5b65`.
- Generado por `fuentes/datos_prep.py` (versión del pipeline 1.1.0) a partir de `datos-v1.0.0`.
- **Cambios respecto a v1.0.0:**
  - Eliminado 1 duplicado exacto (fila original 581, idéntica a la 195).
  - Validación de esquema y tipos: sin errores de conversión.
  - Nulos: ninguno (se mantiene la lógica de imputación para datos futuros).
  - Reglas de negocio: 0 filas fuera de rango o dominio.
  - Atípicos IQR marcados, no eliminados: 139 en `charges` (10.4 %, 97.8 % fumadores) y 9 en `bmi` (0.67 %).
  - Nuevas variables `obese` (IMC ≥ 30) y `smoker_obese`.
- Pruebas unitarias (`tests/test_datos_prep.py`, 6 casos) para estandarización, nulos, duplicados, reglas y esquema.

## [datos-v1.0.0] — 2026-10-18 · Datos originales
- `datos/datos_ini/insurance.csv` — **1 338 filas × 7 columnas**, SHA-256 `505c1cbc…8281`.
- Fuente: `stedy/Machine-Learning-with-R-datasets` (commit `ff933238`), *Medical Cost Personal Datasets*. Archivo sin modificaciones.
