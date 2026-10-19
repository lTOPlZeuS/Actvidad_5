# Actividad 5 — Preparación de datos, entrenamiento y registro con MLflow

**Gestión de proyectos de inteligencia artificial · Universidad Tecmilenio**

Proyecto de **regresión**: predecir el costo anual del seguro médico (`charges`, USD) a partir de edad, sexo, IMC, hijos, tabaquismo y región. Se comparan **Ridge**, **Random Forest** y **XGBoost** con `GridSearchCV` + validación cruzada K-Fold, y todo el ciclo queda registrado en **MLflow** (parámetros, métricas, artefactos, datasets, modelos y Model Registry).

| Modelo | CV RMSE (k=5) | RMSE prueba | MAE prueba | R² prueba |
|---|---:|---:|---:|---:|
| Ridge (lineal) | 4 715 ± 651 | 3 437 | 2 137 | 0.918 |
| Random Forest | 4 726 ± 668 | 3 411 | 2 166 | 0.919 |
| **XGBoost** (`champion`) | **4 701 ± 657** | 3 422 | **2 136** | 0.919 |

Los tres modelos son estadísticamente equivalentes (t de Nadeau-Bengio con corrección de Holm, p > 0.2). Detalle en [`resultados/`](resultados/) y en el cuaderno [`fuentes/entrena.ipynb`](fuentes/entrena.ipynb).

---

## Estructura

```
Actividad5/
├── datos/
│   ├── datos_ini/insurance.csv          # datos originales (v1.0.0, sin modificar)
│   ├── datos_limp/insurance_limpio.csv  # datos limpios (v1.1.0)
│   ├── datos_limp/bitacora_limpieza.json# qué hizo cada regla de limpieza
│   └── MANIFEST.json                    # versión, filas, columnas y SHA-256 de cada dataset
├── fuentes/
│   ├── entrena.ipynb    # cuaderno (Colab/local): calidad, limpieza, EDA, entrenamiento, evidencia MLflow
│   ├── datos_prep.py    # funciones de limpieza, estandarización y versionamiento
│   ├── eda.py           # visualizaciones de calidad, patrones y anomalías
│   └── train.py         # Grid Search + CV, comparación estadística y registro en MLflow
├── tests/test_datos_prep.py # pruebas unitarias de las reglas de limpieza
├── evidencias/
│   ├── eda/             # gráficas del análisis exploratorio
│   ├── modelos/         # predicho vs real, residuos, importancia, comparaciones
│   └── mlflow/          # capturas del panel de MLflow
├── resultados/          # métricas, cv_results del grid, pruebas estadísticas, ablación
├── mlflow.db, mlruns/   # backend de seguimiento (SQLite) y artefactos de MLflow
├── ejecutar.sh          # reproduce todo con un comando
├── requirements.txt     # dependencias (rangos) · requirements-lock.txt: versiones exactas usadas
├── README.md            # esta guía
└── CHANGELOG.md         # historial de versiones de datos, código y modelos
```

## Cómo reproducir

**Requisitos:** Python ≥ 3.10 y Git. En macOS, XGBoost necesita OpenMP: `brew install libomp`.

```bash
git clone https://github.com/lTOPlZeuS/Actividad5.git
cd Actividad5
bash ejecutar.sh --limpio --ui
```

`ejecutar.sh` crea `.venv`, instala dependencias, corre las pruebas, limpia los datos, genera el EDA, ejecuta el cuaderno (que lanza `train.py`) y abre MLflow en <http://127.0.0.1:5000>. `--limpio` borra corridas previas para obtener un experimento limpio.

Paso a paso, sin el script:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # versiones exactas de la corrida reportada
python -m pytest -q tests
python fuentes/datos_prep.py              # datos_ini -> datos_limp + MANIFEST.json
python fuentes/eda.py                     # evidencias/eda/
python fuentes/train.py                   # entrenamiento + MLflow (≈ 4 min en una Mac M-series)
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

**Google Colab:** abre `fuentes/entrena.ipynb`; la primera celda clona el repositorio e instala dependencias.

**Reproducibilidad:** semilla 42 en particiones, pliegues y modelos; partición 80/20 estratificada por `smoker`; `requirements-lock.txt` fija las versiones (Python 3.12, scikit-learn 1.9.1, XGBoost 3.4.1, MLflow 3.17.1).

## Decisiones principales

### Datos
- **Fuente:** *Medical Cost Personal Datasets* (B. Lantz, *Machine Learning with R*), réplica pública en GitHub (`stedy/Machine-Learning-with-R-datasets`, commit `ff93323`) y Kaggle (`mirichoi0218/insurance`). 1 338 registros, 7 columnas.
- **Limpieza** (`datos_prep.py`): validación de esquema y tipos → estandarización de texto → nulos (mediana en numéricas, moda en categóricas; se descartan filas sin objetivo) → duplicados exactos → reglas de negocio (edad 18–64, IMC 10–70, hijos 0–10, costo > 0, dominios cerrados de `sex`, `smoker`, `region`). Resultado: se eliminó 1 duplicado.
- **Atípicos:** se marcan con 1.5·IQR pero **no se eliminan**: el 97.8 % de los costos atípicos son de fumadores, es decir, casos reales.
- **Variables derivadas:** `obese` (IMC ≥ 30, criterio OMS) y `smoker_obese`, motivadas por las tres bandas de costo del EDA.
- **Codificación y escalado** (One-Hot con `drop="first"`, `StandardScaler`) **dentro del Pipeline**, para que se ajusten solo con los datos de cada pliegue y no haya fuga de información.

### Diccionario de datos

| Variable | Tipo | Descripción |
|---|---|---|
| `age` | entero | Edad del asegurado principal (18–64) |
| `sex` | categórica | `female` / `male` |
| `bmi` | real | Índice de masa corporal (kg/m²) |
| `children` | entero | Dependientes cubiertos (0–5) |
| `smoker` | categórica | `yes` / `no` |
| `region` | categórica | Región residencial en EE. UU.: `northeast`, `northwest`, `southeast`, `southwest` |
| `charges` | real | **Objetivo.** Costo médico anual facturado (USD) |
| `obese` | binaria | Derivada: 1 si `bmi ≥ 30` |
| `smoker_obese` | binaria | Derivada: 1 si fumador y `bmi ≥ 30` |

**Supuestos:** cada fila es una póliza individual e independiente; los costos están en USD de un mismo periodo; el dataset no incluye condiciones médicas preexistentes, por lo que parte de la varianza no es explicable con estas variables.

### Modelos y métricas
- **Ridge** (con transformación opcional log del objetivo e interacciones de grado 2), **Random Forest** y **XGBoost**; espacios de búsqueda de 20, 36 y 48 combinaciones.
- `GridSearchCV` con `KFold(k=5, shuffle=True)`, `refit` por RMSE. Métricas: **RMSE** (principal, penaliza errores grandes en dólares), MAE, R² y MAPE.
- **Comparación estadística:** K-Fold repetido 5×5 con los mismos pliegues para todos los modelos; Friedman global y, por pares, t corregida de Nadeau-Bengio y Wilcoxon con corrección de Holm.

## MLflow

- **Experimento:** `seguros-costos-regresion` (109 corridas).
- **Organización:** una corrida padre por modelo (`<modelo>-gridsearch`) con una corrida hija por combinación del grid (`<modelo>-cand-NN`), más `comparacion-estadistica` y `ablacion-variables-derivadas`.
- **Por corrida padre:** parámetros elegidos, métricas de CV/entrenamiento/prueba, datasets de entrenamiento y prueba (`mlflow.log_input`), gráficas, `cv_results`, importancia por permutación y modelo con firma y ejemplo de entrada.
- **Etiquetas de trazabilidad:** `git_commit`, `datos_version`, `datos_sha256`, `semilla`, versiones de Python/scikit-learn/MLflow y `formato_serializacion`.
- **Model Registry:** `seguros-costos-regresor` v1, alias `champion`.

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
modelo = mlflow.pyfunc.load_model("models:/seguros-costos-regresor@champion")
```

Capturas del panel en [`evidencias/mlflow/`](evidencias/mlflow/).

> Las rutas de artefactos en `mlflow.db` son absolutas (`/Users/jesus/Desktop/Actividad5/mlruns`). En otra ubicación, ejecuta `bash ejecutar.sh --limpio` para regenerar el experimento.

## Versionamiento

- **Datos:** etiquetas Git `datos-v1.0.0` (originales) y `datos-v1.1.0` (limpios), huella SHA-256 en `datos/MANIFEST.json` y registro del dataset en cada corrida de MLflow.
- **Modelo:** etiqueta `modelo-v1.0.0` sobre el commit con los resultados del experimento; versión 1 en el Model Registry.
- **Historial:** ver [`CHANGELOG.md`](CHANGELOG.md).
