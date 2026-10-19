"""
train.py — Entrenamiento, ajuste de hiperparámetros (GridSearchCV + K-Fold),
comparación estadística y registro de experimentos en MLflow.

Problema: regresión del costo anual del seguro médico (charges, USD).
Modelos:  1) Regresión lineal regularizada (Ridge) con transformación log
          2) Random Forest
          3) XGBoost

Uso (desde la raíz del repositorio):

    python fuentes/train.py                     # todo con valores por defecto
    python fuentes/train.py --modelos ridge xgb # solo algunos modelos
    mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

Todo se registra en el experimento "seguros-costos-regresion":
  * una corrida padre por modelo con corridas hijas (una por combinación del
    grid) -> parámetros y métricas de validación cruzada;
  * métricas finales en el conjunto de prueba, gráficas y el modelo serializado;
  * una corrida "comparacion-estadistica" con la prueba de Friedman, t de
    Nadeau-Bengio y Wilcoxon (corrección de Holm) sobre K-Fold repetido;
  * el mejor modelo se registra en el Model Registry con el alias "champion".
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from scipy import stats
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, PolynomialFeatures

sys.path.insert(0, str(Path(__file__).resolve().parent))
import datos_prep as dp  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RAIZ = Path(__file__).resolve().parents[1]
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
COLORES = {"ridge": AZUL, "random_forest": NARANJA, "xgboost": AQUA}
NOMBRES = {"ridge": "Ridge (lineal)", "random_forest": "Random Forest", "xgboost": "XGBoost"}

SCORING = {
    "rmse": "neg_root_mean_squared_error",
    "mae": "neg_mean_absolute_error",
    "r2": "r2",
}


# --------------------------------------------------------------------------
# Definición de modelos y espacios de búsqueda
# --------------------------------------------------------------------------
def transformador_log():
    return FunctionTransformer(np.log1p, inverse_func=np.expm1, check_inverse=False)


def transformador_identidad():
    # func=None equivale a la identidad y no requiere funciones propias, lo que
    # mantiene el modelo serializable con skops (formato seguro de MLflow 3).
    return FunctionTransformer(check_inverse=False)


def definir_modelos(semilla: int) -> dict:
    """Devuelve {clave: (estimador, grid)}.

    Justificación de los rangos:
      Ridge  - alpha en escala log (0.01–100) cubre de casi OLS a fuerte
               regularización; grado 2 de interacciones captura edad×fumador
               e IMC×fumador vistos en el EDA; la transformación log corrige
               la asimetría (1.52 -> -0.09) del objetivo.
      RF     - profundidad y hojas mínimas controlan el sobreajuste en un
               dataset pequeño (1 069 filas de entrenamiento); max_features
               decorrelaciona árboles.
      XGB    - árboles poco profundos (2–4) con learning_rate bajo y más
               estimadores es la configuración estándar contra sobreajuste;
               subsample y min_child_weight añaden regularización.
    """
    from xgboost import XGBRegressor

    ridge = TransformedTargetRegressor(
        regressor=Pipeline([
            ("prep", dp.construir_preprocesador(escalar=True)),
            ("poly", PolynomialFeatures(degree=1, interaction_only=True, include_bias=False)),
            ("modelo", Ridge()),
        ]),
        transformer=transformador_identidad(),
    )
    grid_ridge = {
        "transformer": [transformador_identidad(), transformador_log()],
        "regressor__poly__degree": [1, 2],
        "regressor__modelo__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    }

    rf = Pipeline([
        ("prep", dp.construir_preprocesador(escalar=False)),
        ("modelo", RandomForestRegressor(random_state=semilla, n_jobs=-1)),
    ])
    grid_rf = {
        "modelo__n_estimators": [200, 400],
        "modelo__max_depth": [4, 6, None],
        "modelo__min_samples_leaf": [1, 5, 10],
        "modelo__max_features": [1.0, 0.5],
    }

    xgb = Pipeline([
        ("prep", dp.construir_preprocesador(escalar=False)),
        ("modelo", XGBRegressor(objective="reg:squarederror", random_state=semilla,
                                n_jobs=1, tree_method="hist", verbosity=0)),
    ])
    grid_xgb = {
        "modelo__n_estimators": [200, 500],
        "modelo__max_depth": [2, 3, 4],
        "modelo__learning_rate": [0.03, 0.1],
        "modelo__subsample": [0.8, 1.0],
        "modelo__min_child_weight": [1, 5],
    }
    return {
        "ridge": (ridge, grid_ridge),
        "random_forest": (rf, grid_rf),
        "xgboost": (xgb, grid_xgb),
    }


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def metricas(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mape": float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100),
    }


def legible(v):
    """Convierte valores de hiperparámetros a texto corto para MLflow."""
    if isinstance(v, FunctionTransformer):
        return "log1p" if v.func is np.log1p else "identidad"
    return v


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=RAIZ,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "sin-git"


def log_modelo(estimador, X_ejemplo, y_pred_ejemplo, nombre: str):
    """Registra el modelo en MLflow.

    MLflow 3 serializa scikit-learn con skops, que solo carga tipos de una lista
    de confianza. Se calcula esa lista y se acepta únicamente si todos los tipos
    provienen de bibliotecas conocidas; si skops no puede serializar el pipeline
    (p. ej. XGBoost), se usa cloudpickle como respaldo y se deja constancia en
    la etiqueta 'formato_serializacion'.
    """
    firma = infer_signature(X_ejemplo, y_pred_ejemplo)
    base = dict(sk_model=estimador, signature=firma, input_example=X_ejemplo.head(5))
    permitidos = ("numpy.", "sklearn.", "xgboost.", "scipy.", "builtins.", "collections.")

    intentos = []
    try:
        import skops.io as sio
        no_confiables = sio.get_untrusted_types(data=sio.dumps(estimador))
        if all(t.startswith(permitidos) for t in no_confiables):
            intentos.append(("skops", {"skops_trusted_types": no_confiables}))
    except Exception:
        pass
    intentos.append(("cloudpickle", {"serialization_format": "cloudpickle"}))

    ultimo_error = None
    for formato, extra in intentos:
        try:
            try:  # MLflow >= 3
                info = mlflow.sklearn.log_model(name=nombre, **base, **extra)
            except TypeError:  # MLflow 2.x
                extra2 = {k: v for k, v in extra.items() if k != "skops_trusted_types"}
                info = mlflow.sklearn.log_model(artifact_path=nombre, **base, **extra2)
            mlflow.set_tag("formato_serializacion", formato)
            return info
        except Exception as e:  # noqa: BLE001
            ultimo_error = e
            print(f"  [aviso] no se pudo guardar con {formato}: {str(e)[:160]}")
    raise ultimo_error


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA2, "axes.titlecolor": TINTA,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "legend.frameon": False,
    })


def guardar(fig, ruta: Path):
    fig.tight_layout()
    fig.savefig(ruta, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def grafica_pred_vs_real(y, yp, clave, ruta):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(y, yp, s=14, alpha=0.6, color=COLORES[clave], edgecolor="white", linewidth=0.4)
    lim = [0, max(y.max(), yp.max()) * 1.05]
    ax.plot(lim, lim, color=TINTA2, ls="--", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("charges real (USD)"); ax.set_ylabel("charges predicho (USD)")
    ax.set_title(f"{NOMBRES[clave]}: predicho vs real (prueba)")
    guardar(fig, ruta)


def grafica_residuos(y, yp, clave, ruta):
    r = y - yp
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].scatter(yp, r, s=14, alpha=0.6, color=COLORES[clave], edgecolor="white", linewidth=0.4)
    axes[0].axhline(0, color=TINTA2, ls="--", lw=1)
    axes[0].set_xlabel("charges predicho (USD)"); axes[0].set_ylabel("residuo (USD)")
    axes[0].set_title(f"{NOMBRES[clave]}: residuos vs predicción")
    axes[1].hist(r, bins=40, color=COLORES[clave], edgecolor="white", linewidth=0.6)
    axes[1].set_xlabel("residuo (USD)"); axes[1].set_title("Distribución de residuos")
    guardar(fig, ruta)


def grafica_importancia(imp: pd.DataFrame, clave, ruta):
    imp = imp.sort_values("importancia_media")
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(imp["variable"], imp["importancia_media"], xerr=imp["importancia_std"],
            color=COLORES[clave], height=0.6, error_kw=dict(ecolor=TINTA2, lw=1))
    ax.set_xlabel("Aumento del RMSE al permutar (USD)")
    ax.set_title(f"{NOMBRES[clave]}: importancia por permutación")
    guardar(fig, ruta)


# --------------------------------------------------------------------------
# Pruebas estadísticas
# --------------------------------------------------------------------------
def t_corregida_nadeau_bengio(a, b, n_train, n_test):
    """t de Student corregida para K-Fold repetido (Nadeau & Bengio, 2003).
    Corrige la subestimación de la varianza por solapamiento de pliegues."""
    d = np.asarray(a) - np.asarray(b)
    k = len(d)
    var = np.var(d, ddof=1)
    var_corr = (1 / k + n_test / n_train) * var
    t = d.mean() / np.sqrt(var_corr) if var_corr > 0 else 0.0
    p = 2 * stats.t.sf(np.abs(t), df=k - 1)
    return float(t), float(p)


def holm(pvalores: dict) -> dict:
    orden = sorted(pvalores.items(), key=lambda kv: kv[1])
    m, ajustados, maximo = len(orden), {}, 0.0
    for i, (k, p) in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - i) * p))
        ajustados[k] = maximo
    return ajustados


def comparar_estadisticamente(mejores: dict, X, y, semilla, k, repeticiones, carpeta: Path):
    rkf = RepeatedKFold(n_splits=k, n_repeats=repeticiones, random_state=semilla)
    por_fold = {}
    for clave, est in mejores.items():
        cv = cross_validate(clone(est), X, y, cv=rkf, scoring=SCORING, n_jobs=-1)
        por_fold[clave] = {
            "rmse": -cv["test_rmse"], "mae": -cv["test_mae"], "r2": cv["test_r2"],
        }
    claves = list(mejores)
    n_test = len(X) // k
    n_train = len(X) - n_test

    resumen = []
    for c in claves:
        for m in ["rmse", "mae", "r2"]:
            v = por_fold[c][m]
            ic = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            resumen.append({"modelo": c, "metrica": m, "media": v.mean(), "std": v.std(ddof=1),
                            "ic95_inf": ic[0], "ic95_sup": ic[1]})
    resumen = pd.DataFrame(resumen)

    resultado = {"esquema": f"RepeatedKFold k={k} x {repeticiones} repeticiones "
                             f"({k * repeticiones} pliegues pareados)"}
    if len(claves) >= 3:
        f, p = stats.friedmanchisquare(*[por_fold[c]["rmse"] for c in claves])
        resultado["friedman_rmse"] = {"estadistico": float(f), "p_valor": float(p)}

    pares, p_t, p_w = [], {}, {}
    for a, b in combinations(claves, 2):
        da, db = por_fold[a]["rmse"], por_fold[b]["rmse"]
        t, pt = t_corregida_nadeau_bengio(da, db, n_train, n_test)
        w = stats.wilcoxon(da, db)
        nombre = f"{a} vs {b}"
        p_t[nombre], p_w[nombre] = pt, float(w.pvalue)
        pares.append({"par": nombre, "dif_media_rmse": float(np.mean(da - db)),
                      "t_nadeau_bengio": t, "p_t": pt,
                      "wilcoxon_W": float(w.statistic), "p_wilcoxon": float(w.pvalue),
                      "pliegues_gana_" + a: int((da < db).sum())})
    ht, hw = holm(p_t), holm(p_w)
    for fila in pares:
        fila["p_t_holm"] = ht[fila["par"]]
        fila["p_wilcoxon_holm"] = hw[fila["par"]]
        fila["significativo_5pct"] = bool(fila["p_t_holm"] < 0.05)
    pares = pd.DataFrame(pares)
    resultado["pares"] = pares.to_dict(orient="records")

    folds = pd.DataFrame({f"{c}_{m}": por_fold[c][m] for c in claves for m in ["rmse", "mae", "r2"]})
    folds.index.name = "pliegue"
    folds.to_csv(carpeta / "cv_repetido_por_pliegue.csv")
    resumen.to_csv(carpeta / "cv_repetido_resumen.csv", index=False)
    pares.to_csv(carpeta / "pruebas_pareadas.csv", index=False)
    (carpeta / "pruebas_estadisticas.json").write_text(
        json.dumps(resultado, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Gráfica: distribución del RMSE por pliegue
    fig, ax = plt.subplots(figsize=(7, 4))
    datos = [por_fold[c]["rmse"] for c in claves]
    bp = ax.boxplot(datos, patch_artist=True, widths=0.5, medianprops=dict(color=TINTA, lw=1.4),
                    flierprops=dict(marker="o", markersize=3, markeredgecolor="none"))
    for p, c in zip(bp["boxes"], claves):
        p.set_facecolor(COLORES[c]); p.set_alpha(0.35); p.set_edgecolor(COLORES[c])
    rng = np.random.default_rng(semilla)
    for i, (v, c) in enumerate(zip(datos, claves), start=1):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=12, color=COLORES[c], zorder=3)
    ax.set_xticks(range(1, len(claves) + 1), [NOMBRES[c] for c in claves])
    ax.set_ylabel("RMSE por pliegue (USD)")
    ax.set_title(f"RMSE en {k}×{repeticiones} K-Fold repetido (mismos pliegues)")
    guardar(fig, carpeta / "comparacion_rmse_pliegues.png")
    return resultado, resumen, pares


def ablacion_variables_derivadas(X, y, semilla, k) -> pd.DataFrame:
    """Mide el aporte de las variables derivadas (obese, smoker_obese) a un
    modelo lineal simple y a un Random Forest, con los mismos pliegues."""
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    kf = KFold(n_splits=k, shuffle=True, random_state=semilla)
    filas = []
    for con_derivadas in (False, True):
        cols = list(X.columns) if con_derivadas else [c for c in X.columns if c not in ("obese", "smoker_obese")]
        num = [c for c in dp.COLUMNAS_NUMERICAS]
        binarias = [c for c in ("obese", "smoker_obese") if c in cols]
        ct = ColumnTransformer([("num", StandardScaler(), num),
                                ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"),
                                 dp.COLUMNAS_CATEGORICAS)] + ([("bin", "passthrough", binarias)] if binarias else []))
        for nombre, mod in [("regresion_lineal", LinearRegression()),
                            ("random_forest", RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                                    random_state=semilla, n_jobs=-1))]:
            cv = cross_validate(Pipeline([("prep", ct), ("modelo", mod)]), X[cols], y, cv=kf,
                                scoring=SCORING, n_jobs=-1)
            filas.append({"modelo": nombre, "variables_derivadas": con_derivadas,
                          "cv_rmse_media": -cv["test_rmse"].mean(), "cv_rmse_std": cv["test_rmse"].std(ddof=1),
                          "cv_r2_media": cv["test_r2"].mean()})
    return pd.DataFrame(filas)


def grafica_metricas_prueba(tabla: pd.DataFrame, ruta: Path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, m, et in zip(axes, ["rmse", "mae", "r2"], ["RMSE (USD)", "MAE (USD)", "R²"]):
        vals = tabla[f"test_{m}"]
        cols = [COLORES[c] for c in tabla.index]
        ax.bar([NOMBRES[c] for c in tabla.index], vals, color=cols, width=0.6)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:,.3f}" if m == "r2" else f"{v:,.0f}", ha="center", va="bottom",
                    color=TINTA, fontsize=9)
        ax.set_title(et)
        ax.tick_params(axis="x", labelsize=8)
        if m == "r2":
            ax.set_ylim(0, 1)
    fig.suptitle("Métricas en el conjunto de prueba (20 %)", x=0.01, ha="left",
                 fontweight="bold", color=TINTA)
    guardar(fig, ruta)


# --------------------------------------------------------------------------
# Flujo principal
# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", default=str(RAIZ / "datos/datos_limp/insurance_limpio.csv"))
    p.add_argument("--manifest", default=str(RAIZ / "datos/MANIFEST.json"))
    p.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI",
                                                            f"sqlite:///{RAIZ / 'mlflow.db'}"))
    p.add_argument("--experimento", default="seguros-costos-regresion")
    p.add_argument("--modelos", nargs="+", default=["ridge", "random_forest", "xgboost"])
    p.add_argument("--semilla", type=int, default=42)
    p.add_argument("--k", type=int, default=5, help="pliegues de la validación cruzada")
    p.add_argument("--repeticiones", type=int, default=5, help="repeticiones del K-Fold comparativo")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--modelo-registrado", default="seguros-costos-regresor")
    p.add_argument("--salida", default=str(RAIZ / "resultados"))
    a = p.parse_args()

    np.random.seed(a.semilla)
    estilo()
    salida = Path(a.salida); salida.mkdir(parents=True, exist_ok=True)
    figs = RAIZ / "evidencias" / "modelos"; figs.mkdir(parents=True, exist_ok=True)

    # ---------------- Datos ----------------
    df = pd.read_csv(a.datos)
    X, y = dp.separar_xy(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=a.test_size, random_state=a.semilla, stratify=X["smoker"])
    manifest = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    version_datos = manifest["datos_limp"]["version"]
    sha_datos = manifest["datos_limp"]["sha256"]

    mlflow.set_tracking_uri(a.tracking_uri)
    if mlflow.get_experiment_by_name(a.experimento) is None:
        mlflow.create_experiment(a.experimento, tags={
            "proyecto": "Actividad 5 - Gestión de proyectos de IA",
            "problema": "regresion", "objetivo": "charges"})
    mlflow.set_experiment(a.experimento)

    tags_comunes = {
        "git_commit": git_commit(),
        "datos_version": version_datos,
        "datos_sha256": sha_datos,
        "semilla": str(a.semilla),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "mlflow": mlflow.__version__,
    }
    ds_train = mlflow.data.from_pandas(pd.concat([X_tr, y_tr], axis=1), source=a.datos,
                                       targets="charges", name=f"insurance_limpio_v{version_datos}_train")
    ds_test = mlflow.data.from_pandas(pd.concat([X_te, y_te], axis=1), source=a.datos,
                                      targets="charges", name=f"insurance_limpio_v{version_datos}_test")

    kf = KFold(n_splits=a.k, shuffle=True, random_state=a.semilla)
    catalogo = definir_modelos(a.semilla)
    mejores, filas, run_ids = {}, [], {}

    for clave in a.modelos:
        estimador, grid = catalogo[clave]
        print(f"\n=== {NOMBRES[clave]} ===")
        with mlflow.start_run(run_name=f"{clave}-gridsearch") as padre:
            run_ids[clave] = padre.info.run_id
            mlflow.set_tags({**tags_comunes, "modelo": clave, "tipo_corrida": "gridsearch"})
            mlflow.log_input(ds_train, context="training")
            mlflow.log_input(ds_test, context="testing")
            mlflow.log_params({"cv_k": a.k, "test_size": a.test_size,
                               "n_train": len(X_tr), "n_test": len(X_te),
                               "metrica_refit": "rmse",
                               "n_combinaciones_grid": int(np.prod([len(v) for v in grid.values()]))})
            mlflow.log_dict({k: [str(legible(x)) for x in v] for k, v in grid.items()},
                            "espacio_busqueda.json")

            gs = GridSearchCV(estimador, grid, scoring=SCORING, refit="rmse", cv=kf,
                              n_jobs=-1, return_train_score=True)
            gs.fit(X_tr, y_tr)
            cvr = pd.DataFrame(gs.cv_results_)

            # Corridas hijas: una por combinación evaluada
            for i in range(len(cvr)):
                params = {k.split("__")[-1] if k != "transformer" else "transformacion_objetivo":
                          legible(v) for k, v in cvr.loc[i, "params"].items()}
                with mlflow.start_run(run_name=f"{clave}-cand-{i:02d}", nested=True):
                    mlflow.set_tags({"modelo": clave, "tipo_corrida": "candidato",
                                     "rank_rmse": int(cvr.loc[i, "rank_test_rmse"])})
                    mlflow.log_params(params)
                    mlflow.log_metrics({
                        "cv_rmse_media": -cvr.loc[i, "mean_test_rmse"],
                        "cv_rmse_std": cvr.loc[i, "std_test_rmse"],
                        "cv_mae_media": -cvr.loc[i, "mean_test_mae"],
                        "cv_r2_media": cvr.loc[i, "mean_test_r2"],
                        "train_rmse_media": -cvr.loc[i, "mean_train_rmse"],
                        "tiempo_ajuste_s": cvr.loc[i, "mean_fit_time"],
                    })

            mejor = gs.best_estimator_
            mejores[clave] = mejor
            i_best = gs.best_index_
            mejores_params = {("transformacion_objetivo" if k == "transformer" else "best_" + k.split("__")[-1]):
                              legible(v) for k, v in gs.best_params_.items()}
            mlflow.log_params(mejores_params)

            yp_te = mejor.predict(X_te)
            yp_tr = mejor.predict(X_tr)
            m_te, m_tr = metricas(y_te.values, yp_te), metricas(y_tr.values, yp_tr)
            cv_m = {
                "cv_rmse_media": -cvr.loc[i_best, "mean_test_rmse"],
                "cv_rmse_std": cvr.loc[i_best, "std_test_rmse"],
                "cv_mae_media": -cvr.loc[i_best, "mean_test_mae"],
                "cv_mae_std": cvr.loc[i_best, "std_test_mae"],
                "cv_r2_media": cvr.loc[i_best, "mean_test_r2"],
                "cv_r2_std": cvr.loc[i_best, "std_test_r2"],
            }
            mlflow.log_metrics({**cv_m,
                                **{f"test_{k}": v for k, v in m_te.items()},
                                **{f"train_{k}": v for k, v in m_tr.items()}})

            # Artefactos
            cv_csv = salida / f"cv_results_{clave}.csv"
            cvr.assign(params=cvr["params"].map(lambda d: {k: str(legible(v)) for k, v in d.items()})) \
               .drop(columns=[c for c in cvr.columns if c.startswith("param_")]) \
               .sort_values("rank_test_rmse").to_csv(cv_csv, index=False)
            f1, f2, f3 = (figs / f"{clave}_pred_vs_real.png", figs / f"{clave}_residuos.png",
                          figs / f"{clave}_importancia.png")
            grafica_pred_vs_real(y_te.values, yp_te, clave, f1)
            grafica_residuos(y_te.values, yp_te, clave, f2)
            pi = permutation_importance(mejor, X_te, y_te, scoring="neg_root_mean_squared_error",
                                        n_repeats=10, random_state=a.semilla, n_jobs=-1)
            imp = pd.DataFrame({"variable": X_te.columns, "importancia_media": pi.importances_mean,
                                "importancia_std": pi.importances_std})
            imp.to_csv(salida / f"importancia_{clave}.csv", index=False)
            grafica_importancia(imp, clave, f3)
            for f in (cv_csv, f1, f2, f3, salida / f"importancia_{clave}.csv"):
                mlflow.log_artifact(str(f), artifact_path="evaluacion")
            log_modelo(mejor, X_tr, yp_tr, "modelo")

            filas.append({"modelo": clave, "run_id": padre.info.run_id,
                          **{f"best_{k}": v for k, v in mejores_params.items()},
                          **cv_m, **{f"test_{k}": v for k, v in m_te.items()},
                          **{f"train_{k}": v for k, v in m_tr.items()}})
            print(f"  mejores params: {mejores_params}")
            print(f"  CV RMSE {cv_m['cv_rmse_media']:,.1f} ± {cv_m['cv_rmse_std']:,.1f} | "
                  f"test RMSE {m_te['rmse']:,.1f}  MAE {m_te['mae']:,.1f}  R² {m_te['r2']:.4f}")

    tabla = pd.DataFrame(filas).set_index("modelo")
    tabla.to_csv(salida / "metricas_modelos.csv")
    grafica_metricas_prueba(tabla, figs / "comparacion_metricas_prueba.png")

    # ---------------- Comparación estadística ----------------
    with mlflow.start_run(run_name="comparacion-estadistica") as comp:
        mlflow.set_tags({**tags_comunes, "tipo_corrida": "comparacion",
                         "modelos": ",".join(a.modelos)})
        mlflow.log_params({"esquema_cv": "RepeatedKFold", "k": a.k, "repeticiones": a.repeticiones,
                           "prueba_global": "Friedman", "prueba_pareada": "t Nadeau-Bengio + Wilcoxon",
                           "correccion": "Holm"})
        res, resumen, pares = comparar_estadisticamente(mejores, X_tr, y_tr, a.semilla, a.k,
                                                        a.repeticiones, salida)
        fig_pliegues = figs / "comparacion_rmse_pliegues.png"
        (salida / "comparacion_rmse_pliegues.png").replace(fig_pliegues)
        for c in a.modelos:
            fila = resumen[(resumen.modelo == c) & (resumen.metrica == "rmse")].iloc[0]
            mlflow.log_metrics({f"{c}_rcv_rmse_media": fila.media, f"{c}_rcv_rmse_std": fila["std"]})
        if "friedman_rmse" in res:
            mlflow.log_metric("friedman_p_valor", res["friedman_rmse"]["p_valor"])
        for f in ["cv_repetido_por_pliegue.csv", "cv_repetido_resumen.csv", "pruebas_pareadas.csv",
                  "pruebas_estadisticas.json", "metricas_modelos.csv"]:
            mlflow.log_artifact(str(salida / f), artifact_path="comparacion")
        mlflow.log_artifact(str(fig_pliegues), artifact_path="comparacion")
        mlflow.log_artifact(str(figs / "comparacion_metricas_prueba.png"), artifact_path="comparacion")

        # Selección del campeón: menor RMSE medio en K-Fold repetido
        rmse_rcv = resumen[resumen.metrica == "rmse"].set_index("modelo")["media"]
        campeon = rmse_rcv.idxmin()
        mlflow.set_tag("campeon", campeon)
        comp_id = comp.info.run_id

    # ---------------- Ablación de ingeniería de variables ----------------
    with mlflow.start_run(run_name="ablacion-variables-derivadas"):
        mlflow.set_tags({**tags_comunes, "tipo_corrida": "ablacion"})
        mlflow.log_params({"variables_evaluadas": "obese,smoker_obese", "cv_k": a.k})
        abl = ablacion_variables_derivadas(X_tr, y_tr, a.semilla, a.k)
        abl.to_csv(salida / "ablacion_variables.csv", index=False)
        for _, r in abl.iterrows():
            suf = "con" if r.variables_derivadas else "sin"
            mlflow.log_metric(f"{r.modelo}_{suf}_derivadas_cv_rmse", r.cv_rmse_media)
        mlflow.log_artifact(str(salida / "ablacion_variables.csv"))
        print("\nAblación de variables derivadas:")
        print(abl.round(3))

    # ---------------- Model Registry ----------------
    client = MlflowClient()
    mv = mlflow.register_model(f"runs:/{run_ids[campeon]}/modelo", a.modelo_registrado,
                               tags={"datos_version": version_datos, "seleccion": "min RMSE K-Fold repetido"})
    client.update_registered_model(a.modelo_registrado,
                                   description="Regresor del costo anual del seguro médico (charges). "
                                               "Entrenado con fuentes/train.py; ver corrida comparacion-estadistica.")
    client.update_model_version(a.modelo_registrado, mv.version,
                                description=f"{NOMBRES[campeon]} | test RMSE {tabla.loc[campeon, 'test_rmse']:,.1f} "
                                            f"| test R² {tabla.loc[campeon, 'test_r2']:.4f}")
    client.set_registered_model_alias(a.modelo_registrado, "champion", mv.version)

    resumen_final = {
        "experimento": a.experimento,
        "tracking_uri": a.tracking_uri,
        "run_ids": run_ids,
        "run_comparacion": comp_id,
        "campeon": campeon,
        "modelo_registrado": a.modelo_registrado,
        "version_registrada": mv.version,
        "datos_version": version_datos,
        "datos_sha256": sha_datos,
        "git_commit": tags_comunes["git_commit"],
        "entorno": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                    "mlflow": mlflow.__version__, "xgboost": __import__("xgboost").__version__,
                    "pandas": pd.__version__, "numpy": np.__version__},
        "pruebas": res,
    }
    (salida / "resumen_entrenamiento.json").write_text(
        json.dumps(resumen_final, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    print("\n=== Resumen ===")
    print(tabla[["cv_rmse_media", "cv_rmse_std", "test_rmse", "test_mae", "test_r2"]].round(3))
    print("\nPruebas pareadas (RMSE):")
    print(pares[["par", "dif_media_rmse", "p_t_holm", "p_wilcoxon_holm", "significativo_5pct"]])
    print(f"\nCampeón: {campeon} -> registrado como {a.modelo_registrado} v{mv.version} (alias champion)")


if __name__ == "__main__":
    main()
