"""
datos_prep.py — Funciones de limpieza, estandarización y versionamiento del
dataset "Medical Cost Personal Datasets" (insurance.csv).

Uso desde la raíz del repositorio:

    python fuentes/datos_prep.py
        --entrada datos/datos_ini/insurance.csv
        --salida  datos/datos_limp/insurance_limpio.csv

El script:
  1. Valida el esquema esperado (columnas y tipos).
  2. Estandariza texto de variables categóricas (minúsculas, sin espacios).
  3. Maneja valores nulos (mediana en numéricas, moda en categóricas;
     se eliminan filas sin variable objetivo).
  4. Elimina registros duplicados exactos.
  5. Aplica reglas de negocio (rangos válidos y dominios permitidos).
  6. Detecta valores atípicos con el criterio IQR (solo se marcan, no se borran).
  7. Agrega variables derivadas con significado clínico (obesidad OMS).
  8. Guarda el dataset limpio y actualiza datos/MANIFEST.json con su huella
     SHA-256, de modo que cada versión de los datos quede trazable.

La codificación de categóricas y el escalado NO se aplican aquí, sino dentro
del Pipeline de scikit-learn (ver construir_preprocesador), para que se
ajusten solo con los datos de entrenamiento de cada pliegue y no haya fuga
de información (data leakage).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constantes del dominio
# --------------------------------------------------------------------------
VERSION_PIPELINE_DATOS = "1.1.0"

OBJETIVO = "charges"

ESQUEMA = {
    "age": "int64",
    "sex": "string",
    "bmi": "float64",
    "children": "int64",
    "smoker": "string",
    "region": "string",
    "charges": "float64",
}

COLUMNAS_CATEGORICAS = ["sex", "smoker", "region"]
COLUMNAS_NUMERICAS = ["age", "bmi", "children"]

DOMINIOS = {
    "sex": {"female", "male"},
    "smoker": {"yes", "no"},
    "region": {"northeast", "northwest", "southeast", "southwest"},
}

# Reglas de negocio: rangos plausibles para una póliza individual de adultos
REGLAS_RANGO = {
    "age": (18, 64),        # población asegurable del dataset (adultos < 65)
    "bmi": (10.0, 70.0),    # IMC fisiológicamente plausible
    "children": (0, 10),    # dependientes cubiertos
    "charges": (0.01, None),  # costo estrictamente positivo
}

UMBRAL_OBESIDAD_OMS = 30.0


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------
def sha256_archivo(ruta: str | Path) -> str:
    """Huella SHA-256 de un archivo (identifica de forma única cada versión)."""
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 16), b""):
            h.update(bloque)
    return h.hexdigest()


def cargar_datos(ruta: str | Path) -> pd.DataFrame:
    return pd.read_csv(ruta)


# --------------------------------------------------------------------------
# Pasos de limpieza (cada uno devuelve el DataFrame y un dict de bitácora)
# --------------------------------------------------------------------------
def validar_esquema(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    faltantes = set(ESQUEMA) - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(faltantes)}")
    extra = sorted(set(df.columns) - set(ESQUEMA))
    df = df[list(ESQUEMA)].copy()

    errores_conversion = {}
    for col, tipo in ESQUEMA.items():
        if tipo == "string":
            df[col] = df[col].astype("string")
        else:
            convertida = pd.to_numeric(df[col], errors="coerce")
            errores_conversion[col] = int(convertida.isna().sum() - df[col].isna().sum())
            df[col] = convertida  # enteros se castean tras imputar nulos
    return df, {"columnas_descartadas": extra, "valores_no_numericos": errores_conversion}


def estandarizar_texto(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    cambios = {}
    for col in COLUMNAS_CATEGORICAS:
        antes = df[col].copy()
        df[col] = df[col].str.strip().str.lower().str.replace(r"\s+", "", regex=True)
        cambios[col] = int((antes.fillna("<NA>") != df[col].fillna("<NA>")).sum())
    return df, {"valores_modificados": cambios}


def manejar_nulos(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    nulos_antes = {c: int(v) for c, v in df.isna().sum().items()}

    # Sin variable objetivo no hay aprendizaje supervisado: se elimina la fila
    sin_objetivo = int(df[OBJETIVO].isna().sum())
    df = df[df[OBJETIVO].notna()]

    imputaciones = {}
    for col in COLUMNAS_NUMERICAS:
        n = int(df[col].isna().sum())
        if n:
            valor = float(df[col].median())
            df[col] = df[col].fillna(valor)
            imputaciones[col] = {"n": n, "estrategia": "mediana", "valor": valor}
    for col in COLUMNAS_CATEGORICAS:
        n = int(df[col].isna().sum())
        if n:
            valor = str(df[col].mode().iloc[0])
            df[col] = df[col].fillna(valor)
            imputaciones[col] = {"n": n, "estrategia": "moda", "valor": valor}

    df["age"] = df["age"].round().astype("int64")
    df["children"] = df["children"].round().astype("int64")
    return df, {
        "nulos_por_columna": nulos_antes,
        "filas_sin_objetivo_eliminadas": sin_objetivo,
        "imputaciones": imputaciones,
    }


def eliminar_duplicados(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mask = df.duplicated(keep="first")
    indices = df.index[mask].tolist()
    return df[~mask].copy(), {"duplicados_eliminados": int(mask.sum()), "indices": indices}


def aplicar_reglas_negocio(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = df.copy()
    invalidas = pd.Series(False, index=df.index)
    detalle = {}
    for col, (lo, hi) in REGLAS_RANGO.items():
        m = pd.Series(False, index=df.index)
        if lo is not None:
            m |= df[col] < lo
        if hi is not None:
            m |= df[col] > hi
        detalle[f"{col}_fuera_de_rango"] = int(m.sum())
        invalidas |= m
    for col, dominio in DOMINIOS.items():
        m = ~df[col].isin(dominio)
        detalle[f"{col}_fuera_de_dominio"] = int(m.sum())
        invalidas |= m
    detalle["filas_eliminadas"] = int(invalidas.sum())
    return df[~invalidas].copy(), detalle


def detectar_atipicos_iqr(df: pd.DataFrame, columnas=("bmi", "charges")) -> dict:
    """Marca atípicos con el criterio de Tukey (1.5·IQR). No elimina filas:
    en este dominio los costos altos son reales (fumadores, obesidad)."""
    resumen = {}
    for col in columnas:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lim_inf, lim_sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        m = (df[col] < lim_inf) | (df[col] > lim_sup)
        resumen[col] = {
            "limite_inferior": round(float(lim_inf), 4),
            "limite_superior": round(float(lim_sup), 4),
            "n_atipicos": int(m.sum()),
            "pct_atipicos": round(100 * float(m.mean()), 2),
        }
        if col == OBJETIVO and m.any():
            resumen[col]["pct_fumadores_entre_atipicos"] = round(
                100 * float((df.loc[m, "smoker"] == "yes").mean()), 2
            )
    return resumen


def agregar_variables(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Variables derivadas con regla clínica explícita (OMS: IMC ≥ 30 = obesidad)."""
    df = df.copy()
    df["obese"] = (df["bmi"] >= UMBRAL_OBESIDAD_OMS).astype("int64")
    df["smoker_obese"] = ((df["smoker"] == "yes") & (df["obese"] == 1)).astype("int64")
    return df, {
        "variables_nuevas": ["obese", "smoker_obese"],
        "regla": f"obese = bmi >= {UMBRAL_OBESIDAD_OMS}; smoker_obese = smoker=='yes' AND obese",
    }


def limpiar(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Ejecuta el flujo completo y devuelve (df_limpio, bitácora)."""
    bitacora = {"filas_iniciales": int(len(df))}
    for nombre, paso in [
        ("esquema", validar_esquema),
        ("texto", estandarizar_texto),
        ("nulos", manejar_nulos),
        ("duplicados", eliminar_duplicados),
        ("reglas_negocio", aplicar_reglas_negocio),
    ]:
        df, info = paso(df)
        bitacora[nombre] = info
    bitacora["atipicos_iqr"] = detectar_atipicos_iqr(df)
    df, info = agregar_variables(df)
    bitacora["variables_derivadas"] = info
    df = df.reset_index(drop=True)
    bitacora["filas_finales"] = int(len(df))
    return df, bitacora


# --------------------------------------------------------------------------
# Preprocesamiento para modelado (se usa dentro de los Pipelines)
# --------------------------------------------------------------------------
def construir_preprocesador(escalar: bool = True):
    """ColumnTransformer: One-Hot para categóricas (drop='first' evita
    colinealidad perfecta en modelos lineales) y StandardScaler para numéricas."""
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numericas = COLUMNAS_NUMERICAS
    binarias = ["obese", "smoker_obese"]
    transformadores = [
        ("num", StandardScaler() if escalar else "passthrough", numericas),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore",
                              sparse_output=False), COLUMNAS_CATEGORICAS),
        ("bin", "passthrough", binarias),
    ]
    return ColumnTransformer(transformadores, verbose_feature_names_out=False)


def separar_xy(df: pd.DataFrame):
    X = df.drop(columns=[OBJETIVO])
    for c in COLUMNAS_CATEGORICAS:
        X[c] = X[c].astype(object)
    return X, df[OBJETIVO].astype(float)


# --------------------------------------------------------------------------
# Versionamiento
# --------------------------------------------------------------------------
def registrar_version(manifest: Path, clave: str, ruta: Path, descripcion: str,
                      extra: dict | None = None) -> dict:
    """Agrega/actualiza una entrada del manifiesto de datos."""
    datos = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    df = pd.read_csv(ruta)
    entrada = {
        "archivo": ruta.as_posix(),
        "sha256": sha256_archivo(ruta),
        "filas": int(df.shape[0]),
        "columnas": int(df.shape[1]),
        "descripcion": descripcion,
        "generado_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if extra:
        entrada.update(extra)
    datos[clave] = entrada
    manifest.write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return entrada


def _a_json(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    raise TypeError(type(obj))


def main():
    p = argparse.ArgumentParser(description="Limpieza y versionamiento del dataset")
    p.add_argument("--entrada", default="datos/datos_ini/insurance.csv")
    p.add_argument("--salida", default="datos/datos_limp/insurance_limpio.csv")
    p.add_argument("--manifest", default="datos/MANIFEST.json")
    args = p.parse_args()

    entrada, salida, manifest = Path(args.entrada), Path(args.salida), Path(args.manifest)
    df_ini = cargar_datos(entrada)
    df_limpio, bitacora = limpiar(df_ini)

    salida.parent.mkdir(parents=True, exist_ok=True)
    df_limpio.to_csv(salida, index=False)
    ruta_bitacora = salida.with_name("bitacora_limpieza.json")
    ruta_bitacora.write_text(json.dumps(bitacora, indent=2, ensure_ascii=False,
                                        default=_a_json) + "\n", encoding="utf-8")

    registrar_version(
        manifest, "datos_ini", entrada,
        "Datos originales sin modificar (Medical Cost Personal Datasets)",
        {"version": "1.0.0",
         "fuente": "https://github.com/stedy/Machine-Learning-with-R-datasets/blob/master/insurance.csv",
         "commit_fuente": "ff933238d9c30da179b7e3ad4b6ca938d67efc53",
         "origen": "Dataset público del libro 'Machine Learning with R' (B. Lantz); réplica en Kaggle: mirichoi0218/insurance"},
    )
    registrar_version(
        manifest, "datos_limp", salida,
        "Datos limpios: esquema validado, texto estandarizado, nulos tratados, "
        "duplicados eliminados, reglas de negocio y variables derivadas",
        {"version": VERSION_PIPELINE_DATOS, "derivado_de": "datos_ini",
         "script": "fuentes/datos_prep.py"},
    )

    print(f"Filas: {bitacora['filas_iniciales']} -> {bitacora['filas_finales']}")
    print(f"Duplicados eliminados: {bitacora['duplicados']['duplicados_eliminados']}")
    print(f"Guardado en {salida} | bitácora: {ruta_bitacora}")


if __name__ == "__main__":
    main()
