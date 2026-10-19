"""Pruebas unitarias de las reglas de limpieza (python -m pytest -q)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "fuentes"))
import datos_prep as dp  # noqa: E402


def base():
    return pd.DataFrame({
        "age": [19, 30, 45, 60], "sex": ["female", "male", "male", "female"],
        "bmi": [27.9, 31.0, 22.5, 35.2], "children": [0, 1, 2, 3],
        "smoker": ["yes", "no", "no", "yes"],
        "region": ["southwest", "northeast", "northwest", "southeast"],
        "charges": [16884.9, 4000.0, 8000.0, 45000.0],
    })


def test_estandariza_texto():
    df = base(); df.loc[0, "sex"] = "  Female "; df.loc[1, "smoker"] = "NO"
    limpio, _ = dp.limpiar(df)
    assert set(limpio["sex"]) <= dp.DOMINIOS["sex"]
    assert set(limpio["smoker"]) <= dp.DOMINIOS["smoker"]


def test_imputa_nulos_y_elimina_sin_objetivo():
    df = base(); df.loc[0, "bmi"] = np.nan; df.loc[1, "charges"] = np.nan
    limpio, bit = dp.limpiar(df)
    assert limpio.isna().sum().sum() == 0
    assert bit["nulos"]["filas_sin_objetivo_eliminadas"] == 1
    assert bit["nulos"]["imputaciones"]["bmi"]["estrategia"] == "mediana"


def test_elimina_duplicados():
    df = pd.concat([base(), base().iloc[[0]]])
    limpio, bit = dp.limpiar(df)
    assert bit["duplicados"]["duplicados_eliminados"] == 1
    assert len(limpio) == 4


def test_reglas_de_negocio():
    df = base(); df.loc[0, "age"] = 130; df.loc[1, "region"] = "centro"; df.loc[2, "charges"] = -5
    limpio, bit = dp.limpiar(df)
    assert bit["reglas_negocio"]["filas_eliminadas"] == 3
    assert len(limpio) == 1


def test_variables_derivadas():
    limpio, _ = dp.limpiar(base())
    assert limpio.loc[limpio.bmi >= 30, "obese"].eq(1).all()
    esperado = ((limpio.smoker == "yes") & (limpio.bmi >= 30)).astype(int)
    assert limpio["smoker_obese"].equals(esperado)


def test_esquema_incompleto_falla():
    try:
        dp.limpiar(base().drop(columns=["bmi"]))
    except ValueError:
        return
    raise AssertionError("debió fallar por columna faltante")
