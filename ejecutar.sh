#!/usr/bin/env bash
# Reproduce el proyecto completo desde cero.
#   bash ejecutar.sh          -> entorno + limpieza + EDA + entrenamiento (vía cuaderno)
#   bash ejecutar.sh --ui     -> además abre la interfaz de MLflow en http://127.0.0.1:5000
#   bash ejecutar.sh --limpio -> borra corridas previas (mlflow.db, mlruns, resultados) antes de entrenar
set -euo pipefail
cd "$(dirname "$0")"

LIMPIO=0; UI=0
for arg in "$@"; do
  case "$arg" in
    --limpio) LIMPIO=1 ;;
    --ui) UI=1 ;;
  esac
done

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo ">> Creando entorno virtual .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
python -m pip freeze > requirements-lock.txt

if ! python -c "import xgboost" 2>/dev/null; then
  echo "!! XGBoost no pudo cargarse. En macOS suele faltar OpenMP: ejecuta 'brew install libomp' y vuelve a correr este script."
  exit 1
fi

echo ">> Pruebas unitarias de limpieza"
python -m pytest -q tests

echo ">> Limpieza y versionamiento de datos"
python fuentes/datos_prep.py

echo ">> Análisis exploratorio"
python fuentes/eda.py > /dev/null

if [ "$LIMPIO" = 1 ]; then
  echo ">> Eliminando corridas previas"
  rm -rf mlflow.db mlruns resultados evidencias/modelos
fi

echo ">> Entrenamiento y registro en MLflow (ejecutando el cuaderno)"
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=3600 fuentes/entrena.ipynb

echo ">> Listo. Resultados en resultados/ y evidencias/"
if [ "$UI" = 1 ]; then
  echo ">> MLflow UI en http://127.0.0.1:5000  (Ctrl+C para salir)"
  mlflow ui --backend-store-uri "sqlite:///$(pwd)/mlflow.db" --host 127.0.0.1 --port 5000
fi
