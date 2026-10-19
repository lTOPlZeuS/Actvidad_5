"""
eda.py — Visualizaciones de calidad de datos, patrones y anomalías.

Uso:  python fuentes/eda.py  (genera PNG en evidencias/eda/)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, REJILLA = "#0b0b0b", "#52514e", "#e4e3df"
COLOR_FUMADOR = {"no": AZUL, "yes": NARANJA}
ETIQUETA_FUMADOR = {"no": "No fumador", "yes": "Fumador"}


def estilo():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
        "axes.edgecolor": REJILLA, "axes.labelcolor": TINTA2, "axes.titlecolor": TINTA,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
        "xtick.color": TINTA2, "ytick.color": TINTA2, "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })


def guardar(fig, carpeta: Path, nombre: str):
    fig.tight_layout()
    fig.savefig(carpeta / nombre, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def fig_calidad(ini: pd.DataFrame, limp: pd.DataFrame, carpeta: Path):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    cols = list(ini.columns)
    nulos = ini.isna().sum().reindex(cols).values
    ax.bar(cols, nulos, color=AZUL, width=0.6)
    dup = int(ini.duplicated().sum())
    ax.set_ylim(0, max(5, nulos.max() + 1))
    ax.set_ylabel("Valores nulos")
    ax.set_title("Calidad de datos originales: valores nulos por columna")
    ax.text(0.99, 0.92, f"Filas originales: {len(ini)}\nDuplicados exactos: {dup}\n"
            f"Filas tras limpieza: {len(limp)}", transform=ax.transAxes, ha="right",
            va="top", color=TINTA2, fontsize=9)
    guardar(fig, carpeta, "01_calidad_datos.png")


def fig_distribucion_objetivo(df: pd.DataFrame, carpeta: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].hist(df["charges"], bins=40, color=AZUL, edgecolor="white", linewidth=0.6)
    axes[0].axvline(df["charges"].median(), color=TINTA2, ls="--", lw=1)
    axes[0].set_title("Distribución de charges (USD)")
    axes[0].set_xlabel("charges"); axes[0].set_ylabel("Frecuencia")
    axes[0].text(0.97, 0.9, f"asimetría = {df['charges'].skew():.2f}",
                 transform=axes[0].transAxes, ha="right", color=TINTA2)
    lg = np.log1p(df["charges"])
    axes[1].hist(lg, bins=40, color=AQUA, edgecolor="white", linewidth=0.6)
    axes[1].set_title("Distribución de log(1 + charges)")
    axes[1].set_xlabel("log(1 + charges)")
    axes[1].text(0.97, 0.9, f"asimetría = {lg.skew():.2f}",
                 transform=axes[1].transAxes, ha="right", color=TINTA2)
    guardar(fig, carpeta, "02_distribucion_objetivo.png")


def fig_dispersion(df: pd.DataFrame, x: str, carpeta: Path, nombre: str, titulo: str, xlabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for s in ["no", "yes"]:
        d = df[df.smoker == s]
        ax.scatter(d[x], d["charges"], s=14, alpha=0.65, color=COLOR_FUMADOR[s],
                   edgecolor="white", linewidth=0.4, label=ETIQUETA_FUMADOR[s])
    if x == "bmi":
        ax.axvline(30, color=TINTA2, ls="--", lw=1)
        ax.text(30.4, ax.get_ylim()[1] * 0.95, "IMC = 30 (obesidad OMS)", color=TINTA2, fontsize=9)
    ax.set_xlabel(xlabel); ax.set_ylabel("charges (USD)")
    ax.set_title(titulo)
    ax.legend(loc="upper left")
    guardar(fig, carpeta, nombre)


def fig_boxplots(df: pd.DataFrame, carpeta: Path):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, col in zip(axes, ["smoker", "sex", "region", "children"]):
        grupos = sorted(df[col].unique())
        datos = [df.loc[df[col] == g, "charges"] for g in grupos]
        bp = ax.boxplot(datos, patch_artist=True, widths=0.55,
                        medianprops=dict(color=TINTA, lw=1.4),
                        flierprops=dict(marker="o", markersize=3, markerfacecolor=NARANJA,
                                        markeredgecolor="none", alpha=0.6))
        for p in bp["boxes"]:
            p.set_facecolor("#cfe0f6"); p.set_edgecolor(AZUL)
        ax.set_xticks(range(1, len(grupos) + 1), [str(g) for g in grupos],
                      rotation=30 if col == "region" else 0)
        ax.set_title(col)
    axes[0].set_ylabel("charges (USD)")
    fig.suptitle("charges por variable categórica (puntos naranjas = atípicos 1.5·IQR)",
                 x=0.01, ha="left", fontweight="bold", color=TINTA)
    guardar(fig, carpeta, "05_boxplots_categoricas.png")


def fig_correlacion(df: pd.DataFrame, carpeta: Path):
    d = df.copy()
    d["smoker"] = (d.smoker == "yes").astype(int)
    d["sex"] = (d.sex == "male").astype(int)
    cols = ["age", "bmi", "children", "smoker", "sex", "obese", "smoker_obese", "charges"]
    c = d[cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
    ax.set_yticks(range(len(cols)), cols)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 0.6 else TINTA)
    fig.colorbar(im, ax=ax, shrink=0.8, label="ρ de Spearman")
    ax.set_title("Correlación de Spearman")
    guardar(fig, carpeta, "06_correlacion.png")
    return c


def fig_atipicos_bmi(df: pd.DataFrame, carpeta: Path, lim_sup: float):
    fig, ax = plt.subplots(figsize=(8, 2.8))
    kw = dict(widths=0.5, patch_artist=True,
               boxprops=dict(facecolor="#cfe0f6", edgecolor=AZUL),
               medianprops=dict(color=TINTA),
               flierprops=dict(marker="o", markersize=5, markerfacecolor=NARANJA, markeredgecolor="none"))
    try:
        ax.boxplot(df["bmi"], orientation="horizontal", **kw)   # matplotlib >= 3.10
    except TypeError:
        ax.boxplot(df["bmi"], vert=False, **kw)
    ax.axvline(lim_sup, color=TINTA2, ls="--", lw=1)
    ax.text(lim_sup + 0.3, 1.3, f"límite IQR = {lim_sup:.1f}", color=TINTA2, fontsize=9)
    ax.set_yticks([]); ax.set_xlabel("bmi")
    ax.set_title("Atípicos de IMC (se conservan: valores clínicamente plausibles)")
    guardar(fig, carpeta, "07_atipicos_bmi.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ini", default="datos/datos_ini/insurance.csv")
    p.add_argument("--limp", default="datos/datos_limp/insurance_limpio.csv")
    p.add_argument("--bitacora", default="datos/datos_limp/bitacora_limpieza.json")
    p.add_argument("--salida", default="evidencias/eda")
    a = p.parse_args()

    estilo()
    carpeta = Path(a.salida); carpeta.mkdir(parents=True, exist_ok=True)
    ini, limp = pd.read_csv(a.ini), pd.read_csv(a.limp)
    bit = json.loads(Path(a.bitacora).read_text(encoding="utf-8"))

    fig_calidad(ini, limp, carpeta)
    fig_distribucion_objetivo(limp, carpeta)
    fig_dispersion(limp, "age", carpeta, "03_charges_vs_edad.png",
                   "charges vs edad: tres bandas definidas por tabaquismo e IMC", "age (años)")
    fig_dispersion(limp, "bmi", carpeta, "04_charges_vs_imc.png",
                   "charges vs IMC: en fumadores el costo salta al cruzar IMC = 30", "bmi")
    fig_boxplots(limp, carpeta)
    corr = fig_correlacion(limp, carpeta)
    fig_atipicos_bmi(limp, carpeta, bit["atipicos_iqr"]["bmi"]["limite_superior"])

    # Resumen numérico para el reporte
    resumen = {
        "charges_media_por_fumador": limp.groupby("smoker")["charges"].mean().round(2).to_dict(),
        "charges_mediana_por_fumador": limp.groupby("smoker")["charges"].median().round(2).to_dict(),
        "charges_media_fumador_x_obeso": {
            f"smoker={k[0]},obese={k[1]}": v
            for k, v in limp.groupby(["smoker", "obese"])["charges"].mean().round(2).items()},
        "asimetria_charges": round(float(limp["charges"].skew()), 3),
        "asimetria_log_charges": round(float(np.log1p(limp["charges"]).skew()), 3),
        "spearman_con_charges": corr["charges"].drop("charges").round(3).to_dict(),
        "pct_fumadores": round(100 * float((limp.smoker == "yes").mean()), 2),
    }
    (carpeta / "resumen_eda.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False) + "\n",
                                              encoding="utf-8")
    print(json.dumps(resumen, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
