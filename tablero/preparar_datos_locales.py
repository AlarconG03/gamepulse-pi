"""Llena tablero/datos/ con los mismos archivos publicados en S3 (desarrollo, sin AWS).

Uso (desde la raíz del repositorio):  python tablero/preparar_datos_locales.py
En la demostración se usa, en cambio, la exportación de Athena (aws/infra/exportar_tablero.sh).
"""
import shutil
from pathlib import Path

import pandas as pd

R = Path(__file__).resolve().parents[1]
D = R / "tablero" / "datos"
PUB = R / "aws" / "publicar"
for t in ("scores", "perfiles", "shap_global"):
    (D / t).mkdir(parents=True, exist_ok=True)
    for f in (PUB / t).glob("*.parquet"):
        shutil.copy(f, D / t / f.name)
(D / "modelo").mkdir(parents=True, exist_ok=True)
for f in ("modelo.txt", "meta.json"):
    shutil.copy(PUB / "modelos" / "v1" / f, D / "modelo" / f)
s = pd.read_parquet(D / "scores")
oro = pd.read_parquet(R / "runs" / "cache" / "dataset_H30_e7_m2_c20.parquet")
casos = oro[oro["fecha_corte"] >= s["fecha_corte"].min()].drop(columns=["fin_ventana_etiqueta"], errors="ignore")
(D / "casos").mkdir(exist_ok=True)
casos.to_parquet(D / "casos" / "casos.parquet", index=False)
print(f"listo: {len(s):,} scores, {len(casos):,} filas de variables en {D}")
