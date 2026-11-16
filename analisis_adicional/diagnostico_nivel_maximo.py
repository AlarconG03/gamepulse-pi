"""Diagnóstico de la subpoblación en nivel máximo (3.3.4, 3.3.6, 4.4 y 4.5 del documento), sin reentrenar.

Lee los scores de la prueba de la corrida de referencia y la tabla de modelado en caché, y separa dos efectos:
(1) cuánto del bajo desempeño "dentro del grupo" es un artefacto del promedio por corte, y
(2) qué cambia con la expansión del 13/11/2008 (tope de nivel de 70 a 80).

Uso (desde la raíz del repositorio):  python analisis_adicional/diagnostico_nivel_maximo.py
Salida: resultados/adicional/nivel_maximo.csv y el resumen en pantalla.
"""
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from gamepulse_ml.modelos import precision_recall_at_k

RAIZ = Path(__file__).resolve().parents[1]
K = 0.10
EXPANSION = pd.Timestamp("2008-11-13")

scores = pd.read_parquet(RAIZ / "resultados/referencia/scores_prueba.parquet")
tabla = pd.read_parquet(RAIZ / "runs/cache/dataset_H30_e7_m2_c20.parquet",
                        columns=["char_id", "fecha_corte", "nivel", "en_tope"])
for d in (scores, tabla):
    d["fecha_corte"] = pd.to_datetime(d["fecha_corte"]).dt.normalize()
df = scores.merge(tabla, on=["char_id", "fecha_corte"], how="left", validate="1:1")
assert df["en_tope"].notna().all(), "faltan filas de la prueba en la tabla de modelado"
tope = df[df["en_tope"] == 1]

filas = []


def fila(nombre, valor, nota=""):
    filas.append({"metrica": nombre, "valor": valor, "nota": nota})
    print(f"{nombre:<70} {valor}" + (f"   ({nota})" if nota else ""))


fila("filas de prueba en nivel máximo", len(tope))
fila("% de la prueba en nivel máximo", round(len(tope) / len(df), 3))
fila("% del 10 % priorizado (lista global) en nivel máximo",
     round(df.loc[df["prioridad_top_k"].astype(bool), "en_tope"].mean(), 3))
fila("tasa de y = 1 en nivel máximo", round(tope["churn"].mean(), 3))
fila("ROC-AUC dentro del grupo", round(roc_auc_score(tope["churn"], tope["riesgo"]), 3))
p, r = precision_recall_at_k(tope, tope["riesgo"].values, K)
fila("precisión@10 % dentro del grupo (promedio por corte, como el pipeline)", round(p, 3))
fila("recall@10 % dentro del grupo (promedio por corte)", round(r, 3))

print("\nPor corte (grupo en nivel máximo):")
por_corte = tope.groupby("fecha_corte").agg(filas=("churn", "size"), positivos=("churn", "sum"),
                                            tasa=("churn", "mean"))
print(por_corte.to_string())
sin_pos = por_corte[por_corte["positivos"] == 0]
for c, r_ in sin_pos.iterrows():
    fila(f"corte sin ningún y = 1 en el grupo: {c:%d/%m/%Y}", int(r_["filas"]), "filas en el grupo")

con_pos = tope[~tope["fecha_corte"].isin(sin_pos.index)]
p2, r2 = precision_recall_at_k(con_pos, con_pos["riesgo"].values, K)
fila("precisión@10 % dentro del grupo, sin los cortes sin positivos", round(p2, 3))
fila("recall@10 % dentro del grupo, sin los cortes sin positivos", round(r2, 3))

n70 = df[df["nivel"] == 70]
antes, despues = n70[n70["fecha_corte"] < EXPANSION], n70[n70["fecha_corte"] >= EXPANSION]
fila("nivel 70: tasa de y = 1 antes de la expansión", round(antes["churn"].mean(), 3), f"{len(antes)} filas")
fila("nivel 70: tasa de y = 1 después de la expansión", round(despues["churn"].mean(), 3), f"{len(despues)} filas")
if despues["churn"].nunique() == 2:
    fila("nivel 70 después de la expansión: ROC-AUC", round(roc_auc_score(despues["churn"], despues["riesgo"]), 3))

salida = RAIZ / "resultados/adicional/nivel_maximo.csv"
pd.DataFrame(filas).to_csv(salida, index=False)
por_corte.to_csv(RAIZ / "resultados/adicional/nivel_maximo_por_corte.csv")
print(f"\nGuardado en {salida.relative_to(RAIZ)}")
