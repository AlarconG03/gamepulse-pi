"""Mejoras 1 y 2 del componente de ML, calculadas sin reentrenar y sin tocar `resultados/referencia/`.

1. Intervalos de confianza por *bootstrap agrupado por personaje* para las métricas de prueba y para las
   diferencias entre modelos. Se remuestrean personajes completos (con todas sus filas), porque un mismo
   personaje aparece en varios cortes y sus filas no son independientes.
2. Lift (métrica dividida por la tasa base), porque la PR-AUC depende de la tasa de churn de cada bloque
   (13 % en validación, 20 % en prueba) y por eso no es comparable entre bloques.

Además guarda la calibración por corte (riesgo medio estimado frente al churn observado), que sustenta la
mejora pendiente de recalibración.

Uso:
    python analisis_adicional/ic_y_lift.py [carpeta_referencia] [dataset.parquet] [carpeta_salida] [B]

Entradas: scores_prueba.parquet, resultados_modelos.csv y ficha_corrida.json (referencia) y el dataset de
modelado (runs/cache/dataset_*.parquet), solo para recalcular la línea base de recencia.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

RAIZ = Path(__file__).resolve().parents[1]
REF = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "resultados/referencia"
DATASET = Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "runs/cache/dataset_H30_e7_m2_c20.parquet"
SALIDA = Path(sys.argv[3]) if len(sys.argv) > 3 else RAIZ / "resultados/adicional"
B = int(sys.argv[4]) if len(sys.argv) > 4 else 1000
K_FRAC, SEED = 0.10, 42
MODELOS = ["Línea base (recencia)", "Regresión logística", "Random Forest", "LightGBM", "Ensamble"]
NOMBRE_REF = {"Línea base (recencia)": "Heurística recencia (línea base)", "Regresión logística": "Regresión logística",
              "Random Forest": "Random Forest", "LightGBM": "LightGBM", "Ensamble": "Ensamble (promedio de rangos)"}


def cargar() -> pd.DataFrame:
    s = pd.read_parquet(REF / "scores_prueba.parquet")
    d = pd.read_parquet(DATASET, columns=["char_id", "fecha_corte", "dias_desde_ultima", "frac_dias_activos_28"])
    s = s.merge(d, on=["char_id", "fecha_corte"], how="left", validate="one_to_one")
    assert s["dias_desde_ultima"].notna().all(), "faltan filas del dataset para algún (personaje, corte)"
    out = pd.DataFrame({"char_id": s.char_id, "corte": s.fecha_corte.rank(method="dense").astype(int) - 1,
                        "y": s.churn.values})
    out["Línea base (recencia)"] = ((s.dias_desde_ultima / 7 + (1 - s.frac_dias_activos_28)) / 2).values
    out["Regresión logística"] = s["riesgo_regresion-logistica"].values
    out["Random Forest"] = s["riesgo_random-forest"].values
    out["LightGBM"] = s["riesgo_lightgbm"].values
    out["Ensamble"] = np.mean([s[c].rank(pct=True).values for c in
                               ("riesgo_regresion-logistica", "riesgo_random-forest", "riesgo_lightgbm")], axis=0)
    return out


def metricas(y, cortes, S):
    """PR-AUC, ROC-AUC, precisión@K y recall@K (K = 10 % por corte) de cada modelo (columnas de S)."""
    res = {}
    ordenes = {}
    for j, m in enumerate(MODELOS):
        s = S[:, j]
        pa, ra = [], []
        for c in np.unique(cortes):
            ix = np.where(cortes == c)[0]
            k = max(int(round(len(ix) * K_FRAC)), 1)
            top = ix[np.argsort(-s[ix], kind="stable")[:k]]
            pa.append(y[top].mean())
            ra.append(y[top].sum() / max(y[ix].sum(), 1))
        res[m] = {"PR_AUC": average_precision_score(y, s), "ROC_AUC": roc_auc_score(y, s),
                  "precision@K": float(np.mean(pa)), "recall@K": float(np.mean(ra))}
        res[m]["lift_PR_AUC"] = res[m]["PR_AUC"] / y.mean()
        res[m]["lift@K"] = res[m]["precision@K"] / y.mean()
    return res


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    d = cargar()
    y, cortes, S = d["y"].values, d["corte"].values, d[MODELOS].values
    punto = metricas(y, cortes, S)
    ref = pd.read_csv(REF / "resultados_modelos.csv").set_index("modelo")
    for m in MODELOS:  # el recálculo debe coincidir con la corrida de referencia
        for k_ref, k in (("test_PR_AUC", "PR_AUC"), ("test_precision@K", "precision@K"), ("test_recall@K", "recall@K")):
            assert abs(ref.loc[NOMBRE_REF[m], k_ref] - punto[m][k]) < 2e-3, (m, k, ref.loc[NOMBRE_REF[m], k_ref], punto[m][k])
    print("Métricas puntuales coinciden con resultados_modelos.csv")

    # --- bootstrap agrupado por personaje
    rng = np.random.default_rng(SEED)
    filas_por_char = d.groupby("char_id").indices
    chars = np.array(list(filas_por_char.keys()))
    lista = [filas_por_char[c] for c in chars]
    boot = []
    for b in range(B):
        sel = rng.integers(0, len(chars), len(chars))
        ix = np.concatenate([lista[i] for i in sel])
        boot.append(metricas(y[ix], cortes[ix], S[ix]))
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{B}", flush=True)

    filas = []
    for m in MODELOS:
        for k in ("PR_AUC", "ROC_AUC", "precision@K", "recall@K", "lift_PR_AUC", "lift@K"):
            v = np.array([bb[m][k] for bb in boot])
            filas.append({"modelo": m, "metrica": k, "valor": round(punto[m][k], 4),
                          "ic95_inf": round(float(np.percentile(v, 2.5)), 4),
                          "ic95_sup": round(float(np.percentile(v, 97.5)), 4), "B": B})
    pd.DataFrame(filas).to_csv(SALIDA / "ic_bootstrap.csv", index=False)

    pares = [("LightGBM", "Línea base (recencia)"), ("LightGBM", "Regresión logística"), ("LightGBM", "Random Forest"),
             ("LightGBM", "Ensamble"), ("Random Forest", "Regresión logística")]
    dif = []
    for a, b2 in pares:
        for k in ("PR_AUC", "precision@K", "recall@K"):
            v = np.array([bb[a][k] - bb[b2][k] for bb in boot])
            dif.append({"modelo_a": a, "modelo_b": b2, "metrica": k, "diferencia": round(punto[a][k] - punto[b2][k], 4),
                        "ic95_inf": round(float(np.percentile(v, 2.5)), 4), "ic95_sup": round(float(np.percentile(v, 97.5)), 4),
                        "ic_excluye_cero": bool(np.percentile(v, 2.5) > 0 or np.percentile(v, 97.5) < 0)})
    pd.DataFrame(dif).to_csv(SALIDA / "ic_diferencias.csv", index=False)

    # --- lift en validación y en prueba
    ficha = json.loads((REF / "ficha_corrida.json").read_text())
    base_val, base_test = ficha["metricas_validacion"]["tasa_base"], ficha["metricas_prueba"]["tasa_base"]
    lift = []
    for m in MODELOS:
        r = ref.loc[NOMBRE_REF[m]]
        lift.append({"modelo": m, "tasa_base_validacion": base_val, "PR_AUC_validacion": r["val_PR_AUC"],
                     "lift_validacion": round(r["val_PR_AUC"] / base_val, 2), "tasa_base_prueba": base_test,
                     "PR_AUC_prueba": r["test_PR_AUC"], "lift_prueba": round(r["test_PR_AUC"] / base_test, 2),
                     "lift@K_prueba": round(r["test_precision@K"] / base_test, 2)})
    pd.DataFrame(lift).to_csv(SALIDA / "lift.csv", index=False)

    # --- calibración por corte (apoya la mejora pendiente de recalibración)
    s = pd.read_parquet(REF / "scores_prueba.parquet")
    cal = s.groupby("fecha_corte").agg(n=("churn", "size"), churn_observado=("churn", "mean"),
                                       riesgo_medio_lightgbm=("riesgo_lightgbm", "mean")).reset_index()
    cal["fecha_corte"] = cal["fecha_corte"].dt.date
    tot = {"fecha_corte": "Prueba completa", "n": len(s), "churn_observado": s.churn.mean(),
           "riesgo_medio_lightgbm": s.riesgo_lightgbm.mean()}
    cal = pd.concat([cal, pd.DataFrame([tot])], ignore_index=True).round(4)
    cal.to_csv(SALIDA / "calibracion_por_corte.csv", index=False)
    print(pd.DataFrame(filas).to_string(), "\n", pd.DataFrame(dif).to_string(), "\n", pd.DataFrame(lift).to_string(),
          "\n", cal.to_string())


if __name__ == "__main__":
    main()
