"""Análisis adicional del componente de ML: búsqueda de hiperparámetros, diagnóstico sesgo/varianza,
valor de negocio y casos aplicados. No modifica `resultados/referencia/`.

A. Búsqueda de hiperparámetros: se recalcula la rejilla del pipeline con validación de origen móvil
   (3 bloques) y se guarda la tabla completa (el flujo solo la publicaba como artefacto de Prefect).
B. Diagnóstico sesgo vs. varianza: curva de aprendizaje (tamaño de entrenamiento) y curva de validación
   (número de árboles) del LightGBM elegido. Solo usa entrenamiento y validación; la prueba no interviene.
C. Valor de negocio sobre la prueba: beneficio neto esperado de intervenir al 10 % de mayor riesgo por
   corte, frente a no intervenir, a intervenir al azar y a la heurística de recencia, con supuestos explícitos.
D. Casos aplicados: cinco jugadores del corte 10/11/2008 con su explicación SHAP local. Regla de
   selección (sin escoger a mano): dentro del corte, el caso de riesgo mediano de cada categoría.

Uso (desde la raíz del repositorio):
    python analisis_adicional/diagnostico_y_negocio.py
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from gamepulse_ml import modelos as mod                      # noqa: E402
from gamepulse_ml.config import ConfigModelado               # noqa: E402

REF = RAIZ / "resultados/referencia"
DATASET = RAIZ / "runs/cache/dataset_H30_e7_m2_c20.parquet"
SALIDA = RAIZ / "resultados/adicional"
SEED = 42

# --- Supuestos de negocio (ilustrativos; el estudio no dispone de datos de ingresos ni de costos)
V = 45.0      # valor de retener un personaje activo: 3 meses de suscripción a US$ 15
E = 0.10      # fracción de los que abandonarían y se retienen gracias a la intervención
C = 1.0       # costo por intervención (mensaje + recompensa en el juego)
K_FRAC = 0.10  # capacidad de intervención: 10 % de mayor riesgo por corte


def a_busqueda(data, cfg):
    folds = mod.folds_temporales(data, cfg)
    num, cat = mod.columnas_modelo(data)
    cols = num + cat
    filas = []
    for fam, (fab, grid) in mod.GRID.items():
        for prm in grid:
            aps = []
            for f in folds:
                m = fab(num, cat, **prm).fit(f["train"][cols], f["train"]["churn"])
                aps.append(average_precision_score(f["val"]["churn"], m.predict_proba(f["val"][cols])[:, 1]))
            filas.append({"familia": fam, "params": json.dumps(prm), **{f"PR_AUC_bloque{i + 1}": round(a, 4)
                          for i, a in enumerate(aps)}, "PR_AUC_media": round(float(np.mean(aps)), 4),
                          "PR_AUC_std": round(float(np.std(aps)), 4)})
            print("  rejilla", fam, prm, round(float(np.mean(aps)), 4), flush=True)
    t = pd.DataFrame(filas)
    t["elegido"] = t.groupby("familia")["PR_AUC_media"].transform("max") == t["PR_AUC_media"]
    t.to_csv(SALIDA / "busqueda_hiperparametros.csv", index=False)
    return t


def b_diagnostico(data, cfg):
    p = mod.particion(data, cfg)
    num, cat = mod.columnas_modelo(data)
    cols = num + cat
    tr, va = p["entrenamiento"], p["validacion"]
    prm = {"nl": 15, "lr": 0.05, "n": 300}
    rng = np.random.default_rng(SEED)
    chars = tr["char_id"].unique()
    filas = []
    for frac in (0.1, 0.25, 0.5, 0.75, 1.0):
        res = []
        for rep in range(3 if frac < 1 else 1):
            sel = set(rng.choice(chars, int(len(chars) * frac), replace=False)) if frac < 1 else set(chars)
            sub = tr[tr["char_id"].isin(sel)]
            m = mod._lgbm(num, cat, **prm).fit(sub[cols], sub["churn"])
            res.append((len(sub), average_precision_score(sub["churn"], m.predict_proba(sub[cols])[:, 1]),
                        average_precision_score(va["churn"], m.predict_proba(va[cols])[:, 1])))
        r = np.mean(res, axis=0)
        filas.append({"fraccion_entrenamiento": frac, "filas": int(r[0]), "PR_AUC_entrenamiento": round(r[1], 4),
                      "PR_AUC_validacion": round(r[2], 4)})
        print("  curva aprendizaje", filas[-1], flush=True)
    ca = pd.DataFrame(filas)
    ca.to_csv(SALIDA / "curva_aprendizaje.csv", index=False)
    m = mod._lgbm(num, cat, nl=15, lr=0.05, n=600).fit(tr[cols], tr["churn"])
    cv = pd.DataFrame([{"arboles": k,
                        "PR_AUC_entrenamiento": round(average_precision_score(tr["churn"], m.predict_proba(tr[cols], num_iteration=k)[:, 1]), 4),
                        "PR_AUC_validacion": round(average_precision_score(va["churn"], m.predict_proba(va[cols], num_iteration=k)[:, 1]), 4)}
                       for k in (25, 50, 100, 150, 200, 300, 450, 600)])
    cv.to_csv(SALIDA / "curva_validacion_arboles.csv", index=False)
    print(cv.to_string())
    return ca, cv


def beneficio(y, s, cortes, k_frac, e=E, v=V, c=C):
    """Beneficio neto total y n.º de intervenciones al intervenir al `k_frac` de mayor riesgo en cada corte."""
    neto, n_int, vp = 0.0, 0, 0
    for t in np.unique(cortes):
        ix = np.where(cortes == t)[0]
        k = max(int(round(len(ix) * k_frac)), 1)
        top = ix[np.argsort(-s[ix], kind="stable")[:k]]
        neto += e * v * y[top].sum() - c * k
        n_int += k
        vp += y[top].sum()
    return neto, n_int, int(vp)


def c_negocio(d):
    y, cortes = d["churn"].values, d["fecha_corte"].rank(method="dense").astype(int).values
    politicas = {"Modelo (LightGBM)": d["riesgo_lightgbm"].values,
                 "Heurística de recencia": d["base"].values,
                 "Azar": np.random.default_rng(SEED).random(len(d))}
    filas = [{"politica": "Sin modelo (no intervenir)", "intervenciones": 0, "abandonos_alcanzados": 0, "beneficio_neto_USD": 0.0}]
    for nombre, s in politicas.items():
        neto, n, vp = beneficio(y, s, cortes, K_FRAC)
        filas.append({"politica": nombre, "intervenciones": n, "abandonos_alcanzados": vp, "beneficio_neto_USD": round(neto, 0)})
    # intervenir a todos
    neto_t = E * V * y.sum() - C * len(y)
    filas.append({"politica": "Intervenir a todos", "intervenciones": len(y), "abandonos_alcanzados": int(y.sum()),
                  "beneficio_neto_USD": round(neto_t, 0)})
    # política por umbral (el del pipeline, 0,24)
    th = joblib.load(REF / "modelo/lightgbm.joblib")["umbral"]
    sel = d["riesgo_lightgbm"].values >= th
    filas.append({"politica": f"Modelo con umbral {th:.2f}", "intervenciones": int(sel.sum()),
                  "abandonos_alcanzados": int(y[sel].sum()), "beneficio_neto_USD": round(E * V * y[sel].sum() - C * sel.sum(), 0)})
    tabla = pd.DataFrame(filas)
    tabla.to_csv(SALIDA / "valor_negocio.csv", index=False)
    # beneficio neto vs capacidad K
    ks = np.arange(0.01, 0.61, 0.01)
    curva = pd.DataFrame({"K": ks, **{n: [beneficio(y, s, cortes, k)[0] for k in ks] for n, s in politicas.items()}})
    curva.to_csv(SALIDA / "beneficio_vs_K.csv", index=False)
    # sensibilidad a (efecto, costo) para el modelo al 10 %
    sens = pd.DataFrame([{"efecto": e, "costo_USD": c, "beneficio_neto_modelo_USD": round(beneficio(y, politicas["Modelo (LightGBM)"], cortes, K_FRAC, e=e, c=c)[0], 0),
                          "beneficio_neto_heuristica_USD": round(beneficio(y, politicas["Heurística de recencia"], cortes, K_FRAC, e=e, c=c)[0], 0),
                          "precision_equilibrio": round(c / (e * V), 3)}
                         for e in (0.05, 0.10, 0.20) for c in (0.5, 1.0, 2.0)])
    sens.to_csv(SALIDA / "valor_negocio_sensibilidad.csv", index=False)
    print(tabla.to_string(), "\n", sens.to_string(), "\n K óptimo modelo:", curva.loc[curva["Modelo (LightGBM)"].idxmax(), "K"])
    return curva


def d_casos(d):
    import shap
    pack = joblib.load(REF / "modelo/lightgbm.joblib")
    modelo, cols, th = pack["modelo"], pack["columnas"], pack["umbral"]
    t = pd.Timestamp("2008-11-10")
    g = d[d["fecha_corte"] == t].copy()
    g["percentil_riesgo"] = g["riesgo_lightgbm"].rank(pct=True)
    cats = {
        "Acierto: abandona y se alertó (VP)": (g.churn == 1) & (g.riesgo_lightgbm >= th),
        "Falsa alarma: se alertó y volvió (FP)": (g.churn == 0) & (g.riesgo_lightgbm >= th),
        "Omisión: abandonó sin alerta (FN)": (g.churn == 1) & (g.riesgo_lightgbm < th),
        "Acierto: no abandona y sin alerta (VN)": (g.churn == 0) & (g.riesgo_lightgbm < th),
        "Nivel máximo que abandona (subpoblación débil)": (g.churn == 1) & (g.en_tope == 1),
    }
    elegidos = []
    for nombre, m in cats.items():
        s = g[m].sort_values("riesgo_lightgbm")
        if len(s) == 0:
            continue
        elegidos.append((nombre, s.iloc[len(s) // 2]))
    X = pd.DataFrame([r for _, r in elegidos])[cols]
    for c in ("clase", "raza"):
        X[c] = pd.Categorical(X[c], categories=d[c].cat.categories)
    sv = shap.TreeExplainer(modelo).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    sv = pd.DataFrame(sv, columns=cols)
    filas = []
    for i, (nombre, r) in enumerate(elegidos):
        top = sv.iloc[i].sort_values(key=np.abs, ascending=False).head(4)
        filas.append({"caso": nombre, "char_id": int(r.char_id), "fecha_corte": str(t.date()),
                      "riesgo": round(float(r.riesgo_lightgbm), 3), "percentil_en_el_corte": round(float(r.percentil_riesgo), 3),
                      "abandono_observado": int(r.churn), "clase": str(r.clase), "nivel": int(r.nivel),
                      "dias_desde_ultima": int(r.dias_desde_ultima), "frac_dias_activos_28": round(float(r.frac_dias_activos_28), 2),
                      "factores_shap": "; ".join(f"{k}={X.iloc[i][k] if k in ('clase', 'raza') else round(float(X.iloc[i][k]), 2)} ({v:+.2f})" for k, v in top.items())})
    t2 = pd.DataFrame(filas)
    t2.to_csv(SALIDA / "casos_aplicados.csv", index=False)
    print(t2.to_string())
    return t2


def figura(ca, cv, curva):
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(ca["filas"] / 1000, ca["PR_AUC_entrenamiento"], "o-", color="#C0504D", label="Entrenamiento")
    ax[0].plot(ca["filas"] / 1000, ca["PR_AUC_validacion"], "o-", color="#1F4E79", label="Validación")
    ax[0].set(xlabel="Filas de entrenamiento (miles)", ylabel="PR-AUC", title="Con más datos, la validación sigue subiendo\ny la brecha se reduce")
    ax[0].legend(frameon=False)
    ax[1].plot(cv["arboles"], cv["PR_AUC_entrenamiento"], "o-", color="#C0504D", label="Entrenamiento")
    ax[1].plot(cv["arboles"], cv["PR_AUC_validacion"], "o-", color="#1F4E79", label="Validación")
    ax[1].axvline(300, ls="--", color="grey"); ax[1].text(305, ax[1].get_ylim()[0] + 0.01, "elegido (300)", color="grey")
    ax[1].set(xlabel="Número de árboles", ylabel="PR-AUC", title="La validación se estabiliza hacia 300 árboles;\nel entrenamiento sigue subiendo")
    ax[1].legend(frameon=False, loc="upper left")
    for n, col in (("Modelo (LightGBM)", "#1F4E79"), ("Heurística de recencia", "#E69F00"), ("Azar", "#7F7F7F")):
        ax[2].plot(curva["K"] * 100, curva[n] / 1000, label=n, color=col)
    ax[2].axhline(0, color="black", lw=0.6); ax[2].axvline(10, ls="--", color="grey")
    ax[2].set(xlabel="% de personajes intervenidos por corte", ylabel="Beneficio neto (miles de US$)",
              title="El modelo supera a la heurística\nen todo el rango de capacidad")
    ax[2].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_diagnostico_negocio.png", dpi=200)


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    cfg = ConfigModelado()
    data = mod.cargar_dataset(DATASET)
    print("A. búsqueda"); t = a_busqueda(data, cfg); print(t.to_string())
    print("B. diagnóstico"); ca, cv = b_diagnostico(data, cfg)
    s = pd.read_parquet(REF / "scores_prueba.parquet")
    d = s.merge(data.drop(columns=["churn", "fin_ventana_etiqueta"]), on=["char_id", "fecha_corte"], how="left", validate="one_to_one")
    d["base"] = ((d.dias_desde_ultima / 7 + (1 - d.frac_dias_activos_28)) / 2).values
    print("C. negocio"); curva = c_negocio(d)
    print("D. casos"); d_casos(d)
    figura(ca, cv, curva)


if __name__ == "__main__":
    main()
