"""Análisis de la población objetivo y de la partición.

Responde:
  1. ¿Quién entra a la población en cada fecha de corte y quién queda fuera? (embudo de elegibilidad)
  2. ¿Cuántas veces aparece cada jugador? ¿Los de prueba ya se vieron al entrenar? (estructura de panel)
  3. ¿Cómo le va al modelo con jugadores nunca vistos y en cada subpoblación?
  4. ¿Conviene un modelo por subpoblación en vez de uno global?
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from . import logger
from .config import ConfigEtiqueta, ConfigModelado
from .modelos import particion, precision_recall_at_k, GRID
from .dataset import columnas_modelo


def embudo_poblacion(src, data: pd.DataFrame, cfg: ConfigEtiqueta = ConfigEtiqueta()) -> pd.DataFrame:
    """Para cada fecha de corte usada: cuántos personajes tienen historia y por qué quedan fuera."""
    df = pd.read_parquet(src, columns=["char_id", "ts", "id_inconsistente"])
    n_incons = int(df.loc[df["id_inconsistente"], "char_id"].nunique())
    act = (df.loc[~df["id_inconsistente"], ["char_id"]]
           .assign(dia=df.loc[~df["id_inconsistente"], "ts"].dt.normalize()).drop_duplicates())
    churn = data.groupby("fecha_corte")["churn"].agg(["size", "mean"])
    filas = []
    for t in sorted(data["fecha_corte"].unique()):
        t = pd.Timestamp(t)
        h = act[act["dia"] < t].groupby("char_id")["dia"].agg(["max", "size"])
        ausencia = (t - h["max"]).dt.days
        activo = ausencia <= cfg.eleg_dias
        filas.append({
            "fecha_corte": t.date(), "con_historia": len(h),
            "fuera_ausente_mas_de_30d": int((ausencia > 30).sum()),
            "fuera_ausente_8_a_30d": int(((ausencia > cfg.eleg_dias) & (ausencia <= 30)).sum()),
            "fuera_un_solo_dia_de_historia": int((activo & (h["size"] < cfg.min_dias_hist)).sum()),
            "elegibles": int((activo & (h["size"] >= cfg.min_dias_hist)).sum()),
            "tasa_churn_elegibles": round(float(churn.loc[t, "mean"]), 4),
        })
    e = pd.DataFrame(filas)
    assert (e["elegibles"].values == churn["size"].values).all(), "El embudo no cuadra con el dataset"
    e.attrs["ids_inconsistentes_excluidos"] = n_incons
    return e


def estructura_panel(data: pd.DataFrame, cfg: ConfigModelado) -> dict:
    p = particion(data, cfg)
    filas_por_pj = data.groupby("char_id").size()
    vistos = set(p["entren_valid"]["char_id"])
    test = p["prueba"]
    nuevos = ~test["char_id"].isin(vistos)
    return {
        "filas": len(data), "personajes": int(data["char_id"].nunique()),
        "cortes_por_personaje_mediana": float(filas_por_pj.median()),
        "cortes_por_personaje_p90": float(filas_por_pj.quantile(.9)),
        "%_personajes_en_un_solo_corte": round(float((filas_por_pj == 1).mean() * 100), 1),
        "personajes_prueba": int(test["char_id"].nunique()),
        "%_personajes_prueba_vistos_al_entrenar": round(float(test.loc[~nuevos, "char_id"].nunique()
                                                              / test["char_id"].nunique() * 100), 1),
        "%_filas_prueba_de_personajes_nuevos": round(float(nuevos.mean() * 100), 1),
    }


def _metricas_grupo(g: pd.DataFrame, k_frac: float) -> dict:
    out = {"filas": len(g), "personajes": int(g["char_id"].nunique()), "tasa_churn": round(float(g["churn"].mean()), 3)}
    if g["churn"].nunique() < 2:
        return out
    pk, rk = precision_recall_at_k(g, g["riesgo"].values, k_frac)
    out.update({"PR_AUC": round(average_precision_score(g["churn"], g["riesgo"]), 3),
                "ROC_AUC": round(roc_auc_score(g["churn"], g["riesgo"]), 3),
                "lift_PR_AUC": round(average_precision_score(g["churn"], g["riesgo"]) / g["churn"].mean(), 2),
                "precision@K_dentro_del_grupo": round(pk, 3), "recall@K_dentro_del_grupo": round(rk, 3)})
    return out


def subpoblaciones(test: pd.DataFrame, vistos: set) -> dict:
    """Definiciones de subpoblación sobre el bloque de prueba (máscaras booleanas)."""
    return {
        "Todos": pd.Series(True, index=test.index),
        "Jugadores nunca vistos al entrenar": ~test["char_id"].isin(vistos),
        "Jugadores ya vistos al entrenar": test["char_id"].isin(vistos),
        "Nuevos (< 28 días de antigüedad)": test["antiguedad_dias"] < 28,
        "Establecidos (>= 28 días)": test["antiguedad_dias"] >= 28,
        "Vistos el día anterior al corte": test["dias_desde_ultima"] <= 1,
        "Ausentes 2-3 días": test["dias_desde_ultima"].between(2, 3),
        "Ausentes 4-7 días": test["dias_desde_ultima"] >= 4,
        "En nivel máximo": test["en_tope"] == 1,
        "Subiendo de nivel": test["en_tope"] == 0,
        "Con gremio": test["tiene_gremio"] == 1,
        "Sin gremio": test["tiene_gremio"] == 0,
        "Antes de la expansión (cortes 27/10-10/11)": test["fecha_corte"] < pd.Timestamp("2008-11-13"),
        "Después de la expansión (cortes 17/11-01/12)": test["fecha_corte"] >= pd.Timestamp("2008-11-13"),
    }


def desempeno_por_subpoblacion(data: pd.DataFrame, riesgo: np.ndarray, cfg: ConfigModelado) -> pd.DataFrame:
    p = particion(data, cfg)
    test = p["prueba"].assign(riesgo=riesgo)
    vistos = set(p["entren_valid"]["char_id"])
    filas = [{"subpoblacion": n, **_metricas_grupo(test[m], cfg.k_frac)} for n, m in subpoblaciones(test, vistos).items()]
    return pd.DataFrame(filas)


def modelos_especializados(data: pd.DataFrame, cfg: ConfigModelado, params_lgbm: dict, riesgo_global) -> pd.DataFrame:
    """Entrena un LightGBM solo con cada subpoblación y lo compara con el modelo global en esa misma subpoblación."""
    fabrica, _ = GRID["LightGBM"]
    num, cat = columnas_modelo(data)
    cols = num + cat
    p = particion(data, cfg)
    tr, te = p["entren_valid"], p["prueba"].assign(riesgo_global=riesgo_global)
    cortes = {
        "Nuevos (< 28 días)": lambda d: d["antiguedad_dias"] < 28,
        "Establecidos (>= 28 días)": lambda d: d["antiguedad_dias"] >= 28,
        "En nivel máximo": lambda d: d["en_tope"] == 1,
        "Subiendo de nivel": lambda d: d["en_tope"] == 0,
        "Sin gremio": lambda d: d["tiene_gremio"] == 0,
        "Con gremio": lambda d: d["tiene_gremio"] == 1,
    }
    filas = []
    for nombre, f in cortes.items():
        mtr, mte = f(tr), f(te)
        m = fabrica(num, cat, **params_lgbm).fit(tr.loc[mtr, cols], tr.loc[mtr, "churn"])
        s_esp = m.predict_proba(te.loc[mte, cols])[:, 1]
        y = te.loc[mte, "churn"]
        ap_g = average_precision_score(y, te.loc[mte, "riesgo_global"])
        ap_e = average_precision_score(y, s_esp)
        filas.append({"subpoblacion": nombre, "filas_entrenamiento": int(mtr.sum()), "filas_prueba": int(mte.sum()),
                      "tasa_churn_prueba": round(float(y.mean()), 3), "PR_AUC_modelo_global": round(ap_g, 3),
                      "PR_AUC_modelo_especializado": round(ap_e, 3), "diferencia": round(ap_e - ap_g, 3)})
        logger.info("Modelo especializado %s: PR-AUC %.3f vs global %.3f", nombre, ap_e, ap_g)
    return pd.DataFrame(filas)
