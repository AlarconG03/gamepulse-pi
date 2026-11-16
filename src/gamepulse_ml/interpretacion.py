"""Interpretación del modelo: SHAP global y por jugador, perfiles K-Means y ablación de variables censuradas."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from . import logger
from .config import ConfigModelado
from .modelos import entrenar_familia, particion

PERFIL = ["frac_dias_activos_28", "horas_por_dia_activo_28", "tendencia_horas", "nivel", "niveles_ganados_28",
          "entropia_zonas_28", "frac_ciudad_28", "frac_pvp_28", "frac_raid_28", "frac_mazmorra_28",
          "tiene_gremio", "salto_max_28"]
CENSURADAS = ["antiguedad_censurada", "dias_activos_hist", "antiguedad_dias"]


def shap_valores(modelo, X: pd.DataFrame) -> pd.DataFrame:
    """Valores SHAP (log-odds) por fila y variable. Requiere un modelo de árboles (LightGBM)."""
    import shap
    sv = shap.TreeExplainer(modelo).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    return pd.DataFrame(sv, index=X.index, columns=X.columns)


def importancia_shap(sv: pd.DataFrame) -> pd.DataFrame:
    imp = sv.abs().mean().sort_values(ascending=False)
    return imp.rename("mean_abs_shap").round(4).to_frame()


def _log_perfil(df):
    return np.log1p(df[PERFIL].clip(lower=0))


def perfiles_kmeans(data: pd.DataFrame, cfg: ConfigModelado, riesgo_test: np.ndarray):
    """Ajusta K-Means en entrenamiento y asigna segmentos en prueba. Devuelve (perfiles, segmentos, diagnóstico k)."""
    p = particion(data, cfg)
    tr, te = p["entrenamiento"], p["prueba"]
    sc = StandardScaler().fit(_log_perfil(tr))
    Xtr = sc.transform(_log_perfil(tr))
    muestra = np.random.default_rng(cfg.seed).choice(len(Xtr), size=min(15000, len(Xtr)), replace=False)
    diag = []
    for k in range(3, 9):
        km = KMeans(k, n_init=5, random_state=cfg.seed).fit(Xtr)
        diag.append({"k": k, "inercia": round(float(km.inertia_), 1),
                     "silueta": round(float(silhouette_score(Xtr[muestra], km.labels_[muestra])), 4)})
    km = KMeans(cfg.k_segmentos, n_init=10, random_state=cfg.seed).fit(Xtr)
    seg = km.predict(sc.transform(_log_perfil(te)))
    te = te.assign(segmento=seg, riesgo=riesgo_test)
    perf = te.groupby("segmento").agg(n=("churn", "size"), tasa_churn=("churn", "mean"),
                                      riesgo_medio=("riesgo", "mean"),
                                      **{c: (c, "median") for c in PERFIL}).round(3)
    logger.info("K-Means k=%d: tasa de churn por segmento %s", cfg.k_segmentos,
                perf["tasa_churn"].round(3).to_dict())
    return perf, pd.Series(seg, index=te.index, name="segmento"), pd.DataFrame(diag)


def ablacion_censuradas(data: pd.DataFrame, cfg: ConfigModelado, familia: str, base_test: dict) -> dict:
    """Reentrena `familia` sin las variables que dependen del inicio del dataset y compara en prueba."""
    cfg1 = ConfigModelado(**{**cfg.__dict__, "n_folds": 1})
    r = entrenar_familia(familia, data, cfg1, excluir=CENSURADAS)
    dif = round(r["metricas_test"]["PR_AUC"] - base_test["PR_AUC"], 4)
    logger.info("Ablación %s sin %s: PR-AUC prueba %.4f (diferencia %+.4f)", familia, CENSURADAS,
                r["metricas_test"]["PR_AUC"], dif)
    return {"familia": familia, "excluidas": CENSURADAS, "PR_AUC_con": base_test["PR_AUC"],
            "PR_AUC_sin": r["metricas_test"]["PR_AUC"], "diferencia": dif,
            "precision@K_con": base_test["precision@K"], "precision@K_sin": r["metricas_test"]["precision@K"]}
