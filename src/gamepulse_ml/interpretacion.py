"""Interpretación del modelo: SHAP global y por jugador, perfiles K-Means, ablación controlada de variables
censuradas y evaluación de cada variable por separado."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    silhouette_score,
    roc_auc_score,
    average_precision_score,
)
from . import logger
from .config import ConfigModelado
from .modelos import entrenar_familia, particion, folds_temporales
from .dataset import columnas_modelo

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



def ablacion_controlada(data: pd.DataFrame, cfg: ConfigModelado, mejor: dict) -> dict:
    """Mismo modelo y mismos hiperparámetros, con y sin las variables censuradas; se compara en validación."""
    params = mejor["params"]
    cv_con = next(c["PR_AUC_media"] for c in mejor["cv"] if c["params"] == params)
    sin = entrenar_familia(mejor["familia"], data, cfg, excluir=CENSURADAS, grid=[params])
    cv_sin, test_con, test_sin = sin["cv"][0]["PR_AUC_media"], mejor["metricas_test"], sin["metricas_test"]
    return {"familia": mejor["familia"], "variables_excluidas": CENSURADAS, "hiperparametros": params,
            "n_folds": cfg.n_folds, "PR_AUC_cv_con": cv_con, "PR_AUC_cv_sin": cv_sin,
            "diferencia_cv": round(cv_sin - cv_con, 4), "PR_AUC_test_con": test_con["PR_AUC"],
            "PR_AUC_test_sin": test_sin["PR_AUC"], "diferencia_test": round(test_sin["PR_AUC"] - test_con["PR_AUC"], 4),
            "ROC_AUC_test_con": test_con["ROC_AUC"], "ROC_AUC_test_sin": test_sin["ROC_AUC"]}


def evaluar_variables_individuales(
    data: pd.DataFrame,
    cfg: ConfigModelado
) -> pd.DataFrame:
    """Evalúa cada variable por separado usando validación temporal."""

    num, cat = columnas_modelo(data)
    folds = folds_temporales(data, cfg)
    resultados = []

    for columna in num + cat:
        es_categorica = columna in cat

        # Preparar la variable según su tipo
        if es_categorica:
            transformacion = (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                [columna],
            )
        else:
            transformacion = (
                "num",
                StandardScaler(),
                [columna],
            )

        aucs = []
        aps = []

        # Evaluar cada variable en los mismos bloques temporales
        for bloque in folds:
            modelo = Pipeline([
                ("pre", ColumnTransformer([transformacion])),
                ("clf", LogisticRegression(
                    C=1.0,
                    max_iter=400,
                    class_weight="balanced",
                )),
            ])

            modelo.fit(
                bloque["train"][[columna]],
                bloque["train"]["churn"],
            )

            scores = modelo.predict_proba(
                bloque["val"][[columna]]
            )[:, 1]

            aucs.append(roc_auc_score(
                bloque["val"]["churn"], scores
            ))

            aps.append(average_precision_score(
                bloque["val"]["churn"], scores
            ))

        resultados.append({
            "variable": columna,
            "tipo": "categorica" if es_categorica else "numerica",
            "ROC_AUC_cv_media": round(float(np.mean(aucs)), 4),
            "PR_AUC_cv_media": round(float(np.mean(aps)), 4),
            "n_folds": len(folds),
        })

    # Ordenar las variables por ROC-AUC medio
    return (
        pd.DataFrame(resultados)
        .sort_values(
            ["ROC_AUC_cv_media", "PR_AUC_cv_media"],
            ascending=False,
        )
        .reset_index(drop=True)
    )
