"""Partición temporal con embargo, entrenamiento por familia de modelos, selección y evaluación.

Esquema (por FECHA DE CORTE, nunca aleatorio):
  * Selección de hiperparámetros: validación con origen móvil (n_folds bloques de semanas consecutivas;
    el último es el bloque de validación). Cada bloque entrena solo con cortes cuya etiqueta ya se
    conocía el día del primer corte del bloque (embargo).
  * Umbral de decisión: se fija con las predicciones del bloque de validación.
  * Prueba: se reentrena con entrenamiento + validación y se evalúa UNA vez.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss, precision_recall_curve,
                             precision_score, recall_score, f1_score, confusion_matrix)
import lightgbm as lgb
from . import logger
from .config import ConfigModelado
from .dataset import columnas_modelo, CATEGORICAS

BASE = "Heurística recencia (línea base)"


# ----------------------------------------------------------------------------- partición
class ErrorFugaTemporal(Exception):
    """Se lanza si algún bloque de entrenamiento contiene etiquetas posteriores al bloque evaluado."""


def cargar_dataset(path) -> pd.DataFrame:
    data = pd.read_parquet(path)
    for c in CATEGORICAS:
        data[c] = data[c].astype("category")
    return data


def particion(data: pd.DataFrame, cfg: ConfigModelado) -> dict:
    vi, vf, ti = pd.Timestamp(cfg.val_ini), pd.Timestamp(cfg.val_fin), pd.Timestamp(cfg.test_ini)
    p = {"entrenamiento": data[data["fin_ventana_etiqueta"] <= vi],
         "validacion": data[data["fecha_corte"].between(vi, vf)],
         "entren_valid": data[data["fin_ventana_etiqueta"] <= ti],
         "prueba": data[data["fecha_corte"] >= ti]}
    if p["entrenamiento"]["fin_ventana_etiqueta"].max() > p["validacion"]["fecha_corte"].min():
        raise ErrorFugaTemporal("El entrenamiento ve etiquetas posteriores al inicio de la validación")
    if p["entren_valid"]["fin_ventana_etiqueta"].max() > p["prueba"]["fecha_corte"].min():
        raise ErrorFugaTemporal("El reentrenamiento ve etiquetas posteriores al inicio de la prueba")
    return p


def describir_particion(p: dict) -> list:
    return [{"bloque": k, "cortes": f"{v.fecha_corte.min().date()} a {v.fecha_corte.max().date()}",
             "n_cortes": int(v.fecha_corte.nunique()), "filas": len(v), "personajes": int(v.char_id.nunique()),
             "tasa_churn": round(float(v.churn.mean()), 4),
             "etiqueta_conocida_hasta": str(v.fin_ventana_etiqueta.max().date())} for k, v in p.items()]


def folds_temporales(data: pd.DataFrame, cfg: ConfigModelado) -> list:
    """Bloques de validación con origen móvil. El último es el bloque de validación oficial."""
    vi, vf = pd.Timestamp(cfg.val_ini), pd.Timestamp(cfg.val_fin)
    cortes = np.sort(data["fecha_corte"].unique())
    folds = [(vi, vf)]
    previos = [c for c in cortes if c < vi]
    for _ in range(cfg.n_folds - 1):
        bloque = previos[-cfg.semanas_por_fold:]
        if len(bloque) < 2:
            break
        folds.insert(0, (pd.Timestamp(bloque[0]), pd.Timestamp(bloque[-1])))
        previos = previos[:-cfg.semanas_por_fold]
    salida = []
    for a, b in folds:
        tr = data[data["fin_ventana_etiqueta"] <= a]
        va = data[data["fecha_corte"].between(a, b)]
        if tr["fin_ventana_etiqueta"].max() > va["fecha_corte"].min():
            raise ErrorFugaTemporal(f"Fold {a.date()}: entrenamiento con etiquetas del futuro")
        salida.append({"val_ini": a, "val_fin": b, "train": tr, "val": va})
    return salida


# ----------------------------------------------------------------------------- métricas
def precision_recall_at_k(df_: pd.DataFrame, score, k_frac: float):
    r, p = [], []
    for _, g in df_.assign(s=np.asarray(score)).groupby("fecha_corte"):
        k = max(int(round(len(g) * k_frac)), 1)
        top = g.nlargest(k, "s")
        r.append(top["churn"].sum() / max(g["churn"].sum(), 1))
        p.append(top["churn"].mean())
    return float(np.mean(p)), float(np.mean(r))


def metricas(df_: pd.DataFrame, score, k_frac: float, umbral=None, es_probabilidad: bool = True) -> dict:
    y, s = df_["churn"].values, np.asarray(score)
    m = {"PR_AUC": average_precision_score(y, s), "ROC_AUC": roc_auc_score(y, s)}
    m["precision@K"], m["recall@K"] = precision_recall_at_k(df_, s, k_frac)
    m["tasa_base"] = float(y.mean())
    if umbral is not None:
        yp = (s >= umbral).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yp, labels=[0, 1]).ravel()
        m.update({"umbral": umbral, "precision": precision_score(y, yp, zero_division=0),
                  "recall": recall_score(y, yp, zero_division=0), "F1": f1_score(y, yp, zero_division=0),
                  "brier": brier_score_loss(y, np.clip(s, 0, 1)) if es_probabilidad else float("nan"), "VP": tp, "FP": fp, "FN": fn, "VN": tn})
    return {k: (round(float(v), 4) if isinstance(v, (float, np.floating)) else int(v)) for k, v in m.items()}


def umbral_f1(y, s) -> float:
    p, r, th = precision_recall_curve(y, s)
    f1 = 2 * p * r / (p + r + 1e-12)
    return float(th[np.nanargmax(f1[:-1])])


# ----------------------------------------------------------------------------- modelos
def _logistica(num, cat, C):
    pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("m", LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))])


def _bosque(num, cat, min_leaf, seed=42):
    pre = ColumnTransformer([("num", "passthrough", num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("pre", pre), ("m", RandomForestClassifier(
        n_estimators=300, min_samples_leaf=min_leaf, max_features="sqrt", class_weight="balanced_subsample",
        n_jobs=-1, random_state=seed))])


def _lgbm(num, cat, nl, lr, n, seed=42):
    return lgb.LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=nl, min_child_samples=100,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                              random_state=seed, verbose=-1)


GRID = {
    "Regresión logística": (_logistica, [{"C": c} for c in (0.01, 0.1, 1.0)]),
    "Random Forest": (_bosque, [{"min_leaf": m} for m in (20, 50, 100)]),
    "LightGBM": (_lgbm, [{"nl": nl, "lr": lr, "n": n} for nl in (15, 31) for lr, n in ((0.05, 300), (0.03, 600))]),
}


def heuristica(df_: pd.DataFrame) -> np.ndarray:
    """Línea base sin aprendizaje: más días sin aparecer y menos días activos = más riesgo."""
    return ((df_["dias_desde_ultima"] / 7 + (1 - df_["frac_dias_activos_28"])) / 2).values


def evaluar_linea_base(p: dict, cfg: ConfigModelado) -> dict:
    sv, st = heuristica(p["validacion"]), heuristica(p["prueba"])
    th = umbral_f1(p["validacion"]["churn"], sv)
    return {"familia": BASE, "params": {}, "cv": [], "metricas_val": metricas(p["validacion"], sv, cfg.k_frac),
            "metricas_test": metricas(p["prueba"], st, cfg.k_frac, th, es_probabilidad=False), "scores_test": st,
            "modelo": None}


def entrenar_familia(nombre: str, data: pd.DataFrame, cfg: ConfigModelado, excluir=(), grid=None) -> dict:
    """Busca hiperparámetros con validación de origen móvil, fija umbral, reentrena y evalúa en prueba.
    `grid` reemplaza la malla de la familia (p. ej. [params] para evaluar una configuración fija)."""
    fabrica, malla = GRID[nombre]
    grid = grid or malla
    num, cat = columnas_modelo(data, excluir)
    cols = num + cat
    folds = folds_temporales(data, cfg)
    p = particion(data, cfg)
    cv = []
    for prm in grid:
        aps, s_last = [], None
        for f in folds:
            m = fabrica(num, cat, **prm).fit(f["train"][cols], f["train"]["churn"])
            s = m.predict_proba(f["val"][cols])[:, 1]
            aps.append(average_precision_score(f["val"]["churn"], s))
            s_last = s
        cv.append({"params": prm, "PR_AUC_folds": [round(a, 4) for a in aps],
                   "PR_AUC_media": round(float(np.mean(aps)), 4), "PR_AUC_std": round(float(np.std(aps)), 4),
                   "_scores_val": s_last})
        logger.info("%s %s -> PR-AUC por bloque %s (media %.4f)", nombre, prm, cv[-1]["PR_AUC_folds"],
                    cv[-1]["PR_AUC_media"])
    mejor = max(cv, key=lambda c: c["PR_AUC_media"])
    sv = mejor.pop("_scores_val")
    for c in cv:
        c.pop("_scores_val", None)
    th = umbral_f1(p["validacion"]["churn"], sv)
    final = fabrica(num, cat, **mejor["params"]).fit(p["entren_valid"][cols], p["entren_valid"]["churn"])
    st = final.predict_proba(p["prueba"][cols])[:, 1]
    res = {"familia": nombre, "params": mejor["params"], "cv": cv, "columnas": cols,
           "metricas_val": {**metricas(p["validacion"], sv, cfg.k_frac),
                            "PR_AUC_cv_media": mejor["PR_AUC_media"], "PR_AUC_cv_std": mejor["PR_AUC_std"]},
           "metricas_test": metricas(p["prueba"], st, cfg.k_frac, th), "scores_test": st, "scores_val": sv,
           "modelo": final}
    logger.info("%s elegido %s | PR-AUC val %.4f | PR-AUC prueba %.4f | precisión@K prueba %.3f",
                nombre, mejor["params"], res["metricas_val"]["PR_AUC"], res["metricas_test"]["PR_AUC"],
                res["metricas_test"]["precision@K"])
    return res


ENSAMBLE = "Ensamble (promedio de rangos)"


def ensamble(resultados: list, data: pd.DataFrame, cfg: ConfigModelado) -> dict:
    """Promedia el rango percentil de las predicciones de cada familia (las escalas de probabilidad difieren,
    p. ej. la logística con pesos de clase). Responde a la pregunta de la propuesta: ¿un ensamble mejora?"""
    p = particion(data, cfg)
    rv = np.mean([pd.Series(r["scores_val"]).rank(pct=True).values for r in resultados], axis=0)
    rt = np.mean([pd.Series(r["scores_test"]).rank(pct=True).values for r in resultados], axis=0)
    th = umbral_f1(p["validacion"]["churn"], rv)
    return {"familia": ENSAMBLE, "params": {"miembros": [r["familia"] for r in resultados]}, "cv": [],
            "metricas_val": metricas(p["validacion"], rv, cfg.k_frac),
            "metricas_test": metricas(p["prueba"], rt, cfg.k_frac, th, es_probabilidad=False), "scores_test": rt,
            "modelo": None}


def tabla_resultados(resultados: list) -> pd.DataFrame:
    filas = []
    for r in resultados:
        mv, mt = r["metricas_val"], r["metricas_test"]
        filas.append({"modelo": r["familia"], "params": json.dumps(r["params"], ensure_ascii=False),
                      "val_PR_AUC_cv_media": mv.get("PR_AUC_cv_media"), "val_PR_AUC_cv_std": mv.get("PR_AUC_cv_std"),
                      "val_PR_AUC": mv["PR_AUC"], "val_ROC_AUC": mv["ROC_AUC"],
                      **{f"test_{k}": v for k, v in mt.items()}})
    return pd.DataFrame(filas)


def seleccionar(resultados: list) -> dict:
    """Elige el mejor modelo entrenado por PR-AUC media de validación (nunca por la prueba)."""
    cands = [r for r in resultados if r["familia"] not in (BASE, ENSAMBLE)]
    return max(cands, key=lambda r: r["metricas_val"].get("PR_AUC_cv_media", r["metricas_val"]["PR_AUC"]))


def desempeno_por_corte(p: dict, mejor: dict, base: dict, k_frac: float) -> pd.DataFrame:
    filas = []
    test = p["prueba"].assign(s=mejor["scores_test"], b=base["scores_test"])
    for t, g in test.groupby("fecha_corte"):
        k = max(int(round(len(g) * k_frac)), 1)
        filas.append({"fecha_corte": t.date(), "n": len(g), "tasa_churn": round(g.churn.mean(), 3),
                      "PR_AUC_modelo": round(average_precision_score(g.churn, g.s), 3),
                      "PR_AUC_base": round(average_precision_score(g.churn, g.b), 3),
                      "precision@K_modelo": round(g.nlargest(k, "s").churn.mean(), 3),
                      "recall@K_modelo": round(g.nlargest(k, "s").churn.sum() / g.churn.sum(), 3)})
    return pd.DataFrame(filas)


def guardar_modelo(res: dict, carpeta: Path, meta: dict) -> Path:
    carpeta.mkdir(parents=True, exist_ok=True)
    nombre = res["familia"].lower().replace(" ", "_").replace("ó", "o").replace("í", "i")
    path = carpeta / f"{nombre}.joblib"
    joblib.dump({"modelo": res["modelo"], "columnas": res["columnas"], "params": res["params"],
                 "umbral": res["metricas_test"].get("umbral"), "meta": meta}, path)
    return path
