"""Tabla de modelado: una fila por personaje x fecha de corte (lunes), con variables y etiqueta.

Reglas anti-fuga: las variables usan solo filas con ts < t; la etiqueta solo filas con ts >= t.
La etiqueta cuenta días CON COBERTURA: la ventana se alarga en calendario si contiene días caídos.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import logger
from .config import ConfigEtiqueta, FIN_DATOS, EXPANSION
from .auditoria import cobertura_diaria
from .zonas import tipo_zona

VENTANAS = (7, 14, 28)
SESION_MIN = 30
TIPOS = ["ciudad", "pvp", "raid", "mazmorra", "mundo"]
COLS_META = ["char_id", "fecha_corte", "fin_ventana_etiqueta", "churn"]
CATEGORICAS = ["clase", "raza"]


def _entropia(g):
    p = g / g.sum()
    return float(-(p * np.log(p)).sum())


def preparar_tablas_diarias(src, cfg: ConfigEtiqueta):
    """Carga la plata y devuelve (D diaria por personaje, Z zonas por día, cubierto, min_por_pasada, actividad)."""
    df = pd.read_parquet(src)
    _, cubierto, min_pasada = cobertura_diaria(df, cfg.umbral_cobertura)
    n_antes = df["char_id"].nunique()
    df = df[~df["id_inconsistente"]].drop(columns="id_inconsistente")
    logger.info("Excluidos %d IDs inconsistentes", n_antes - df["char_id"].nunique())
    df["dia"] = df["ts"].dt.normalize()
    df["tipo_zona"] = df["zona"].map(tipo_zona)
    df = df.sort_values(["char_id", "ts"])
    df["nueva_sesion"] = df.groupby("char_id")["ts"].diff().dt.total_seconds().div(60).fillna(1e9) > SESION_MIN

    D = df.groupby(["char_id", "dia"]).agg(
        obs=("ts", "size"), sesiones=("nueva_sesion", "sum"), nivel=("nivel", "max"),
        gremio=("gremio", "last"), clase=("clase", "last"), raza=("raza", "last")).reset_index()
    D["horas"] = D["obs"] * D["dia"].map(min_pasada).values / 60
    tz = df.groupby(["char_id", "dia", "tipo_zona"]).size().unstack(fill_value=0)
    D = D.merge(tz.add_prefix("obs_").reset_index(), on=["char_id", "dia"], how="left")
    for tp in TIPOS:
        if f"obs_{tp}" not in D:
            D[f"obs_{tp}"] = 0
    D["subio_nivel"] = D.groupby("char_id")["nivel"].diff().fillna(0) > 0
    D["cambio_gremio"] = D.groupby("char_id")["gremio"].transform(lambda s: s.fillna(-1).diff().fillna(0) != 0)
    D["finde"] = D["dia"].dt.dayofweek >= 5
    Z = df.groupby(["char_id", "dia", "zona"]).size().rename("n").reset_index()
    return D, Z, cubierto


def _fin_ventana(t, cubierto, H):
    acum = cubierto.loc[t:].cumsum()
    idx = acum[acum >= H].index
    return None if len(idx) == 0 else idx[0] + pd.Timedelta(days=1)


def _variables_en_corte(t, D, Z, cubierto, cfg):
    """Variables y etiqueta para los personajes elegibles en la fecha de corte t."""
    hist = D[D["dia"] < t]
    ult = hist.groupby("char_id")["dia"].max()
    ndias = hist.groupby("char_id")["dia"].size()
    eleg = ult.index[(ult >= t - pd.Timedelta(days=cfg.eleg_dias)) & (ndias >= cfg.min_dias_hist)]
    if len(eleg) == 0:
        return None
    w28 = hist[(hist["dia"] >= t - pd.Timedelta(days=28)) & hist["char_id"].isin(eleg)]
    primer = D[D["char_id"].isin(eleg)].groupby("char_id")["dia"].min()
    X = pd.DataFrame(index=pd.Index(eleg, name="char_id"))
    X["fecha_corte"] = t

    X["dias_desde_ultima"] = (t - ult.loc[eleg]).dt.days
    X["antiguedad_dias"] = (t - primer.loc[eleg]).dt.days
    X["antiguedad_censurada"] = (primer.loc[eleg] <= pd.Timestamp("2008-01-07")).astype(int)
    X["dias_activos_hist"] = ndias.loc[eleg]

    raw, covs = {}, {}
    for w in VENTANAS:
        ww = w28[w28["dia"] >= t - pd.Timedelta(days=w)]
        cov = max(int(cubierto.loc[t - pd.Timedelta(days=w): t - pd.Timedelta(days=1)].sum()), 1)
        g = ww.groupby("char_id")
        raw[w] = (g["dia"].size().reindex(eleg).fillna(0), g["horas"].sum().reindex(eleg).fillna(0))
        covs[w] = cov
        X[f"frac_dias_activos_{w}"] = (raw[w][0] / cov).clip(upper=1)
        X[f"horas_{w}"] = raw[w][1] * w / cov
        X[f"sesiones_{w}"] = g["sesiones"].sum().reindex(eleg).fillna(0) * w / cov
    cov_prev = max(covs[28] - covs[7], 1)
    X["tendencia_horas"] = (raw[7][1] / covs[7] + 0.1) / ((raw[28][1] - raw[7][1]) / cov_prev + 0.1)
    X["tendencia_dias"] = (raw[7][0] / covs[7] + 0.05) / ((raw[28][0] - raw[7][0]) / cov_prev + 0.05)
    g = w28.groupby("char_id")
    X["horas_por_sesion_28"] = (g["horas"].sum() / g["sesiones"].sum().clip(lower=1)).reindex(eleg).fillna(0)
    X["horas_por_dia_activo_28"] = (g["horas"].sum() / g["dia"].size()).reindex(eleg).fillna(0)
    X["frac_finde_28"] = (w28[w28["finde"]].groupby("char_id")["horas"].sum() / g["horas"].sum()).reindex(eleg).fillna(0)

    s = w28[["char_id", "dia"]].copy()
    s["salto"] = s.groupby("char_id")["dia"].diff().dt.days
    gs = s.groupby("char_id")["salto"]
    X["salto_max_28"] = np.maximum(gs.max().reindex(eleg).fillna(0), X["dias_desde_ultima"])
    X["salto_std_28"] = gs.std().reindex(eleg).fillna(0)

    last = hist[hist["char_id"].isin(eleg)].sort_values("dia").groupby("char_id").tail(1).set_index("char_id")
    tope = 80 if t >= EXPANSION else 70
    X["nivel"] = last["nivel"].reindex(eleg)
    X["distancia_al_tope"] = tope - X["nivel"]
    X["en_tope"] = (X["nivel"] >= tope).astype(int)
    X["niveles_ganados_28"] = (g["nivel"].max() - g["nivel"].min()).reindex(eleg).fillna(0)
    sub = hist[hist["subio_nivel"] & hist["char_id"].isin(eleg)].groupby("char_id")["dia"].max()
    X["dias_desde_subida_nivel"] = (t - sub.reindex(eleg)).dt.days.fillna(X["antiguedad_dias"])

    zz = Z[(Z["dia"] < t) & (Z["dia"] >= t - pd.Timedelta(days=28)) & Z["char_id"].isin(eleg)]
    zc = zz.groupby(["char_id", "zona"])["n"].sum()
    X["n_zonas_28"] = zc.groupby("char_id").size().reindex(eleg).fillna(0)
    X["entropia_zonas_28"] = zc.groupby("char_id").apply(_entropia).reindex(eleg).fillna(0).clip(lower=0)
    tot = g["obs"].sum()
    for tp in TIPOS:
        X[f"frac_{tp}_28"] = (g[f"obs_{tp}"].sum() / tot).reindex(eleg).fillna(0)

    X["tiene_gremio"] = last["gremio"].reindex(eleg).notna().astype(int)
    X["cambios_gremio_28"] = g["cambio_gremio"].sum().reindex(eleg).fillna(0)
    rec = hist[(hist["dia"] >= t - pd.Timedelta(days=7)) & hist["gremio"].notna()]
    X["tamano_gremio_activo_7"] = last["gremio"].reindex(eleg).map(rec.groupby("gremio")["char_id"].nunique()).fillna(0)
    X["clase"] = last["clase"].reindex(eleg)
    X["raza"] = last["raza"].reindex(eleg)
    return X


def construir_dataset(src, out_path, cfg: ConfigEtiqueta = ConfigEtiqueta(), progreso=None) -> dict:
    """Construye y guarda la tabla de modelado. `progreso(fraccion)` es un callback opcional."""
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    D, Z, cubierto = preparar_tablas_diarias(src, cfg)
    cortes = pd.date_range(cfg.primer_corte, FIN_DATOS, freq="W-MON")
    filas, log = [], []
    for i, t in enumerate(cortes):
        e = _fin_ventana(t, cubierto, cfg.horizonte_dias)
        cub_eleg = int(cubierto.loc[t - pd.Timedelta(days=cfg.eleg_dias): t - pd.Timedelta(days=1)].sum())
        if e is None or e > FIN_DATOS + pd.Timedelta(days=1):
            estado = "descartado: sin seguimiento futuro completo"
        elif cub_eleg < min(cfg.min_dias_cubiertos_eleg, cfg.eleg_dias):
            estado = f"descartado: solo {cub_eleg} días cubiertos antes del corte"
        else:
            X = _variables_en_corte(t, D, Z, cubierto, cfg)
            if X is None:
                estado = "descartado: sin elegibles"
            else:
                fut = D[(D["dia"] >= t) & (D["dia"] < e)]
                X["churn"] = (~X.index.isin(fut["char_id"].unique())).astype(int)
                X["fin_ventana_etiqueta"] = e
                filas.append(X.reset_index())
                estado = f"ok: {len(X)} personajes, churn {X['churn'].mean():.1%}"
        log.append({"fecha_corte": t.date(), "estado": estado})
        logger.info("Corte %s -> %s", t.date(), estado)
        if progreso:
            progreso((i + 1) / len(cortes))

    data = pd.concat(filas, ignore_index=True)
    data.to_parquet(out_path, index=False)
    pd.DataFrame(log).to_csv(out_path.with_name(out_path.stem + "_log_cortes.csv"), index=False)
    resumen = {"archivo": str(out_path), "config": cfg.clave(), "filas": len(data),
               "personajes": int(data["char_id"].nunique()), "cortes_validos": int(data["fecha_corte"].nunique()),
               "tasa_churn": round(float(data["churn"].mean()), 4),
               "variables": len([c for c in data.columns if c not in COLS_META]),
               "reaparicion_60d_%": tasa_reaparicion(data, D)}
    out_path.with_suffix(".json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False))
    logger.info("Dataset %s: %s filas, %d cortes, churn %.1f %%, %.1f %% de los churn reaparece en 60 d",
                cfg.clave(), f"{len(data):,}", resumen["cortes_validos"], resumen["tasa_churn"] * 100,
                resumen["reaparicion_60d_%"])
    return resumen


def tasa_reaparicion(data, D, dias=60) -> float:
    """% de filas con churn = 1 (y seguimiento suficiente) que reaparecen en los `dias` posteriores a la ventana."""
    c = data[(data["churn"] == 1) & (data["fin_ventana_etiqueta"] + pd.Timedelta(days=dias) <= FIN_DATOS)]
    if len(c) == 0:
        return float("nan")
    act = D[["char_id", "dia"]]
    m = act.merge(c[["char_id", "fecha_corte", "fin_ventana_etiqueta"]], on="char_id")
    m = m[(m["dia"] >= m["fin_ventana_etiqueta"]) & (m["dia"] < m["fin_ventana_etiqueta"] + pd.Timedelta(days=dias))]
    return round(len(m[["char_id", "fecha_corte"]].drop_duplicates()) / len(c) * 100, 1)


# Orden de la tabla de pandas: el orden de las columnas fija el de las variables del modelo.
COLUMNAS = ["char_id", "fecha_corte", "dias_desde_ultima", "antiguedad_dias", "antiguedad_censurada",
            "dias_activos_hist", "frac_dias_activos_7", "horas_7", "sesiones_7", "frac_dias_activos_14", "horas_14",
            "sesiones_14", "frac_dias_activos_28", "horas_28", "sesiones_28", "tendencia_horas", "tendencia_dias",
            "horas_por_sesion_28", "horas_por_dia_activo_28", "frac_finde_28", "salto_max_28", "salto_std_28",
            "nivel", "distancia_al_tope", "en_tope", "niveles_ganados_28", "dias_desde_subida_nivel", "n_zonas_28",
            "entropia_zonas_28", "frac_ciudad_28", "frac_pvp_28", "frac_raid_28", "frac_mazmorra_28",
            "frac_mundo_28", "tiene_gremio", "cambios_gremio_28", "tamano_gremio_activo_7", "clase", "raza",
            "churn", "fin_ventana_etiqueta"]


def leer_oro(src, out_path) -> dict:
    """Usa como tabla de modelado la capa oro construida con Spark (aws/jobs/construir_oro.py)."""
    data = pd.read_parquet(src)
    distintas = sorted(set(COLUMNAS) ^ set(data.columns))
    if distintas:
        raise ValueError(f"El oro no tiene las columnas de la tabla de modelado: {distintas}")
    data = data[COLUMNAS].sort_values(["fecha_corte", "char_id"], ignore_index=True)
    data = data.astype({c: "int64" for c in COLUMNAS if str(data[c].dtype) == "int32"})
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_parquet(out_path, index=False)
    return {"archivo": str(out_path), "config": "oro", "origen": str(src), "filas": len(data),
            "personajes": int(data["char_id"].nunique()), "cortes_validos": int(data["fecha_corte"].nunique()),
            "tasa_churn": round(float(data["churn"].mean()), 4), "variables": len(COLUMNAS) - len(COLS_META)}


def columnas_modelo(data: pd.DataFrame, excluir=()):
    num = [c for c in data.columns if c not in COLS_META and c not in CATEGORICAS and c not in excluir]
    return num, list(CATEGORICAS)
