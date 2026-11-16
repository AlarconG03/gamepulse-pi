"""Auditoría de cobertura temporal: ¿los días sin registros son inactividad real o huecos del colector?

Responde:
  A. ¿Hay datos todos los días? (cobertura diaria, días sin cobertura)
  B. ¿Cada cuánto consulta el colector? (pasadas, huecos, cambio de cadencia)
  C. ¿30 días sin aparecer es inactividad? (saltos entre días activos, tasa de retorno, saltos ajustados)
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from . import logger
from .config import FIN_DATOS, INICIO_DATOS


def cobertura_diaria(df: pd.DataFrame, umbral: float = 0.20):
    """Devuelve (registros por día, serie booleana 'cubierto', minutos por pasada por día)."""
    calendario = pd.date_range(INICIO_DATOS, FIN_DATOS, freq="D")
    reg = df.groupby(df["ts"].dt.normalize()).size().reindex(calendario, fill_value=0)
    cubierto = reg >= umbral * reg.median()
    ts_u = pd.Series(np.sort(df["ts"].unique()))
    pas = ts_u[ts_u.diff().dt.total_seconds().fillna(1e9) > 120]      # inicio de cada pasada
    gap = pas.diff().dt.total_seconds().div(60)
    min_pasada = gap.groupby(pas.dt.normalize()).median().reindex(calendario).fillna(10).clip(5, 20)
    return reg, cubierto, min_pasada


def ejecutar_auditoria(src, out, umbral_cobertura: float = 0.20) -> dict:
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(src, columns=["char_id", "ts", "id_inconsistente"])
    df["dia"] = df["ts"].dt.normalize()
    calendario = pd.date_range(INICIO_DATOS, FIN_DATOS, freq="D")
    res = {}

    # ---------- A. Cobertura diaria
    por_dia = df.groupby("dia").agg(registros=("char_id", "size"), personajes=("char_id", "nunique"))
    sin_datos = calendario.difference(por_dia.index)
    por_dia = por_dia.reindex(calendario, fill_value=0); por_dia.index.name = "dia"
    med = por_dia["registros"].median()
    por_dia["cubierto"] = por_dia["registros"] >= umbral_cobertura * med

    # ---------- B. Pasadas del colector y huecos
    ts_u = np.sort(df["ts"].unique())
    dif = np.diff(ts_u).astype("timedelta64[s]").astype(int)
    pasadas = pd.Series(ts_u[np.r_[True, dif > 120]])
    gap_min = pasadas.diff().dt.total_seconds().div(60)
    por_dia["pasadas"] = pasadas.dt.normalize().value_counts().reindex(calendario, fill_value=0)
    huecos = pd.DataFrame({"inicio": pasadas.shift(1), "fin": pasadas, "min": gap_min}).dropna()
    huecos = huecos[huecos["min"] > 30].copy()
    huecos["horas"] = (huecos["min"] / 60).round(2)
    huecos["dia_semana"] = huecos["inicio"].dt.day_name()
    huecos["tipo"] = np.where((huecos["dia_semana"] == "Thursday") & (huecos["horas"] < 24),
                              "mantenimiento semanal (jueves)",
                              np.where(huecos["horas"] >= 24, "caída del colector (>= 24 h)", "hueco corto"))
    por_dia["horas_sin_recoleccion"] = (huecos.assign(dia=huecos["inicio"].dt.normalize())
                                        .groupby("dia")["horas"].sum().reindex(calendario, fill_value=0))
    por_dia.to_csv(out / "auditoria_por_dia.csv")
    huecos.sort_values("min", ascending=False).to_csv(out / "auditoria_huecos_recoleccion.csv", index=False)

    cub = por_dia[por_dia["cubierto"]]
    pas_mes = cub.groupby(cub.index.month)["pasadas"].median()
    res["A_cobertura_diaria"] = {
        "dias_calendario": len(calendario),
        "dias_con_datos": int((por_dia["registros"] > 0).sum()),
        "dias_sin_datos": [d.strftime("%Y-%m-%d") for d in sin_datos],
        "dias_sin_cobertura": [d.strftime("%Y-%m-%d") for d in por_dia.index[~por_dia["cubierto"]]],
        "registros_dia_mediana": float(med),
        "personajes_dia_mediana": float(por_dia["personajes"].median()),
    }
    res["B_cadencia"] = {
        "pasadas_totales": int(len(pasadas)),
        "min_entre_pasadas_mediana": round(float(gap_min.median()), 2),
        "pasadas_dia_mediana_por_mes": {int(k): float(v) for k, v in pas_mes.items()},
        "huecos_por_tipo": huecos["tipo"].value_counts().to_dict(),
        "hueco_maximo_horas": float(huecos["horas"].max()) if len(huecos) else 0.0,
        "top5_huecos": huecos.nlargest(5, "min")[["inicio", "fin", "horas"]].astype(str).to_dict("records"),
    }

    # ---------- C. Saltos entre días activos y tasa de retorno (sin IDs inconsistentes)
    ok = df.loc[~df["id_inconsistente"], ["char_id", "dia"]].drop_duplicates().sort_values(["char_id", "dia"])
    ok["sig"] = ok.groupby("char_id")["dia"].shift(-1)
    ok["s"] = (ok["sig"] - ok["dia"]).dt.days
    saltos = ok["s"].dropna()
    saltos.clip(upper=61).value_counts().sort_index().to_csv(out / "auditoria_saltos_dias.csv", header=["n"])
    res["C_saltos"] = {
        "n_saltos": int(len(saltos)),
        "%_salto=1dia": round(float((saltos == 1).mean() * 100), 2),
        "%_salto<=7": round(float((saltos <= 7).mean() * 100), 2),
        "%_salto>=30": round(float((saltos >= 30).mean() * 100), 2),
        "p95": float(saltos.quantile(.95)), "p99": float(saltos.quantile(.99)),
    }
    ok["seguimiento"] = (FIN_DATOS - ok["dia"]).dt.days
    filas = []
    for d in [3, 7, 14, 21, 30, 45, 60, 90]:
        eleg = ok[ok["seguimiento"] >= d + 60]
        alcanzan = eleg[eleg["s"].isna() | (eleg["s"] > d)]
        filas.append({"d_inactivo": d, "rachas": len(alcanzan),
                      "%_vuelve_algun_momento": round(alcanzan["s"].notna().mean() * 100, 1),
                      "%_vuelve_en_60d_siguientes": round((alcanzan["s"] <= d + 60).mean() * 100, 1)})
    retorno = pd.DataFrame(filas)
    retorno.to_csv(out / "auditoria_tasa_retorno.csv", index=False)
    res["C_tasa_retorno"] = retorno.to_dict("records")

    # Saltos >= 30 días que dejan de serlo al descontar días sin cobertura (¿artefacto del colector?)
    sin_cob = np.array(sorted(por_dia.index[~por_dia["cubierto"]]), dtype="datetime64[ns]")
    largos = ok[ok["s"] >= 30]
    a, b = largos["dia"].values[:, None], largos["sig"].values[:, None]
    n_sin = ((sin_cob[None, :] > a) & (sin_cob[None, :] < b)).sum(1)
    observados = largos["s"].values - n_sin - 1
    res["C_saltos_largos_ajustados"] = {
        "saltos_>=30": int(len(largos)),
        "%_con_algun_dia_sin_cobertura": round(float((n_sin > 0).mean() * 100), 1),
        "%_que_bajan_de_30_dias_observados": round(float((observados < 30).mean() * 100), 1),
    }

    pc = df.groupby("char_id").agg(primera=("dia", "min"), ultima=("dia", "max"),
                                   dias_activos=("dia", "nunique"), obs=("ts", "size"))
    pc.to_parquet(out / "resumen_por_personaje.parquet")
    res["historial_por_personaje"] = {
        "personajes": int(len(pc)), "dias_activos_p50": float(pc["dias_activos"].median()),
        "%_personajes_1_dia": round(float((pc["dias_activos"] == 1).mean() * 100), 1),
    }
    (out / "auditoria_resumen.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str))
    logger.info("Auditoría: %d días con datos, %d sin cobertura, hueco máximo %.0f h, %.1f %% de saltos >= 30 días",
                res["A_cobertura_diaria"]["dias_con_datos"], len(res["A_cobertura_diaria"]["dias_sin_cobertura"]),
                res["B_cadencia"]["hueco_maximo_horas"], res["C_saltos"]["%_salto>=30"])
    return res
