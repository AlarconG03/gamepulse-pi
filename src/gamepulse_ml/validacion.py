"""Controles de calidad sobre la capa plata antes de usarla.

Reglas DURAS (si fallan, el pipeline se detiene): columnas y tipos, nulos en columnas obligatorias,
duplicados por (char_id, ts), nivel fuera de 1..80, fechas fuera de 2008.
Reglas BLANDAS (solo advertencia): número de filas o de IDs inconsistentes distinto al documentado.
"""
import hashlib
from pathlib import Path
import pandas as pd
from . import logger

COLUMNAS = {"char_id": "int", "nivel": "int", "raza": "str", "clase": "str", "zona": "str",
            "gremio": "num", "ts": "datetime", "id_inconsistente": "bool"}
OBLIGATORIAS = ["char_id", "nivel", "raza", "clase", "zona", "ts", "id_inconsistente"]
ESPERADO = {"filas": 10_823_177, "ids_inconsistentes": 819}


class ErrorCalidadDatos(Exception):
    """Se lanza cuando la capa plata no cumple una regla dura."""


def hash_archivo(path, bloque=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(bloque):
            h.update(chunk)
    return h.hexdigest()


def _tipo_ok(serie, tipo):
    t = serie.dtype
    return {"int": pd.api.types.is_integer_dtype(t),
            "num": pd.api.types.is_numeric_dtype(t),
            "str": pd.api.types.is_string_dtype(t) or pd.api.types.is_object_dtype(t),
            "datetime": pd.api.types.is_datetime64_any_dtype(t),
            "bool": pd.api.types.is_bool_dtype(t)}[tipo]


def validar_plata(src) -> dict:
    """Valida el parquet de plata. Devuelve un resumen; lanza ErrorCalidadDatos si falla una regla dura."""
    src = Path(src)
    if not src.exists():
        raise ErrorCalidadDatos(f"No existe el archivo de datos: {src}")
    df = pd.read_parquet(src)
    errores, advertencias = [], []

    faltan = [c for c in COLUMNAS if c not in df.columns]
    if faltan:
        raise ErrorCalidadDatos(f"Faltan columnas: {faltan}")
    for c, t in COLUMNAS.items():
        if not _tipo_ok(df[c], t):
            errores.append(f"Tipo inesperado en {c}: {df[c].dtype} (esperado {t})")

    nulos = df[OBLIGATORIAS].isna().sum()
    for c, n in nulos[nulos > 0].items():
        errores.append(f"{n} nulos en columna obligatoria {c}")
    dup = int(df.duplicated(["char_id", "ts"]).sum())
    if dup:
        errores.append(f"{dup} duplicados por (char_id, ts)")
    fuera_nivel = int((~df["nivel"].between(1, 80)).sum())
    if fuera_nivel:
        errores.append(f"{fuera_nivel} filas con nivel fuera de 1..80")
    fuera_fecha = int((~df["ts"].between("2008-01-01", "2009-01-01")).sum())
    if fuera_fecha:
        errores.append(f"{fuera_fecha} filas con ts fuera de 2008")

    ids_incons = int(df.loc[df["id_inconsistente"], "char_id"].nunique())
    if len(df) != ESPERADO["filas"]:
        advertencias.append(f"Filas = {len(df):,}; el README documenta {ESPERADO['filas']:,}")
    if ids_incons != ESPERADO["ids_inconsistentes"]:
        advertencias.append(f"IDs inconsistentes = {ids_incons}; el README documenta {ESPERADO['ids_inconsistentes']}")

    resumen = {
        "archivo": str(src), "sha256": hash_archivo(src), "filas": len(df),
        "personajes": int(df["char_id"].nunique()), "ids_inconsistentes": ids_incons,
        "ts_min": str(df["ts"].min()), "ts_max": str(df["ts"].max()),
        "gremio_nulo_%": round(float(df["gremio"].isna().mean() * 100), 2),
        "errores": errores, "advertencias": advertencias,
    }
    for a in advertencias:
        logger.warning(a)
    if errores:
        for e in errores:
            logger.error(e)
        raise ErrorCalidadDatos("; ".join(errores))
    logger.info("Capa plata válida: %s filas, %s personajes, sha256 %s…",
                f"{resumen['filas']:,}", f"{resumen['personajes']:,}", resumen["sha256"][:12])
    return resumen
