"""Carga de la instantánea local que alimenta el tablero.

La carpeta de datos (por defecto `tablero/datos/`) se llena de una de dos formas:
  * Producción: `aws/infra/exportar_tablero.sh` en CloudShell (UNLOAD de Athena a Parquet) y se descarga el zip.
  * Desarrollo: `python tablero/preparar_datos_locales.py` copia los mismos archivos desde el repositorio.
Estructura esperada: scores/, perfiles/, shap_global/, casos/ (Parquet, uno o varios archivos) y modelo/
(modelo.txt y meta.json, usados sólo por el modo sin conexión del detalle).
"""
from pathlib import Path

import pandas as pd

CARPETA = Path(__file__).resolve().parent / "datos"

# Descripción corta de cada variable del modelo (para gráficos y detalle del caso).
VARIABLES = {
    "dias_desde_ultima": "Días desde la última conexión",
    "antiguedad_dias": "Antigüedad del personaje (días)",
    "antiguedad_censurada": "Visto desde el inicio del registro",
    "dias_activos_hist": "Días activos en todo su historial",
    "frac_dias_activos_7": "% de días activos (7 días)",
    "horas_7": "Horas por día activo (7 días)",
    "sesiones_7": "Sesiones por día activo (7 días)",
    "frac_dias_activos_14": "% de días activos (14 días)",
    "horas_14": "Horas por día activo (14 días)",
    "sesiones_14": "Sesiones por día activo (14 días)",
    "frac_dias_activos_28": "% de días activos (28 días)",
    "horas_28": "Horas por día activo (28 días)",
    "sesiones_28": "Sesiones por día activo (28 días)",
    "tendencia_horas": "Tendencia de horas (7 vs 28 días)",
    "tendencia_dias": "Tendencia de días activos (7 vs 28)",
    "horas_por_sesion_28": "Horas por sesión (28 días)",
    "horas_por_dia_activo_28": "Horas por día activo (28 días)",
    "frac_finde_28": "% del juego en fin de semana",
    "salto_max_28": "Mayor ausencia seguida (28 días)",
    "salto_std_28": "Irregularidad de las ausencias",
    "nivel": "Nivel",
    "distancia_al_tope": "Niveles hasta el máximo",
    "en_tope": "En el nivel máximo",
    "niveles_ganados_28": "Niveles ganados (28 días)",
    "dias_desde_subida_nivel": "Días desde la última subida de nivel",
    "n_zonas_28": "Zonas distintas (28 días)",
    "entropia_zonas_28": "Variedad de zonas (entropía)",
    "frac_ciudad_28": "% del tiempo en ciudades",
    "frac_pvp_28": "% del tiempo en PvP",
    "frac_raid_28": "% del tiempo en raids",
    "frac_mazmorra_28": "% del tiempo en mazmorras",
    "frac_mundo_28": "% del tiempo en mundo abierto",
    "tiene_gremio": "Pertenece a un gremio",
    "cambios_gremio_28": "Cambios de gremio (28 días)",
    "tamano_gremio_activo_7": "Miembros activos de su gremio (7 días)",
    "clase": "Clase",
    "raza": "Raza",
}

SENALES = {
    "ausencia_reciente": "Lleva varios días sin conectarse",
    "irregularidad": "Juega de forma irregular",
    "baja_frecuencia": "Pocos días activos en el último mes",
    "poco_tiempo_de_juego": "Sesiones cortas o pocas horas",
    "caida_reciente": "Su actividad cayó frente a semanas anteriores",
    "progresion": "Progresión lenta o estancada",
    "poca_variedad": "Poca variedad de zonas y actividades",
    "vinculo_social": "Sin gremio o con gremio poco activo",
    "poco_historial": "Jugador o personaje nuevo",
    "patron_horario": "Concentra su juego en fines de semana",
    "perfil_personaje": "Clase o raza del personaje",
    "aparcado_en_ciudad": "Pasa casi todo su tiempo en ciudades",
}


# Métrica con la que se evaluaría la acción asociada a cada señal (catálogo de recomendaciones).
METRICA = {
    "ausencia_reciente": "% que vuelve en 7 días",
    "irregularidad": "Días activos en las 4 semanas siguientes",
    "baja_frecuencia": "% de días activos en 30 días",
    "poco_tiempo_de_juego": "Horas por semana",
    "caida_reciente": "Recuperación de la tendencia en 14 días",
    "progresion": "Niveles ganados o actividades de tope en 30 días",
    "poca_variedad": "Zonas distintas visitadas en 30 días",
    "vinculo_social": "% que se une a un gremio y sigue activo a 30 días",
    "poco_historial": "Retención a 30 días de nuevos",
    "patron_horario": "Asistencia a eventos",
    "aparcado_en_ciudad": "% con actividad fuera de ciudad en 30 días",
}


def formato(variable: str, valor) -> str:
    """Valor legible: fracciones como porcentaje y coma decimal."""
    if isinstance(valor, str) or valor is None:
        return valor or "—"
    v = float(valor)
    if variable.startswith("frac_"):
        return f"{v * 100:.0f} %"
    if variable in ("antiguedad_censurada", "en_tope", "tiene_gremio"):
        return "sí" if v >= 0.5 else "no"
    return f"{v:.2f}".rstrip("0").rstrip(".").replace(".", ",") if v % 1 else f"{v:.0f}"


def _leer(nombre: str, carpeta: Path) -> pd.DataFrame:
    ruta = carpeta / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"Falta {ruta}. Ejecute preparar_datos_locales.py o descomprima la exportación de Athena.")
    return pd.read_parquet(ruta)


def cargar(carpeta: Path = CARPETA) -> dict:
    scores = _leer("scores", carpeta)
    scores["fecha_corte"] = pd.to_datetime(scores["fecha_corte"]).dt.normalize()
    scores["prioridad_top_k"] = scores["prioridad_top_k"].astype(bool)
    perfiles = _leer("perfiles", carpeta).sort_values("segmento")
    corto = perfiles["perfil"].str.split(", ").str[:2].str.join(", ")  # "Nivel ~15, sin gremio"
    perfiles["etiqueta"] = "S" + perfiles["segmento"].astype(str) + " · " + corto
    scores = scores.merge(perfiles[["segmento", "etiqueta"]], on="segmento", how="left")
    casos = _leer("casos", carpeta)
    casos["fecha_corte"] = pd.to_datetime(casos["fecha_corte"]).dt.normalize()
    shap = _leer("shap_global", carpeta).sort_values("mean_abs_shap", ascending=False)
    return {"scores": scores, "perfiles": perfiles, "casos": casos, "shap": shap, "carpeta": carpeta}


def lista_de_accion(df: pd.DataFrame, pct: float) -> pd.DataFrame:
    """Marca el pct % de mayor riesgo de cada fecha de corte (la regla del documento usa 10 %)."""
    df = df.copy()
    rango = df.groupby("fecha_corte")["riesgo"].rank(method="first", ascending=False)
    n = df.groupby("fecha_corte")["riesgo"].transform("size")
    df["en_lista"] = rango <= (n * pct / 100).round()
    return df
