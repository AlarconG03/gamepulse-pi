"""Matriz de recomendaciones: señal de riesgo -> acción posible (objetivo específico 6 de la propuesta).

Las acciones son HIPÓTESIS de intervención, no reglas causales: los datos no demuestran qué acción evita
el abandono. Cada una trae la métrica con la que debería evaluarse en un experimento controlado.
"""
import numpy as np
import pandas as pd

# variable -> grupo de señal
SENAL_DE_VARIABLE = {
    "dias_desde_ultima": "ausencia_reciente", "salto_max_28": "ausencia_reciente", "salto_std_28": "irregularidad",
    "frac_dias_activos_7": "baja_frecuencia", "frac_dias_activos_14": "baja_frecuencia",
    "frac_dias_activos_28": "baja_frecuencia", "sesiones_7": "baja_frecuencia", "sesiones_14": "baja_frecuencia",
    "sesiones_28": "baja_frecuencia", "horas_7": "poco_tiempo_de_juego", "horas_14": "poco_tiempo_de_juego",
    "horas_28": "poco_tiempo_de_juego", "horas_por_sesion_28": "poco_tiempo_de_juego",
    "horas_por_dia_activo_28": "poco_tiempo_de_juego",
    "tendencia_horas": "caida_reciente", "tendencia_dias": "caida_reciente",
    "nivel": "progresion", "distancia_al_tope": "progresion", "en_tope": "progresion",
    "niveles_ganados_28": "progresion", "dias_desde_subida_nivel": "progresion",
    "n_zonas_28": "poca_variedad", "entropia_zonas_28": "poca_variedad", "frac_ciudad_28": "poca_variedad",
    "frac_pvp_28": "poca_variedad", "frac_raid_28": "poca_variedad", "frac_mazmorra_28": "poca_variedad",
    "frac_mundo_28": "poca_variedad",
    "tiene_gremio": "vinculo_social", "cambios_gremio_28": "vinculo_social", "tamano_gremio_activo_7": "vinculo_social",
    "antiguedad_dias": "poco_historial", "antiguedad_censurada": "poco_historial", "dias_activos_hist": "poco_historial",
    "frac_finde_28": "patron_horario", "clase": "perfil_personaje", "raza": "perfil_personaje",
}

CATALOGO = {
    "ausencia_reciente": ("Lleva varios días sin conectarse", "Mensaje o recompensa de regreso en los primeros días de ausencia", "% que vuelve en 7 días"),
    "irregularidad": ("Juega de forma irregular", "Recordatorios alineados a sus días habituales de juego", "Días activos en las 4 semanas siguientes"),
    "baja_frecuencia": ("Pocos días activos en el último mes", "Evento de reactivación o recompensa por conexión diaria", "% de días activos en 30 días"),
    "poco_tiempo_de_juego": ("Sesiones cortas o pocas horas", "Misiones cortas y objetivos alcanzables en una sesión", "Horas por semana"),
    "caida_reciente": ("Su actividad cayó frente a semanas anteriores", "Contacto temprano: evento o contenido nuevo esa semana", "Recuperación de la tendencia en 14 días"),
    "progresion": ("Progresión lenta o estancada", "Revisar tutoriales, orientación o dificultad; en el tope, contenido de nivel máximo", "Niveles ganados o actividades de tope en 30 días"),
    "poca_variedad": ("Poca variedad de zonas y actividades", "Sugerir mazmorras, PvP o zonas nuevas acordes a su nivel", "Zonas distintas visitadas en 30 días"),
    "vinculo_social": ("Sin gremio o con gremio poco activo", "Recomendación de gremios activos y eventos sociales", "% que se une a un gremio y sigue activo a 30 días"),
    "poco_historial": ("Jugador o personaje nuevo", "Acompañamiento de los primeros días (onboarding)", "Retención a 30 días de nuevos"),
    "patron_horario": ("Concentra su juego en fines de semana", "Programar eventos en sus horarios de juego", "Asistencia a eventos"),
    "perfil_personaje": ("Clase o raza del personaje", "No accionable directamente; usar para segmentar campañas", "-"),
    "aparcado_en_ciudad": ("Pasa casi todo su tiempo en ciudades (posible personaje secundario)",
                           "Verificar si es un personaje secundario de un jugador activo antes de intervenir; si no, proponer actividades fuera de ciudad",
                           "% con actividad fuera de ciudad en 30 días"),
}


def senales_por_jugador(sv: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Para cada fila, las n señales (grupos) que más empujan el riesgo hacia arriba (SHAP positivo)."""
    grupos = sv.T.groupby(sv.columns.map(lambda c: SENAL_DE_VARIABLE.get(c, "otra"))).sum().T
    pos = grupos.clip(lower=0)
    top = np.argsort(-pos.values, axis=1)[:, :n]
    nombres = np.array(pos.columns)
    out = pd.DataFrame(index=sv.index)
    for i in range(n):
        s = nombres[top[:, i]]
        val = pos.values[np.arange(len(pos)), top[:, i]]
        out[f"senal_{i + 1}"] = np.where(val > 0, s, None)
    out["accion_sugerida"] = out["senal_1"].map(lambda s: CATALOGO.get(s, ("", "", ""))[1] if s else None)
    return out


def _nombre_segmento(r) -> str:
    fr = {"en ciudad": r["frac_ciudad_28"], "PvP": r["frac_pvp_28"],
          "raids y mazmorras": r["frac_raid_28"] + r["frac_mazmorra_28"]}
    act = max(fr, key=fr.get)
    actividad = act if fr[act] >= 0.15 else "mundo abierto"
    gremio = "con gremio" if r["tiene_gremio"] >= 0.5 else "sin gremio"
    return f"Nivel ~{int(r['nivel'])}, {gremio}, {actividad}, {r['frac_dias_activos_28']:.0%} días activos"


def rasgos_diferenciales(r: pd.Series, glob: pd.Series) -> list:
    """Rasgos del perfil (medianas) que lo distinguen de la población y tienen una acción asociada."""
    rasgos = []
    if r["frac_ciudad_28"] >= 0.6:
        rasgos.append("aparcado_en_ciudad")
    if r["tiene_gremio"] < 0.5:
        rasgos.append("vinculo_social")
    if r["niveles_ganados_28"] == 0 and r["nivel"] < 60:
        rasgos.append("progresion")
    if r["entropia_zonas_28"] < 0.5 * glob["entropia_zonas_28"]:
        rasgos.append("poca_variedad")
    if r["frac_dias_activos_28"] < 0.5 * glob["frac_dias_activos_28"]:
        rasgos.append("baja_frecuencia")
    if r["tendencia_horas"] < 0.8:
        rasgos.append("caida_reciente")
    return rasgos


def matriz_recomendaciones(perfiles: pd.DataFrame, segmentos: pd.Series, senales: pd.DataFrame,
                           riesgo: pd.Series, k_frac: float, medianas_globales: pd.Series) -> pd.DataFrame:
    """Una fila por perfil: tamaño, churn, riesgo, prioridad, señal principal del modelo (SHAP),
    rasgos que lo diferencian y acciones posibles (solo para perfiles con prioridad alta o media)."""
    df = senales.join(segmentos).assign(riesgo=riesgo)
    corte_top = df["riesgo"].quantile(1 - k_frac)
    tasa_global = float(df["riesgo"].mean())
    filas = []
    for seg, r in perfiles.iterrows():
        g = df[df["segmento"] == seg]
        en_top = float((g["riesgo"] >= corte_top).mean())
        prioridad = ("alta" if r["riesgo_medio"] >= 1.5 * tasa_global or en_top >= 1.5 * k_frac
                     else "media" if r["riesgo_medio"] >= tasa_global else "baja")
        senal_modelo = g["senal_1"].value_counts(normalize=True)
        s1 = senal_modelo.index[0] if len(senal_modelo) else None
        rasgos = rasgos_diferenciales(r, medianas_globales)
        if prioridad == "baja":
            acciones, evaluacion = ["Sin intervención prioritaria: monitorear el riesgo semana a semana"], ["-"]
        else:
            claves = list(dict.fromkeys(rasgos + ([s1] if s1 else [])))[:3]
            acciones = [CATALOGO[s][1] for s in claves if s in CATALOGO]
            evaluacion = [CATALOGO[s][2] for s in claves if s in CATALOGO]
        filas.append({"segmento": seg, "perfil": _nombre_segmento(r), "jugadores_prueba": int(r["n"]),
                      "churn_real": round(r["tasa_churn"], 3), "riesgo_medio": round(r["riesgo_medio"], 3),
                      "%_en_top_riesgo": round(en_top * 100, 1), "prioridad": prioridad,
                      "senal_principal_modelo": f"{CATALOGO.get(s1, (s1,))[0]} ({senal_modelo.iloc[0]:.0%})" if s1 else "",
                      "rasgos_diferenciales": "; ".join(CATALOGO[s][0] for s in rasgos) or "Ninguno destacado",
                      "accion_posible": " / ".join(acciones), "como_evaluarla": " / ".join(evaluacion),
                      "naturaleza": "Hipótesis: validar con experimento controlado (A/B)"})
    return pd.DataFrame(filas).sort_values("riesgo_medio", ascending=False)


def catalogo_df() -> pd.DataFrame:
    return pd.DataFrame([{"senal": k, "descripcion": v[0], "accion_posible": v[1], "metrica_de_evaluacion": v[2]}
                         for k, v in CATALOGO.items()])
