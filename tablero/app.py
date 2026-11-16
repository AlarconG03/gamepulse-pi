"""GamePulse · tablero de riesgo de abandono (sección 6 del documento).

Ejecutar desde la raíz del repositorio:  streamlit run tablero/app.py
Lee la instantánea local exportada de Athena (ver datos.py) y pide el detalle de cada caso a la API.
"""
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import api
import datos

# Paleta (método dataviz, instancia de referencia): azul = modelo/categoría principal, naranja = abandono real,
# gris = referencia. Aumenta/reduce el riesgo usa los dos polos (naranja/azul).
AZUL, NARANJA, GRIS, TINTA, TINTA2 = "#2a78d6", "#eb6834", "#a3a29c", "#1b1f24", "#5a6170"
FUENTE = dict(family="Archivo, system-ui, sans-serif", size=13, color=TINTA2)

st.set_page_config(page_title="GamePulse: riesgo de abandono", page_icon="🎮", layout="wide")


@st.cache_data(show_spinner="Cargando la instantánea…")
def _cargar():
    return datos.cargar()


def _fig(fig: go.Figure, alto: int = 340) -> go.Figure:
    fig.update_layout(separators=",.", height=alto, margin=dict(l=8, r=16, t=8, b=8), font=FUENTE, plot_bgcolor="white",
                      paper_bgcolor="white", bargap=0.25, hoverlabel=dict(bgcolor="white", font_color=TINTA),
                      legend=dict(orientation="h", y=1.08, x=0, title=None))
    fig.update_xaxes(showgrid=False, linecolor="#dfe2e7", title_font=dict(size=12))
    fig.update_yaxes(gridcolor="#eef0f3", zeroline=False, title_font=dict(size=12))
    return fig


def _pct(x: float, dec: int = 0) -> str:
    return f"{x * 100:.{dec}f}\u00a0%".replace(".", ",")  # espacio duro: "33 %" no se parte


def _miles(n: int) -> str:
    return f"{n:,}".replace(",", ".")


# Estilos propios (clases gp-*): no tocan la tabla seleccionable ni los gráficos de Plotly.
st.markdown("""
<style>
/* Tipografía: Archivo; títulos en ancho semiexpandido. Neutros fríos; la paleta de datos no cambia. */
h1, h2, h3, .gp-h {font-stretch: 112%; letter-spacing: -0.012em;}
/* Escala tipográfica: h1 en una línea a 1366 px; cifras del detalle con el mismo peso que las del bloque 1. */
[data-testid="stHeading"] h1 {font-size: clamp(1.7rem, 2.6vw, 2.25rem); line-height: 1.15; padding: .2rem 0 .5rem 0;}
[data-testid="stMetricValue"] {font-size: 1.6rem; font-weight: 650; font-variant-numeric: tabular-nums;}
/* Sin marco al hacer scroll: barra superior opaca y sin contorno de foco en el contenedor principal. */
[data-testid="stHeader"] {background: #ffffff; border: 0; box-shadow: none;}
[data-testid="stMain"], [data-testid="stMain"]:focus, [data-testid="stMain"]:focus-visible,
[data-testid="stAppViewContainer"]:focus, [data-testid="stAppViewContainer"]:focus-visible {outline: none; box-shadow: none;}
.gp-marca {font-weight: 700; font-stretch: 125%; font-size: 1rem; color: #2a78d6; margin-bottom: -.3rem;}
.gp-sub {color: #5a6170; font-size: 1rem; line-height: 1.5; max-width: 72ch; margin: -.3rem 0 .8rem 0;}
.gp-sub b {color: #1b1f24; font-weight: 600;}
.gp-franja {display: flex; flex-wrap: wrap; gap: .4rem 1.2rem; font-size: .84rem; color: #5a6170;
            padding: .55rem 0; border-top: 1px solid #dfe2e7; border-bottom: 1px solid #dfe2e7; margin-bottom: .4rem;}
.gp-franja span {display: inline-flex; align-items: center; gap: .45rem;}
.gp-punto {width: .5rem; height: .5rem; border-radius: 50%; display: inline-block;}
.gp-filtros {font-size: .84rem; color: #5a6170; margin-bottom: 1.6rem;}
/* Encabezado de bloque: número en el margen (los 4 bloques son una secuencia) y título declarativo. */
.gp-sec {display: grid; grid-template-columns: 2.6rem 1fr; align-items: baseline; gap: .2rem;
         border-top: 1px solid #dfe2e7; padding-top: 1.4rem; margin: 1.6rem 0 .6rem 0;}
.gp-n {font-stretch: 125%; font-weight: 700; font-size: 1.05rem; color: #2a78d6; font-variant-numeric: tabular-nums;}
.gp-h {font-size: 1.45rem; font-weight: 650; line-height: 1.25; color: #1b1f24; margin: 0; padding: 0;}
/* Bloque 1: la brecha como cifra principal; el resto, una fila de datos sobria. */
.gp-hero {display: grid; grid-template-columns: minmax(0, 1.5fr) minmax(0, 1fr); gap: 2rem; align-items: end;
          margin: .6rem 0 .6rem 0;}
.gp-cmp {display: flex; align-items: flex-end; gap: 1.6rem; flex-wrap: wrap;}
.gp-big {font-size: clamp(2.2rem, 3.6vw, 3.2rem); font-weight: 700; font-stretch: 112%; line-height: 1;
         font-variant-numeric: tabular-nums; white-space: nowrap;}
.gp-acc {color: #2a78d6;} .gp-mut {color: #a3a29c;}
.gp-lab {font-size: .86rem; color: #5a6170; margin-top: .45rem; max-width: 22ch; line-height: 1.35;}
.gp-vs {font-size: .9rem; color: #5a6170; padding-bottom: 1.9rem;}
.gp-stats {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .8rem; margin: 0;
           border-left: 1px solid #dfe2e7; padding-left: 1.4rem;}
.gp-stats dt {font-size: .82rem; color: #5a6170; line-height: 1.3;}
.gp-stats dd {margin: .25rem 0 0 0; font-size: 1.35rem; font-weight: 650; color: #1b1f24;
              font-variant-numeric: tabular-nums; white-space: nowrap;}
.gp-cuerpo {margin-left: 2.8rem;}
.gp-pie {color: #5a6170; font-size: .82rem; line-height: 1.4; margin: -.6rem 0 .6rem 0;}
@media (max-width: 900px) {
  .gp-hero {grid-template-columns: 1fr; margin-left: 0;} .gp-stats {border-left: 0; padding-left: 0;}
  .gp-sec {grid-template-columns: 2rem 1fr;}
}
</style>
""", unsafe_allow_html=True)


# Variables de los grupos de señal que nombra el título 3 (baja frecuencia y ausencia reciente).
FRECUENCIA_AUSENCIA = {"frac_dias_activos_7", "frac_dias_activos_14", "frac_dias_activos_28", "sesiones_7",
                       "sesiones_14", "sesiones_28", "dias_desde_ultima", "salto_max_28"}


def _pie(texto: str) -> None:
    """Contexto bajo cada gráfico: n, unidad y fuente."""
    st.markdown(f'<div class="gp-pie">{texto}</div>', unsafe_allow_html=True)


def _dato(texto: str, color: str) -> str:
    return f'<span><span class="gp-punto" style="background:{color}"></span>{texto}</span>'


def _seccion(n: int, titulo: str) -> None:
    st.markdown(f'<div class="gp-sec"><div class="gp-n">{n}</div><div class="gp-h" role="heading" aria-level="2">'
                f'{titulo}</div></div>', unsafe_allow_html=True)


def _y(items: list) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " y " + items[-1]


try:
    D = _cargar()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

scores, perfiles, casos, shap = D["scores"], D["perfiles"], D["casos"], D["shap"]
meta = json.load(open(D["carpeta"] / "modelo" / "meta.json", encoding="utf-8"))
UMBRAL = meta["umbral"]

# ------------------------------------------------------------------ filtros (barra lateral)
with st.sidebar:
    st.markdown("### Filtros")
    cortes = sorted(scores["fecha_corte"].unique())  # también se usa en el encabezado
    opciones = ["Todos los cortes"] + [pd.Timestamp(c).strftime("%d/%m/%Y") for c in cortes]
    elegido = st.selectbox("Fecha de corte (lunes)", opciones, index=0,
                           help="Cada lunes se puntúa a los personajes activos en las 4 semanas previas.")
    pct = st.slider("Tamaño de la lista de acción (% de mayor riesgo por corte)", 5, 30, 10, step=5,
                    help="El documento evalúa el 10 % de mayor riesgo de cada corte.")
    ids = perfiles["segmento"].tolist()
    segs = st.pills("Perfiles", ids, selection_mode="multi", default=ids, format_func=lambda x: f"S{x}",
                    help="\n".join(f"- {e}" for e in perfiles["etiqueta"]))
    st.caption("  \n".join(f"S{r.segmento}: {r.etiqueta.split(' · ')[1]}" for r in perfiles.itertuples()))
    desenlace = st.toggle("Mostrar desenlace real", value=True,
                          help="Período de prueba (27/10–01/12/2008): el desenlace ya se conoce. En operación "
                               "sólo existiría el riesgo.")
    st.divider()
    st.caption(f"Modelo {meta['familia']}, umbral de alerta {str(UMBRAL).replace('.', ',')}, datos "
               f"{meta['dataset']}. Fuente: instantánea de Athena (gamepulse.scores, perfiles_segmento, "
               "shap_global y dataset_modelado).")

base = datos.lista_de_accion(scores, pct)            # la lista se calcula sobre TODO el corte
if elegido != "Todos los cortes":
    base = base[base["fecha_corte"] == pd.to_datetime(elegido, dayfirst=True)]
vista = base[base["segmento"].isin(segs or [])]
lista = vista[vista["en_lista"]].sort_values("riesgo", ascending=False)

try:  # .streamlit/secrets.toml (no se versiona): [api] url = "...", clave = "..."
    url, clave = st.secrets["api"]["url"], st.secrets["api"]["clave"]
except Exception:
    url = clave = None

ini, fin = min(cortes), max(cortes)
st.markdown('<div class="gp-marca">GamePulse</div>', unsafe_allow_html=True)
st.title("¿Qué personajes están por dejar de jugar?")
st.markdown('<div class="gp-sub"><b>Para el equipo de retención: a quién contactar esta semana y por qué.</b> '
            f'Período de prueba del {pd.Timestamp(ini):%d/%m} al {pd.Timestamp(fin):%d/%m/%Y} en World of Warcraft '
            "Avatar History, nunca visto por el modelo. El riesgo es la probabilidad estimada de no registrar "
            "actividad en los 30 días siguientes al corte.</div>", unsafe_allow_html=True)
franja = st.empty()


def _franja(detalle: str, en_linea: bool) -> None:
    franja.markdown('<div class="gp-franja">'
                    + _dato(f"Datos: instantánea de Athena con {_miles(len(scores))} puntuaciones en {len(cortes)} cortes", GRIS)
                    + _dato(detalle, AZUL if en_linea else GRIS) + "</div>", unsafe_allow_html=True)


_franja("Detalle de cada personaje: API en AWS (configurada)" if url and clave
        else "Detalle de cada personaje: modo sin conexión", bool(url and clave))
corte_txt = "todos los cortes" if elegido == "Todos los cortes" else f"el corte del {elegido}"
perf_txt = "todos los perfiles" if len(segs or []) == len(ids) else (
    "ningún perfil" if not segs else "los perfiles " + _y([f"S{x}" for x in sorted(segs)]))
st.markdown(f'<div class="gp-filtros">Mostrando {corte_txt}, una lista de acción del {pct}\u00a0% de mayor riesgo '
            f'por corte y {perf_txt}. Los filtros están en la barra lateral.</div>', unsafe_allow_html=True)
if vista.empty:
    st.warning("Ningún personaje cumple estos filtros. Elija al menos un perfil en la barra lateral.")
    st.stop()

# ------------------------------------------------------------------ 1. resumen
tasa_base = base["churn"].mean()
tasa_lista_total = base.loc[base["en_lista"], "churn"].mean()
if desenlace:
    _seccion(1, f"En el {pct}\u00a0% de mayor riesgo abandona el {_pct(tasa_lista_total)}, frente al "
                f"{_pct(tasa_base)} del total")
    grande = (f'<div><div class="gp-big gp-acc">{_pct(tasa_lista_total)}</div><div class="gp-lab">abandona dentro '
              f'de la lista de acción ({pct}\u00a0% de mayor riesgo)</div></div><div class="gp-vs">frente a</div>'
              f'<div><div class="gp-big gp-mut">{_pct(tasa_base)}</div><div class="gp-lab">abandona en el total de '
              'personajes puntuados</div></div>')
    cuarto = ("Con alerta (riesgo ≥ " + str(UMBRAL).replace(".", ",") + ")",
              _miles(int((vista["riesgo"] >= UMBRAL).sum())))
else:
    _seccion(1, f"El modelo prioriza {_miles(int(base['en_lista'].sum()))} personajes de {_miles(len(base))}")
    grande = (f'<div><div class="gp-big gp-acc">{_miles(int(base["en_lista"].sum()))}</div><div class="gp-lab">'
              f'personajes en la lista de acción ({pct}\u00a0%)</div></div><div class="gp-vs">de</div>'
              f'<div><div class="gp-big gp-mut">{_miles(len(base))}</div><div class="gp-lab">personajes '
              'puntuados</div></div>')
    cuarto = ("Con alerta (riesgo ≥ " + str(UMBRAL).replace(".", ",") + ")",
              _miles(int((vista["riesgo"] >= UMBRAL).sum())))
stats = [("Puntuados", _miles(len(vista))), ("En la lista", _miles(len(lista))),
         ("Riesgo medio de la lista", _pct(lista["riesgo"].mean() if len(lista) else 0))]
st.markdown(f'<div class="gp-hero"><div class="gp-cmp">{grande}</div><dl class="gp-stats">'
            + "".join(f"<div><dt>{a}</dt><dd>{b}</dd></div>" for a, b in stats) + "</dl></div>",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ 2. dónde está el riesgo
conc = base[base["en_lista"]]["segmento"].value_counts(normalize=True)
s_a, s_b = conc.index[:2]
_seccion(2, f"Los perfiles S{s_a} y S{s_b} reúnen el {_pct(conc.iloc[:2].sum())} de la lista de acción")
izq, der = st.columns(2)

with izq:
    part = (vista.groupby("etiqueta")
            .agg(poblacion=("char_id", "size"), en_lista=("en_lista", "sum")).reset_index())
    part["% de la población"] = part["poblacion"] / part["poblacion"].sum()
    part["% de la lista"] = part["en_lista"] / max(part["en_lista"].sum(), 1)
    part = part.sort_values("% de la lista")
    fig = go.Figure()
    for col, color in (("% de la población", GRIS), ("% de la lista", AZUL)):
        texto = [_pct(v) if col == "% de la lista" and v >= 0.05 else "" for v in part[col]]
        fig.add_bar(y=part["etiqueta"], x=part[col], name=col, orientation="h", marker_color=color,
                    text=texto, textposition="outside", cliponaxis=False, textfont=dict(color=TINTA, size=12),
                    hovertemplate="%{y}<br>" + col + ": %{x:.1%}<extra></extra>")
    fig.update_layout(barmode="group", xaxis_tickformat=".0%")
    fig.update_xaxes(title="Participación (% de personajes)", range=[0, max(part[["% de la población", "% de la lista"]].max()) * 1.18])
    st.markdown("**Peso de cada perfil en la población y en la lista de acción**")
    st.plotly_chart(_fig(fig), width="stretch")
    _pie(f"n = {_miles(len(vista))} personajes-corte, {_miles(len(lista))} en la lista. Fuente: gamepulse.scores "
         "y perfiles_segmento (Athena).")

with der:
    vista_d = vista.copy()
    vista_d["decil"] = vista_d.groupby("fecha_corte")["riesgo"].transform(
        lambda s: pd.qcut(s.rank(method="first"), 10, labels=False) + 1)
    dec = vista_d.groupby("decil").agg(riesgo=("riesgo", "mean"), churn=("churn", "mean")).reset_index()
    serie, nombre = ("churn", "Abandono real") if desenlace else ("riesgo", "Riesgo medio estimado")
    primero_lista = 11 - max(1, pct // 10)                    # deciles que forman la lista de acción
    colores = [AZUL if d >= primero_lista else GRIS for d in dec["decil"]]
    texto = [_pct(v) if d >= primero_lista else "" for d, v in zip(dec["decil"], dec[serie])]
    promedio = vista[serie].mean()
    fig = go.Figure(go.Bar(x=dec["decil"], y=dec[serie], marker_color=colores, text=texto, textposition="outside",
                           cliponaxis=False, textfont=dict(color=TINTA, size=12),
                           customdata=dec["riesgo"],
                           hovertemplate="Decil %{x}<br>" + nombre + ": %{y:.1%}<br>Riesgo medio estimado: "
                                         "%{customdata:.1%}<extra></extra>"))
    fig.add_hline(y=promedio, line_dash="dot", line_color=TINTA2, line_width=1.5,
                  annotation_text=f"promedio {_pct(promedio)}", annotation_position="top left",
                  annotation_font=dict(color=TINTA2, size=12))
    fig.update_layout(yaxis_tickformat=".0%", showlegend=False)
    fig.update_yaxes(range=[0, dec[serie].max() * 1.18])
    fig.update_xaxes(title="Decil de riesgo dentro de cada corte (10 = mayor riesgo)", dtick=1)
    fig.update_yaxes(title=nombre)
    st.markdown("**El abandono real crece con el decil de riesgo**" if desenlace
                else "**Riesgo medio por decil**")
    st.plotly_chart(_fig(fig), width="stretch")
    _pie(f"En azul, los deciles de la lista de acción. n = {_miles(len(vista))} personajes-corte, unos "
         f"{_miles(round(len(vista) / 10))} por decil; abandono es no tener registros en los 30 días siguientes. "
         "Fuente: gamepulse.scores (Athena).")

with st.expander("Perfiles: rasgos y acción posible (hipótesis)"):
    st.dataframe(perfiles[["etiqueta", "churn_real", "riesgo_medio", "prioridad", "senal_principal_modelo",
                           "accion_posible", "como_evaluarla"]]
                 .rename(columns={"etiqueta": "Perfil", "churn_real": "Abandono real", "riesgo_medio": "Riesgo medio",
                                  "prioridad": "Prioridad", "senal_principal_modelo": "Señal principal",
                                  "accion_posible": "Acción posible", "como_evaluarla": "Cómo evaluarla"}),
                 hide_index=True, width="stretch")

# ------------------------------------------------------------------ 3. señales
_seccion(3, "La baja frecuencia y la ausencia reciente son las señales más asociadas al riesgo")
izq, der = st.columns(2)
with izq:
    top = shap.head(10).iloc[::-1]
    fig = go.Figure(go.Bar(y=[datos.VARIABLES.get(v, v) for v in top["variable"]], x=top["mean_abs_shap"],
                           orientation="h",
                           marker_color=[AZUL if v in FRECUENCIA_AUSENCIA else GRIS for v in top["variable"]],
                           hovertemplate="%{y}<br>|SHAP| medio: %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="|SHAP| medio (log-odds)")
    st.markdown("**Variables que más mueven el riesgo (todo el período de prueba)**")
    st.plotly_chart(_fig(fig, 360), width="stretch")
    _pie("En azul, las variables de frecuencia y de ausencia reciente; se muestran 10 de 37. "
         "n = 22.380 personajes-corte. Fuente: gamepulse.shap_global (Athena).")
with der:
    s1 = lista["senal_1"].map(datos.SENALES).fillna(lista["senal_1"]).value_counts(normalize=True).head(8).iloc[::-1]
    fig = go.Figure(go.Bar(y=s1.index, x=s1.values, orientation="h",
                           marker_color=[AZUL if i == len(s1) - 1 else GRIS for i in range(len(s1))],
                           text=[_pct(v) if v >= 0.005 else "<1\u00a0%" for v in s1.values], textposition="outside", cliponaxis=False,
                           textfont=dict(color=TINTA, size=12),
                           hovertemplate="%{y}<br>%{x:.1%} de la lista<extra></extra>"))
    fig.update_layout(xaxis_tickformat=".0%")
    fig.update_xaxes(range=[0, (s1.max() if len(s1) else 1) * 1.18])
    fig.update_xaxes(title="% de la lista de acción")
    st.markdown("**Señal principal en la lista de acción filtrada**")
    st.plotly_chart(_fig(fig, 360), width="stretch")
    _pie(f"n = {_miles(len(lista))} personajes en la lista (100\u00a0%), una señal principal por personaje. "
         "Fuente: gamepulse.scores (Athena).")
st.caption("SHAP describe asociaciones dentro del modelo, no causas: las acciones sugeridas son hipótesis de "
           "intervención que deben validarse con un experimento controlado.")

# ------------------------------------------------------------------ 4. lista y caso
_seccion(4, "Lista de acción: seleccione un personaje para ver por qué está en riesgo")
tabla = lista.assign(riesgo=lista["riesgo"] * 100, **{"Señal principal": lista["senal_1"].map(datos.SENALES).fillna("—"),
                        "Corte": lista["fecha_corte"].dt.strftime("%d/%m/%Y"),
                        "Perfil": "S" + lista["segmento"].astype(str)})
cols = ["char_id", "Corte", "riesgo", "Perfil", "Señal principal", "accion_sugerida"] + (["churn"] if desenlace else [])
tabla = tabla[cols].rename(columns={"char_id": "Personaje", "riesgo": "Riesgo",
                                    "accion_sugerida": "Acción sugerida (hipótesis)", "churn": "Abandonó"})
st.markdown(f'<div class="gp-pie" style="margin:0 0 .4rem 0">{_miles(len(tabla))} personajes en la lista, del '
            'mayor al menor riesgo. El perfil de cada código está en la barra lateral.</div>', unsafe_allow_html=True)
sel = st.dataframe(tabla, hide_index=True, width="stretch", height=300, on_select="rerun",
                   selection_mode="single-row",
                   column_config={"Riesgo": st.column_config.ProgressColumn(format="%.0f %%", min_value=0, max_value=100,
                                                                                   width=110),
                                  "Abandonó": st.column_config.CheckboxColumn(width="small"),
                                  "Personaje": st.column_config.NumberColumn(format="%d", width="small"),
                                  "Corte": st.column_config.TextColumn(width=95),
                                  "Perfil": st.column_config.TextColumn(width=60),
                                  "Acción sugerida (hipótesis)": st.column_config.TextColumn(width="large")})
st.download_button("Descargar lista (CSV)", tabla.to_csv(index=False).encode("utf-8"), "lista_de_accion.csv",
                   "text/csv")

filas = sel.selection.rows if sel and sel.selection else []
if not filas:
    st.info("Seleccione una fila de la tabla para abrir el detalle del personaje.")
    st.stop()

caso = lista.iloc[filas[0]]
fila = casos[(casos["char_id"] == caso["char_id"]) & (casos["fecha_corte"] == caso["fecha_corte"])]
if fila.empty:
    st.warning("Este personaje no está en la tabla de variables de la instantánea.")
    st.stop()
fila = fila.iloc[0].to_dict()
with st.spinner("Consultando la API…"):
    res, fuente = api.explicar(fila, meta["columnas"], D["carpeta"] / "modelo", url, clave)
en_linea = fuente.startswith("API")
_franja("Detalle de cada personaje: API en AWS (respondió)" if en_linea
        else "Detalle de cada personaje: modo sin conexión", en_linea)

with st.container(border=True):
    st.markdown(f"#### Personaje {int(caso['char_id'])}, corte del {caso['fecha_corte']:%d/%m/%Y}")
    a, b, c = st.columns(3)
    a.metric("Riesgo estimado", _pct(res["riesgo"], 1), "alerta" if res["alerta"] else "sin alerta",
             delta_color="off")
    b.metric("Perfil", f"S{int(caso['segmento'])}", help=caso["etiqueta"])
    if desenlace:
        c.metric("Desenlace real", "Abandonó" if caso["churn"] == 1 else "Siguió activo")
    st.caption(f"Respuesta de: {fuente}.")

    izq, der = st.columns(2)
    with izq:
        f = pd.DataFrame(res["factores"]).iloc[::-1]
        f["nombre"] = f["variable"].map(datos.VARIABLES).fillna(f["variable"])
        f["texto"] = [f"{n} = {datos.formato(v_, v)}" for n, v_, v in zip(f["nombre"], f["variable"], f["valor"])]
        fig = go.Figure(go.Bar(y=f["texto"], x=f["aporte_logodds"], orientation="h",
                               marker_color=[NARANJA if x > 0 else AZUL for x in f["aporte_logodds"]],
                               hovertemplate="%{y}<br>Aporte: %{x:+.2f} log-odds<extra></extra>"))
        fig.add_vline(x=0, line_color="#dfe2e7")
        fig.update_xaxes(title="Aporte al riesgo (log-odds)")
        st.markdown("**Los 5 factores que más pesan en su riesgo** (naranja lo aumenta, azul lo reduce)")
        st.plotly_chart(_fig(fig, 280), width="stretch")
    with der:
        perfil = perfiles[perfiles["segmento"] == caso["segmento"]].iloc[0]
        st.markdown("**Qué haría el equipo de retención**")
        st.markdown(f"- **Acción sugerida (hipótesis):** {caso['accion_sugerida'] or '—'}")
        st.markdown(f"- **Perfil:** {perfil['perfil']}; abandono real del perfil {_pct(perfil['churn_real'])}")
        st.markdown(f"- **Cómo evaluarla:** {datos.METRICA.get(caso['senal_1'], '—')} (experimento controlado)")
        comp = casos.merge(scores[["char_id", "fecha_corte", "segmento"]], on=["char_id", "fecha_corte"])
        comp = comp[comp["segmento"] == caso["segmento"]]
        vars_ = [v for v in f["variable"][::-1] if v not in ("clase", "raza")][:3]
        st.dataframe(pd.DataFrame({"Variable": [datos.VARIABLES.get(v, v) for v in vars_],
                                   "Personaje": [datos.formato(v, fila[v]) for v in vars_],
                                   "Mediana del perfil": [datos.formato(v, comp[v].median()) for v in vars_]}),
                     hide_index=True, width="stretch")
