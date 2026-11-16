"""Figuras para el informe (PNG). Cada función recibe datos ya calculados y una ruta de salida."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import precision_recall_curve

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
                     "axes.titlelocation": "left"})


def _guardar(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    return str(path)


def cobertura(por_dia_csv, path):
    d = pd.read_csv(por_dia_csv, parse_dates=["dia"])
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.plot(d["dia"], d["personajes"], color=SERIES[0], lw=2)
    for x in d.loc[~d["cubierto"], "dia"]:
        ax.axvspan(x, x + pd.Timedelta(days=1), color=SERIES[1], alpha=0.25, lw=0)
    ax.axvline(pd.Timestamp("2008-11-13"), color=INK2, lw=1, ls="--")
    ax.text(pd.Timestamp("2008-11-15"), ax.get_ylim()[1] * 0.95, "Wrath of the\nLich King", fontsize=8, color=INK2, va="top")
    ax.set_title("Personajes distintos observados por día (2008)"); ax.set_ylabel("personajes / día")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.text(0, -0.2, f"Franjas naranjas: {int((~d['cubierto']).sum())} días sin cobertura del colector.",
            transform=ax.transAxes, fontsize=8, color=INK2)
    return _guardar(fig, path)


def tasa_retorno(retorno_csv, path):
    r = pd.read_csv(retorno_csv)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(r["d_inactivo"], r["%_vuelve_algun_momento"], color=SERIES[0], lw=2, marker="o", ms=6, label="vuelve en algún momento de 2008")
    ax.plot(r["d_inactivo"], r["%_vuelve_en_60d_siguientes"], color=SERIES[1], lw=2, marker="o", ms=6, label="vuelve en los 60 días siguientes")
    ax.axvline(30, color=INK2, lw=1, ls="--")
    ax.set_xlabel("días consecutivos sin aparecer"); ax.set_ylabel("% de rachas que terminan en retorno")
    ax.set_title("Probabilidad de volver tras N días de inactividad"); ax.set_ylim(0, 85)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    return _guardar(fig, path)


def validacion_temporal(data, cfg, path):
    c = data.groupby("fecha_corte")["churn"].mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 3.8))
    bloques = [("Entrenamiento", data["fecha_corte"].min(),
                data.loc[data["fin_ventana_etiqueta"] <= pd.Timestamp(cfg.val_ini), "fecha_corte"].max(), SERIES[0]),
               ("Validación", pd.Timestamp(cfg.val_ini), pd.Timestamp(cfg.val_fin), SERIES[2]),
               ("Prueba", pd.Timestamp(cfg.test_ini), data["fecha_corte"].max(), SERIES[1])]
    for n, a, b, col in bloques:
        ax.axvspan(a - pd.Timedelta(days=3), b + pd.Timedelta(days=3), color=col, alpha=0.12, lw=0)
        ax.text(a, 0.255, n, color=INK, fontsize=9, fontweight="bold")
    ax.plot(c["fecha_corte"], c["churn"], color=INK, lw=2, marker="o", ms=5)
    ax.set_ylim(0, 0.28); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Tasa de churn por fecha de corte semanal"); ax.set_ylabel("% de elegibles con churn")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.text(0, -0.24, "Huecos entre bloques = embargo: ningún corte de entrenamiento tiene una etiqueta que termine\n"
            "después del primer corte evaluado.", transform=ax.transAxes, fontsize=8, color=INK2)
    return _guardar(fig, path)


def shap_top(importancia: pd.DataFrame, path, n=12):
    s = importancia["mean_abs_shap"].head(n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh(s.index, s.values, color=SERIES[0], height=0.6)
    ax.set_title(f"Top {n} variables (|SHAP| medio, prueba)"); ax.set_xlabel("|SHAP| medio (log-odds)")
    ax.grid(axis="y", visible=False)
    return _guardar(fig, path)


def curvas_pr(y, scores: dict, path):
    """Curvas precisión-recall en prueba. `scores` = {nombre: array}; máximo 4 series."""
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for (nombre, s), col in zip(list(scores.items())[:4], SERIES):
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, color=col, lw=2, label=nombre)
    ax.axhline(y.mean(), color=INK2, lw=1, ls="--")
    ax.text(0.99, y.mean() + 0.015, f"azar = {y.mean():.2f}", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("recall"); ax.set_ylabel("precisión"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Curvas precisión-recall en prueba"); ax.legend(frameon=False, fontsize=9)
    return _guardar(fig, path)


def sensibilidad(tabla: pd.DataFrame, path):
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(tabla["horizonte_dias"], tabla["PR_AUC_test"], color=SERIES[0], lw=2, marker="o", ms=7, label="PR-AUC modelo")
    ax.plot(tabla["horizonte_dias"], tabla["tasa_churn_test"], color=INK2, lw=2, marker="o", ms=7, ls="--", label="tasa de churn (azar)")
    ax.plot(tabla["horizonte_dias"], tabla["reaparicion_60d_%"] / 100, color=SERIES[1], lw=2, marker="o", ms=7, label="% de churn que reaparece en 60 d")
    if "cortes_prueba" in tabla:
        for _, r in tabla.iterrows():
            ax.annotate(f"{int(r['cortes_prueba'])} cortes\nde prueba", (r["horizonte_dias"], r["PR_AUC_test"]),
                        xytext=(0, 10), textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_xlabel("horizonte de la etiqueta (días con cobertura)"); ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_title("Sensibilidad de la etiqueta al horizonte"); ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_ylim(0, 0.85)
    return _guardar(fig, path)


def ventanas_corte(path):
    """Diagrama de una fecha de corte: qué datos se usan para las variables, quién es elegible y qué mide la etiqueta."""
    import matplotlib.patches as mpatches
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.set_xlim(-92, 50); ax.set_ylim(-0.6, 4.2); ax.axis("off")
    filas = [  # (y, inicio, fin, color, texto a la izquierda de la barra)
        (3.2, -28, 0, SERIES[0], "VARIABLES (solo pasado)\nactividad, progresión, zonas y gremio\nde los 7, 14 y 28 días previos"),
        (2.0, -7, 0, SERIES[2], "POBLACIÓN (elegibles)\nvisto al menos 1 vez en los 7 días previos\ny con >= 2 días de historia"),
        (0.8, 0, 30, SERIES[1], "ETIQUETA (solo futuro)\nchurn = 1 si NO aparece en los\n30 días con cobertura siguientes"),
    ]
    for y, a, b, col, txt in filas:
        ax.add_patch(mpatches.FancyBboxPatch((a, y - 0.25), b - a, 0.5, boxstyle="round,pad=0,rounding_size=0.25",
                                             color=col, alpha=0.9, lw=0))
        ax.text(min(a, 0) - 2 if a < 0 else -2, y, txt, fontsize=8.5, color=INK, va="center", ha="right")
    ax.add_patch(mpatches.Rectangle((30, 0.55), 4, 0.5, facecolor=SURF, edgecolor=SERIES[1], hatch="///", lw=0.8))
    ax.text(35, 0.8, "+ días sin\ncobertura", fontsize=8, color=INK2, va="center")
    ax.axvline(0, ymin=0.08, ymax=0.93, color=INK, lw=1.5)
    ax.text(0, 4.0, "fecha de corte t (lunes)", ha="center", fontsize=10, fontweight="bold")
    ax.plot([-28, 34], [-0.05, -0.05], color=INK2, lw=0.8)
    for x in (-28, -14, -7, 0, 30):
        ax.plot([x, x], [-0.1, 0.0], color=INK2, lw=0.8)
        ax.text(x, -0.35, f"{x:+d} d" if x else "t", ha="center", fontsize=8, color=INK2)
    ax.set_title("Qué se usa en cada fecha de corte", loc="left")
    return _guardar(fig, path)


def embudo_poblacion(embudo: pd.DataFrame, path):
    """Personajes con historia en cada corte, partidos según por qué entran o no a la población."""
    e = embudo.copy(); e["fecha_corte"] = pd.to_datetime(e["fecha_corte"])
    partes = [("elegibles", "Elegibles (población del modelo)", SERIES[0]),
              ("fuera_un_solo_dia_de_historia", "Activos con 1 solo día de historia", SERIES[3]),
              ("fuera_ausente_8_a_30d", "Ausentes 8-30 días (no se puntúan)", SERIES[1]),
              ("fuera_ausente_mas_de_30d", "Ausentes > 30 días (ya inactivos)", "#c9c7c1")]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.stackplot(e["fecha_corte"], *[e[c] for c, _, _ in partes], colors=[c for _, _, c in partes],
                 labels=[l for _, l, _ in partes], edgecolor=SURF, linewidth=0.8)
    ax.set_title("¿Quién entra a la población en cada fecha de corte?"); ax.set_ylabel("personajes con historia")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[::-1], l[::-1], frameon=False, fontsize=8, loc="upper left")
    return _guardar(fig, path)
