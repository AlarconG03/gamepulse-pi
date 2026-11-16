"""Diagrama de la arquitectura implementada (Medallion sobre AWS). Uso: python analisis_adicional/diagrama_arquitectura.py <salida.png>"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

AZ, BR, PL, OR, GR, VE = "#dbe9f6", "#e8c9a0", "#d9dde2", "#f3dc8a", "#ececec", "#d6ecd2"
fig, ax = plt.subplots(figsize=(11, 6.6)); ax.set_xlim(0, 110); ax.set_ylim(0, 66); ax.axis("off")


def caja(x, y, w, h, txt, c, bold=False, dash=False, fs=7.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1", fc=c, ec="#444",
                                lw=1.1, ls="--" if dash else "-"))
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal")


def flecha(a, b, txt="", dy=1.6, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=1.2, color="#333", ls=ls))
    if txt:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, txt, fontsize=6.6, ha="center", color="#333")


# ---- Fila A: Workflow semanal
ax.add_patch(FancyBboxPatch((0.5, 31), 108, 33, boxstyle="round,pad=0.2,rounding_size=2", fc="none", ec="#2b6cb0", lw=1.5, ls="--"))
ax.text(2.5, 61.5, "AWS Glue Workflow semanal (lunes 03:00, hora de Bogotá): cada Job inicia sólo si el anterior termina bien",
        fontsize=8, color="#2b6cb0", fontweight="bold")
Y, H = 44, 12
caja(2, Y + 1, 8, 10, "Kaggle\nCSV\n643 MB", GR, fs=7.2)
caja(14, Y, 15, H, "Job 1: ingesta\nGlue Python Shell\nvalida y registra", AZ, True, fs=7.4)
caja(33, Y + 1, 11, 10, "BRONCE\nbronce/\nCSV", BR, True, fs=7.4)
caja(48, Y, 15, H, "Job 2: plata\nGlue + Spark\nlimpia, tipa, marca", AZ, True, fs=7.4)
caja(67, Y + 1, 12, 10, "PLATA\nParquet\npor mes", PL, True, fs=7.4)
caja(83, Y, 15, H, "Job 3: oro\nGlue + Spark\n37 variables", AZ, True, fs=7.4)
caja(101.5, Y + 1, 6.3, 10, "ORO", OR, True, fs=7.4)
for a, b in [((10.2, Y + 6), (13.8, Y + 6)), ((29.2, Y + 6), (32.8, Y + 6)), ((44.2, Y + 6), (47.8, Y + 6)),
             ((63.2, Y + 6), (66.8, Y + 6)), ((79.2, Y + 6), (82.8, Y + 6)), ((98.2, Y + 6), (101.3, Y + 6))]:
    flecha(a, b)
caja(14, 34, 15, 6, "Secrets Manager\ntoken de Kaggle", GR, fs=7)
flecha((21.5, 40.3), (21.5, 43.8))
caja(48, 34, 15, 6, "rechazadas/\ncon su motivo", PL, fs=7)
flecha((55.5, 43.8), (55.5, 40.3))

# ---- Fila B: catálogo y consulta
caja(40, 18.5, 68, 8, "AWS Glue Data Catalog + Amazon Athena (SQL sobre plata, oro, scores y rechazadas)", VE, True, fs=7.8)
flecha((73, 43.8), (73, 26.8)); flecha((104.6, 43.8), (104.6, 26.8))
caja(2, 18.5, 22, 8, "S3: oro/scores/\nresultados del modelo", OR, fs=7.2)
flecha((24.3, 22.5), (39.8, 22.5), "tabla scores")

# ---- Fila C: modelo y consumo
ax.add_patch(FancyBboxPatch((0.5, 0.3), 108, 12.5, boxstyle="round,pad=0.2,rounding_size=2", fc="none", ec="#777", lw=1.2, ls=":"))
ax.text(56, 10.6, "Modelo y consumo (fuera del Workflow semanal)", fontsize=7.6, color="#555", fontweight="bold", ha="center")
caja(3, 2, 21, 7.5, "Pipeline de ML (local)\nPrefect, LightGBM, SHAP", GR, True, fs=7.2)
caja(34, 2, 21, 7.5, "S3: modelos/v1/\nmodelo.txt + meta.json", PL, True, fs=7.2)
caja(65, 2, 20, 7.5, "AWS Lambda +\nAPI Gateway: /predict", AZ, True, fs=7.2)
caja(92, 2, 15.5, 7.5, "Tablero Streamlit\n(sección 6)", GR, True, fs=7.2)
flecha((24.3, 5.8), (33.8, 5.8), "publica", -2.4, ls="--"); flecha((55.3, 5.8), (64.8, 5.8), "carga", 1.1)
flecha((85.3, 5.8), (91.8, 5.8), "responde", 1.1)
flecha((13.5, 9.8), (13.5, 18.2), ls="--")
flecha((99.5, 18.2), (99.5, 9.8), "", ls="--")
ax.text(100.3, 14, "instantánea", fontsize=6.6, color="#333")
# Copia local de la plata hacia el pipeline de ML (paso manual)
ax.plot([69.5, 69.5, 27.5, 27.5], [43.8, 29, 29, 8.4], ls="--", lw=1.2, color="#333")
ax.add_patch(FancyArrowPatch((27.5, 8.4), (24.3, 8.4), arrowstyle="-|>", mutation_scale=11, lw=1.2, color="#333", ls="--"))
ax.text(48.5, 29.6, "copia local de plata y oro (manual)", fontsize=6.6, ha="center", color="#333")
ax.text(55, -1.4, "Flecha continua: automática. Flecha discontinua: paso manual con un script. "
        "Zonas Medallion: bronce (crudo), plata (validado), oro (listo para ML).", fontsize=7, color="#444", ha="center")
plt.savefig(sys.argv[1], dpi=200, bbox_inches="tight", facecolor="white")
