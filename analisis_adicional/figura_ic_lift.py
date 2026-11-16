"""Figura de las mejoras 1 y 2: IC 95 % por bootstrap agrupado y lift en validación frente a prueba.
Uso: python analisis_adicional/figura_ic_lift.py [carpeta_resultados_adicional]"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "resultados/adicional"
ic = pd.read_csv(D / "ic_bootstrap.csv")
lift = pd.read_csv(D / "lift.csv")
orden = ["Línea base (recencia)", "Regresión logística", "Random Forest", "LightGBM", "Ensamble"]
etq = {"Línea base (recencia)": "Línea base", "Regresión logística": "Regresión\nlogística", "Random Forest": "Random\nForest",
       "LightGBM": "LightGBM", "Ensamble": "Ensamble"}
col = "#2a78d6"
fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
for a, met, tit in ((ax[0], "PR_AUC", "PR-AUC en prueba"), (ax[1], "precision@K", "Precisión en el 10 % de mayor riesgo")):
    for i, m in enumerate(orden):
        r = ic[(ic.modelo == m) & (ic.metrica == met)].iloc[0]
        a.errorbar(r.valor, i, xerr=[[r.valor - r.ic95_inf], [r.ic95_sup - r.valor]], fmt="o", color=col, capsize=4, lw=2)
        a.text(r.ic95_sup + 0.006, i, f"{r.valor:.3f}", va="center", fontsize=9)
    a.set_yticks(range(len(orden)))
    a.set_yticklabels([etq[m] for m in orden]); a.invert_yaxis(); a.set_title(tit, fontsize=11, loc="left", fontweight="bold")
    a.grid(axis="x", alpha=.3); a.set_xlabel("valor e IC 95 % (bootstrap por personaje)")
    a.set_xlim(left=a.get_xlim()[0], right=a.get_xlim()[1] + 0.04)
w = 0.38
for i, m in enumerate(orden):
    r = lift[lift.modelo == m].iloc[0]
    ax[2].bar(i - w / 2, r.lift_validacion, w, color="#9ab8e6", label="Validación (tasa base 13,2 %)" if i == 0 else None)
    ax[2].bar(i + w / 2, r.lift_prueba, w, color=col, label="Prueba (tasa base 19,6 %)" if i == 0 else None)
    ax[2].text(i - w / 2, r.lift_validacion + .05, f"{r.lift_validacion:.1f}", ha="center", fontsize=8)
    ax[2].text(i + w / 2, r.lift_prueba + .05, f"{r.lift_prueba:.1f}", ha="center", fontsize=8)
ax[2].set_xticks(range(len(orden))); ax[2].set_xticklabels([etq[m] for m in orden], fontsize=8)
ax[2].set_title("Lift de la PR-AUC (sobre el azar)", fontsize=11, loc="left", fontweight="bold")
ax[2].set_ylim(0, 4.9); ax[2].legend(fontsize=8, frameon=False, loc="upper center", ncol=2); ax[2].grid(axis="y", alpha=.3)
for a in ax:
    for s in ("top", "right"): a.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(D / "fig_ic_lift.png", dpi=170); print("ok")
