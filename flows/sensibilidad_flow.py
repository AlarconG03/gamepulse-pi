"""
GamePulse - Flujo Prefect de sensibilidad de la etiqueta.

Reconstruye el dataset con varias definiciones de la etiqueta (21, 30, 45 y 60 días con cobertura) y de la
población (activos en 7 o 14 días; con o sin personajes de un solo día) y entrena el mismo modelo en cada una.
Responde: ¿las decisiones de etiqueta y población son robustas?
Cada horizonte aparece en el dashboard como sus propias tareas (construir-dataset, entrenar-LightGBM).

    python flows/sensibilidad_flow.py
    python flows/sensibilidad_flow.py --horizontes 30 45
"""
import os
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "gamepulse")

import argparse
import sys
from pathlib import Path
import pandas as pd
from prefect import flow, get_run_logger
from prefect.artifacts import create_table_artifact, create_markdown_artifact
from prefect.runtime import flow_run

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src")); sys.path.insert(0, str(RAIZ / "flows"))

from pipeline_flow import validar_datos, construir_dataset, entrenar, _ruta
from gamepulse_ml import figuras as fig, registro as reg, modelos as mod
from sklearn.metrics import average_precision_score
import numpy as np
from gamepulse_ml.config import ConfigEtiqueta, ConfigModelado


ESCENARIOS = [
    # Horizonte de la etiqueta (población base: activos en 7 días con >= 2 días de historia)
    {"nombre": "H21", "horizonte_dias": 21, "eleg_dias": 7, "min_dias_hist": 2},
    {"nombre": "H30 (base)", "horizonte_dias": 30, "eleg_dias": 7, "min_dias_hist": 2},
    {"nombre": "H45", "horizonte_dias": 45, "eleg_dias": 7, "min_dias_hist": 2},
    {"nombre": "H60", "horizonte_dias": 60, "eleg_dias": 7, "min_dias_hist": 2},
    # Definición de la población (etiqueta de 30 días)
    {"nombre": "Población: activos en 14 días", "horizonte_dias": 30, "eleg_dias": 14, "min_dias_hist": 2},
    {"nombre": "Población: incluye personajes de 1 día", "horizonte_dias": 30, "eleg_dias": 7, "min_dias_hist": 1},
]


@flow(name="gamepulse-sensibilidad-etiqueta", log_prints=True,
      description="Compara definiciones de la etiqueta (horizonte) y de la población entrenando el mismo modelo en cada una.")
def sensibilidad_etiqueta(src: str = "data/plata_wowah.parquet", salida: str = "runs",
                          escenarios: list[dict] = ESCENARIOS, familia: str = "LightGBM",
                          reusar_dataset: bool = True) -> list:
    """Para cada escenario reconstruye el dataset y entrena `familia`. Además de las métricas en su propia
    población, mide el desempeño sobre la POBLACIÓN BASE (activos en 7 días, >= 2 días de historia), porque
    una población más amplia agrega casos fáciles y puede inflar las métricas sin predecir mejor."""
    log = get_run_logger()
    src_p, salida_p = _ruta(src), _ruta(salida)
    carpeta = reg.carpeta_corrida(salida_p, f"sensibilidad_{flow_run.name or 'local'}")
    datos = validar_datos(str(src_p))
    cm = ConfigModelado()
    claves_base = None
    filas = []
    for esc in escenarios:
        c_et = ConfigEtiqueta(horizonte_dias=esc["horizonte_dias"], eleg_dias=esc["eleg_dias"],
                              min_dias_hist=esc["min_dias_hist"])
        dset = construir_dataset(str(src_p), str(salida_p / "cache"), c_et.__dict__, datos["sha256"], reusar_dataset)
        r = entrenar.with_options(task_run_name=f"entrenar-{familia}-{c_et.clave()}")(
            familia, dset["archivo"], ConfigModelado(n_folds=1).__dict__)
        mt = r["metricas_test"]
        data = mod.cargar_dataset(dset["archivo"])
        test = mod.particion(data, cm)["prueba"][["char_id", "fecha_corte", "churn"]].assign(s=r["scores_test"])
        if claves_base is None:
            base_cfg = ConfigEtiqueta()
            base = mod.cargar_dataset(Path(salida_p / "cache") / f"dataset_{base_cfg.clave()}.parquet") \
                if (Path(salida_p / "cache") / f"dataset_{base_cfg.clave()}.parquet").exists() else None
            if base is None:
                bd = construir_dataset(str(src_p), str(salida_p / "cache"), base_cfg.__dict__, datos["sha256"], True)
                base = mod.cargar_dataset(bd["archivo"])
            claves_base = mod.particion(base, cm)["prueba"][["char_id", "fecha_corte"]]
        en_base = test.merge(claves_base, on=["char_id", "fecha_corte"])
        ap_base = average_precision_score(en_base["churn"], en_base["s"]) if en_base["churn"].nunique() > 1 else None
        filas.append({"escenario": esc["nombre"], "horizonte_dias": esc["horizonte_dias"], "eleg_dias": esc["eleg_dias"],
                      "min_dias_hist": esc["min_dias_hist"], "elegibles_por_corte": round(dset["filas"] / dset["cortes_validos"]),
                      "cortes_prueba": int(test["fecha_corte"].nunique()), "filas": dset["filas"],
                      "tasa_churn_total": dset["tasa_churn"], "reaparicion_60d_%": dset["reaparicion_60d_%"],
                      "tasa_churn_test": mt["tasa_base"], "PR_AUC_test": mt["PR_AUC"],
                      "lift_PR_AUC": round(mt["PR_AUC"] / mt["tasa_base"], 2),
                      "precision@K_test": mt["precision@K"], "recall@K_test": mt["recall@K"],
                      "PR_AUC_en_poblacion_base": round(ap_base, 4) if ap_base is not None else None,
                      "tasa_churn_poblacion_base": round(float(en_base["churn"].mean()), 4)})
        log.info("%s: %d elegibles/corte, churn %.1f %%, PR-AUC propia %.3f, PR-AUC en población base %s",
                 esc["nombre"], filas[-1]["elegibles_por_corte"], dset["tasa_churn"] * 100, mt["PR_AUC"],
                 filas[-1]["PR_AUC_en_poblacion_base"])
    tabla = pd.DataFrame(filas)
    tabla.to_csv(carpeta / "sensibilidad_etiqueta.csv", index=False)
    horiz = tabla[(tabla["eleg_dias"] == 7) & (tabla["min_dias_hist"] == 2)]
    if len(horiz) > 1:
        fig.sensibilidad(horiz, carpeta / "figuras" / "fig6_sensibilidad_etiqueta.png")
    create_table_artifact(key="gamepulse-sensibilidad-etiqueta", table=tabla.replace({np.nan: None}).to_dict("records"),
                          description="Definiciones de etiqueta y población vs. desempeño")
    md = ["# Sensibilidad de la etiqueta y de la población", "",
          "| Escenario | Elegibles por corte | Churn | Reaparece en 60 d | Cortes de prueba | PR-AUC propia | PR-AUC en población base |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    md += [f"| {r.escenario} | {r.elegibles_por_corte:,} | {r.tasa_churn_total:.1%} | {r['reaparicion_60d_%']} % | "
           f"{r.cortes_prueba} | {r.PR_AUC_test:.3f} | {r.PR_AUC_en_poblacion_base} |" for _, r in tabla.iterrows()]
    md += ["", "La PR-AUC propia depende de la tasa de churn de cada población; para comparar escenarios usen la "
           "columna de población base. Con horizontes largos quedan menos cortes de prueba."]
    create_markdown_artifact(key="gamepulse-reporte-sensibilidad", markdown="\n".join(md))
    return filas


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizontes", type=int, nargs="+", default=None,
                    help="solo estos horizontes con la población base (por defecto: todos los escenarios)")
    ap.add_argument("--familia", default="LightGBM")
    a = ap.parse_args()
    esc = ESCENARIOS if a.horizontes is None else [
        {"nombre": f"H{h}", "horizonte_dias": h, "eleg_dias": 7, "min_dias_hist": 2} for h in a.horizontes]
    sensibilidad_etiqueta(escenarios=esc, familia=a.familia)
