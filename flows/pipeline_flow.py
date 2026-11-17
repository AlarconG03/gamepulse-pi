"""
GamePulse - Flujo Prefect del componente de aprendizaje automático.

Cada etapa del pipeline es una tarea (task) de Prefect, de modo que en el dashboard se ve, por corrida:
estado de cada etapa (Completed / Failed / Retrying), duración, logs, parámetros y artefactos
(tablas de métricas, reporte en markdown, progreso de construcción del dataset).

Ejecución local (con el servidor Prefect levantado, ver README_PREFECT.md):
    python flows/pipeline_flow.py
    python flows/pipeline_flow.py --horizonte-dias 45 --n-folds 1        # cambiar parámetros
"""
import os
os.environ.setdefault("PREFECT_LOGGING_EXTRA_LOGGERS", "gamepulse")   # logs del paquete visibles en Prefect
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")     # evita escrituras de telemetría en SQLite

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from prefect import flow, task, get_run_logger
from prefect.artifacts import (create_markdown_artifact, create_table_artifact, create_progress_artifact,
                               update_progress_artifact)
from prefect.cache_policies import NO_CACHE
from prefect.runtime import flow_run

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))   # por si el paquete no está instalado con pip install -e .

import gamepulse_ml
from gamepulse_ml import validacion, auditoria as aud, dataset as ds, modelos as mod
from gamepulse_ml import interpretacion as interp, recomendaciones as rec, figuras as fig, registro as reg
from gamepulse_ml import poblacion as pob
from gamepulse_ml.config import ConfigEtiqueta, ConfigModelado, a_dict

FAMILIAS = ["Regresión logística", "Random Forest", "LightGBM"]


class ErrorCalidadModelo(Exception):
    """El mejor modelo no supera a la línea base en validación."""


def _ruta(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else RAIZ / p


def _slug(nombre: str) -> str:
    return (nombre.lower().replace(" ", "-").replace("ó", "o").replace("í", "i")
            .replace("(", "").replace(")", ""))


# ============================================================================ tareas
@task(name="validar-datos", retries=2, retry_delay_seconds=5, cache_policy=NO_CACHE)
def validar_datos(src: str) -> dict:
    log = get_run_logger()
    res = validacion.validar_plata(src)
    create_table_artifact(key="gamepulse-validacion-datos", description="Controles de calidad de la capa plata",
                          table=[{"control": k, "valor": str(v)} for k, v in res.items()])
    log.info("Datos válidos (sha256 %s…)", res["sha256"][:12])
    return res


@task(name="auditoria-cobertura", cache_policy=NO_CACHE)
def tarea_auditoria(src: str, carpeta: str, umbral: float) -> dict:
    res = aud.ejecutar_auditoria(src, carpeta, umbral)
    r30 = next(r for r in res["C_tasa_retorno"] if r["d_inactivo"] == 30)
    create_table_artifact(key="gamepulse-auditoria", description="Auditoría de cobertura temporal", table=[
        {"indicador": "Días con datos", "valor": res["A_cobertura_diaria"]["dias_con_datos"]},
        {"indicador": "Días sin cobertura", "valor": len(res["A_cobertura_diaria"]["dias_sin_cobertura"])},
        {"indicador": "Hueco máximo del colector (h)", "valor": res["B_cadencia"]["hueco_maximo_horas"]},
        {"indicador": "% saltos >= 30 días", "valor": res["C_saltos"]["%_salto>=30"]},
        {"indicador": "% saltos >= 30 que bajan de 30 días observados",
         "valor": res["C_saltos_largos_ajustados"]["%_que_bajan_de_30_dias_observados"]},
        {"indicador": "% vuelve en 60 d tras 30 días inactivo", "valor": r30["%_vuelve_en_60d_siguientes"]}])
    return res


@task(name="construir-dataset", cache_policy=NO_CACHE, timeout_seconds=3600)
def construir_dataset(src: str, cache_dir: str, cfg: dict, sha_datos: str, reusar: bool) -> dict:
    """Construye la tabla de modelado. Si ya existe una con el mismo hash de datos y la misma
    configuración de etiqueta, la reutiliza (queda registrado en los logs)."""
    log = get_run_logger()
    c = ConfigEtiqueta(**cfg)
    destino = Path(cache_dir) / f"dataset_{c.clave()}.parquet"
    meta = destino.with_suffix(".json")
    if reusar and destino.exists() and meta.exists():
        m = pd.read_json(meta, typ="series")
        if m.get("sha256_datos") == sha_datos:
            log.info("Reutilizando dataset %s (mismo hash de datos y configuración)", destino.name)
            return {**m.to_dict(), "archivo": str(destino), "reutilizado": True}
        log.info("El hash de los datos cambió: se reconstruye el dataset")
    pid = create_progress_artifact(progress=0.0, key="gamepulse-progreso-dataset",
                                   description=f"Construcción del dataset {c.clave()}")
    res = ds.construir_dataset(src, destino, c, progreso=lambda f: update_progress_artifact(pid, f * 100))
    res["sha256_datos"] = sha_datos
    reg.guardar_json(res, meta)
    return {**res, "reutilizado": False}


@task(name="leer-oro", cache_policy=NO_CACHE)
def leer_oro(oro: str, cache_dir: str) -> dict:
    """Toma la tabla de modelado de la capa oro de AWS (copia local de s3://<bucket>/oro/dataset_modelado/)."""
    res = ds.leer_oro(oro, Path(cache_dir) / "dataset_oro.parquet")
    get_run_logger().info("Tabla de modelado desde el oro: %s filas, %s cortes", f"{res['filas']:,}", res["cortes_validos"])
    return res


@task(name="particion-temporal", cache_policy=NO_CACHE)
def tarea_particion(data_path: str, cfg: dict) -> list:
    data = mod.cargar_dataset(data_path)
    desc = mod.describir_particion(mod.particion(data, ConfigModelado(**cfg)))   # lanza ErrorFugaTemporal
    folds = mod.folds_temporales(data, ConfigModelado(**cfg))
    for f in folds:
        desc.append({"bloque": f"fold val {f['val_ini'].date()}", "cortes": f"{f['val_ini'].date()} a {f['val_fin'].date()}",
                     "n_cortes": int(f["val"].fecha_corte.nunique()), "filas": len(f["val"]),
                     "personajes": int(f["val"].char_id.nunique()), "tasa_churn": round(float(f["val"].churn.mean()), 4),
                     "etiqueta_conocida_hasta": f"entrena hasta {f['train'].fin_ventana_etiqueta.max().date()}"})
    create_table_artifact(key="gamepulse-particion", description="Partición temporal con embargo", table=desc)
    get_run_logger().info("Partición sin fuga verificada: %d bloques de validación con origen móvil", len(folds))
    return desc


@task(name="linea-base", cache_policy=NO_CACHE)
def tarea_linea_base(data_path: str, cfg: dict) -> dict:
    data = mod.cargar_dataset(data_path)
    return mod.evaluar_linea_base(mod.particion(data, ConfigModelado(**cfg)), ConfigModelado(**cfg))


@task(cache_policy=NO_CACHE, retries=1, retry_delay_seconds=10,
      task_run_name="entrenar-{familia}")
def entrenar(familia: str, data_path: str, cfg: dict) -> dict:
    data = mod.cargar_dataset(data_path)
    r = mod.entrenar_familia(familia, data, ConfigModelado(**cfg))
    create_table_artifact(key=f"gamepulse-cv-{_slug(familia)}", description=f"Búsqueda de hiperparámetros: {familia}",
                          table=[{"params": str(c["params"]), "PR_AUC_por_bloque": str(c["PR_AUC_folds"]),
                                  "media": c["PR_AUC_media"], "std": c["PR_AUC_std"]} for c in r["cv"]])
    return r


@task(name="evaluar-ensamble", cache_policy=NO_CACHE)
def evaluar_ensamble(resultados: list, data_path: str, cfg: dict) -> dict:
    r = mod.ensamble(resultados, mod.cargar_dataset(data_path), ConfigModelado(**cfg))
    get_run_logger().info("Ensamble de %d modelos: PR-AUC val %.4f | prueba %.4f", len(resultados),
                          r["metricas_val"]["PR_AUC"], r["metricas_test"]["PR_AUC"])
    return r


@task(name="seleccionar-y-control-de-calidad", cache_policy=NO_CACHE)
def seleccionar(resultados: list, base: dict, exigir_mejora: bool, ens: dict | None = None) -> dict:
    log = get_run_logger()
    mejor = mod.seleccionar(resultados)
    ap_mejor = mejor["metricas_val"]["PR_AUC"]
    ap_base = base["metricas_val"]["PR_AUC"]
    log.info("Mejor en validación: %s (PR-AUC %.4f vs línea base %.4f)", mejor["familia"], ap_mejor, ap_base)
    if exigir_mejora and ap_mejor <= ap_base:
        raise ErrorCalidadModelo(f"{mejor['familia']} no supera a la línea base en validación "
                                 f"({ap_mejor:.4f} <= {ap_base:.4f}); no se publica el modelo")
    if ens is not None:
        d = ens["metricas_val"]["PR_AUC"] - ap_mejor
        log.info("El ensamble %s al mejor modelo individual en validación (%+.4f PR-AUC). Se publica el modelo "
                 "individual salvo mejora > 0.01, por simplicidad de despliegue e interpretación.",
                 "supera" if d > 0 else "no supera", d)
    tabla = mod.tabla_resultados([base] + resultados + ([ens] if ens else []))
    create_table_artifact(key="gamepulse-metricas", description="Comparación de modelos (validación y prueba)",
                          table=tabla.replace({np.nan: None}).to_dict("records"))
    return mejor


@task(name="interpretar-y-recomendar", cache_policy=NO_CACHE)
def interpretar(data_path: str, cfg: dict, resultados: list, mejor: dict, carpeta: str) -> dict:
    log = get_run_logger()
    c = ConfigModelado(**cfg)
    data = mod.cargar_dataset(data_path)
    p = mod.particion(data, c)
    test = p["prueba"]
    arbol = mejor if mejor["familia"] == "LightGBM" else next(r for r in resultados if r["familia"] == "LightGBM")
    if arbol is not mejor:
        log.warning("SHAP se calcula con LightGBM porque el modelo elegido (%s) no es LightGBM", mejor["familia"])
    sv = interp.shap_valores(arbol["modelo"], test[arbol["columnas"]])
    imp = interp.importancia_shap(sv)
    riesgo = pd.Series(mejor["scores_test"], index=test.index)
    perf, seg, diag = interp.perfiles_kmeans(data, c, mejor["scores_test"])
    senales = rec.senales_por_jugador(sv)
    matriz = rec.matriz_recomendaciones(perf, seg, senales, riesgo, c.k_frac,
                                        test[interp.PERFIL].median())

    out = Path(carpeta)
    imp.to_csv(out / "importancia_shap_prueba.csv")
    perf.to_csv(out / "perfiles_kmeans_prueba.csv")
    diag.to_csv(out / "kmeans_diagnostico_k.csv", index=False)
    matriz.to_csv(out / "matriz_recomendaciones.csv", index=False)
    rec.catalogo_df().to_csv(out / "catalogo_senales_acciones.csv", index=False)
    scores = test[["char_id", "fecha_corte", "churn"]].assign(riesgo=riesgo.values, segmento=seg.values)
    for r in resultados:
        scores[f"riesgo_{_slug(r['familia'])}"] = r["scores_test"]
    scores = scores.join(senales)
    k_corte = scores.groupby("fecha_corte")["riesgo"].rank(ascending=False, method="first")
    scores["prioridad_top_k"] = k_corte <= scores.groupby("fecha_corte")["riesgo"].transform(
        lambda s: max(int(round(len(s) * c.k_frac)), 1))
    scores.to_parquet(out / "scores_prueba.parquet", index=False)

    create_table_artifact(key="gamepulse-matriz-recomendaciones", description="Perfil -> señales -> acción posible",
                          table=matriz.to_dict("records"))
    create_table_artifact(key="gamepulse-shap", description="Importancia global (|SHAP| medio, prueba)",
                          table=imp.head(15).reset_index(names="variable").to_dict("records"))
    log.info("Scores, señales por jugador y matriz de recomendaciones guardados en %s", out)
    return {"importancia": imp, "perfiles": perf, "matriz": matriz, "diag_k": diag}


@task(name="analizar-poblacion", cache_policy=NO_CACHE)
def analizar_poblacion(src: str, data_path: str, cfg_et: dict, cfg: dict, resultados: list, mejor: dict,
                       carpeta: str) -> dict:
    """Embudo de elegibilidad, estructura de panel, desempeño en jugadores no vistos y por subpoblación,
    y comparación con modelos especializados por subpoblación."""
    log = get_run_logger()
    c = ConfigModelado(**cfg)
    data = mod.cargar_dataset(data_path)
    out = Path(carpeta)
    emb = pob.embudo_poblacion(src, data, ConfigEtiqueta(**cfg_et))
    emb.to_csv(out / "embudo_poblacion.csv", index=False)
    panel = pob.estructura_panel(data, c)
    reg.guardar_json(panel, out / "estructura_panel.json")
    sub = pob.desempeno_por_subpoblacion(data, mejor["scores_test"], c)
    sub.to_csv(out / "desempeno_por_subpoblacion.csv", index=False)
    lgbm = next(r for r in resultados if r["familia"] == "LightGBM")
    esp = pob.modelos_especializados(data, c, lgbm["params"], lgbm["scores_test"])
    esp.to_csv(out / "modelos_especializados.csv", index=False)
    fig.ventanas_corte(out / "figuras" / "fig7_ventanas_corte.png")
    fig.embudo_poblacion(emb, out / "figuras" / "fig8_embudo_poblacion.png")
    prom = emb.drop(columns="fecha_corte").mean()
    create_table_artifact(key="gamepulse-poblacion", description="Embudo de elegibilidad (promedio por corte) y panel",
                          table=[{"indicador": "Personajes con historia", "valor": round(prom["con_historia"])},
                                 {"indicador": "Fuera: ausentes > 30 días", "valor": round(prom["fuera_ausente_mas_de_30d"])},
                                 {"indicador": "Fuera: ausentes 8-30 días", "valor": round(prom["fuera_ausente_8_a_30d"])},
                                 {"indicador": "Fuera: 1 solo día de historia", "valor": round(prom["fuera_un_solo_dia_de_historia"])},
                                 {"indicador": "Elegibles", "valor": round(prom["elegibles"])}]
                          + [{"indicador": k, "valor": v} for k, v in panel.items()])
    create_table_artifact(key="gamepulse-subpoblaciones", description="Desempeño del modelo elegido por subpoblación (prueba)",
                          table=sub.replace({np.nan: None}).to_dict("records"))
    create_table_artifact(key="gamepulse-modelos-especializados",
                          description="LightGBM entrenado solo con cada subpoblación vs. modelo global",
                          table=esp.to_dict("records"))
    log.info("Población: %d elegibles por corte en promedio; %.1f %% de las filas de prueba son de jugadores nunca vistos",
             prom["elegibles"], panel["%_filas_prueba_de_personajes_nuevos"])
    return {"panel": panel, "subpoblaciones": sub, "especializados": esp}


@task(name="ablacion-variables-censuradas", cache_policy=NO_CACHE)
def ablacion(data_path: str, cfg: dict, mejor: dict, carpeta: str) -> dict:
    """Mismo modelo y mismos hiperparámetros, con y sin las variables censuradas; se compara en validación."""
    r = interp.ablacion_controlada(mod.cargar_dataset(data_path), ConfigModelado(**cfg), mejor)
    pd.DataFrame([r]).to_csv(Path(carpeta) / "ablacion_controlada.csv", index=False)
    create_table_artifact(key="gamepulse-ablacion", description="Modelo con y sin variables censuradas (mismos hiperparámetros)",
                          table=[{k: r[k] for k in ("familia", "PR_AUC_cv_con", "PR_AUC_cv_sin", "diferencia_cv",
                                                    "PR_AUC_test_con", "PR_AUC_test_sin")}])
    return r


@task(name="variables-individuales", cache_policy=NO_CACHE)
def variables_individuales(data_path: str, cfg: dict, carpeta: str) -> dict:
    """Capacidad predictiva de cada variable por separado, sólo con los bloques de validación."""
    rk = interp.evaluar_variables_individuales(mod.cargar_dataset(data_path), ConfigModelado(**cfg))
    rk.to_csv(Path(carpeta) / "variables_individuales_cv.csv", index=False)
    create_table_artifact(key="gamepulse-variables-individuales", description="Ranking por ROC-AUC en validación",
                          table=rk.head(10).to_dict("records"))
    m = rk.iloc[0]
    return {"mejor_variable": str(m["variable"]), "ROC_AUC_cv_media": float(m["ROC_AUC_cv_media"]),
            "PR_AUC_cv_media": float(m["PR_AUC_cv_media"]), "n_variables": len(rk)}


@task(name="figuras", cache_policy=NO_CACHE)
def figuras(carpeta: str, aud_dir: str, data_path: str, cfg: dict, base: dict, resultados: list,
            interp_res: dict) -> list:
    out = Path(carpeta) / "figuras"
    data = mod.cargar_dataset(data_path)
    test = mod.particion(data, ConfigModelado(**cfg))["prueba"]
    rutas = []
    if (Path(aud_dir) / "auditoria_por_dia.csv").exists():
        rutas.append(fig.cobertura(Path(aud_dir) / "auditoria_por_dia.csv", out / "fig1_cobertura_diaria.png"))
        rutas.append(fig.tasa_retorno(Path(aud_dir) / "auditoria_tasa_retorno.csv", out / "fig2_tasa_retorno.png"))
    rutas.append(fig.validacion_temporal(data, ConfigModelado(**cfg), out / "fig3_validacion_temporal.png"))
    rutas.append(fig.shap_top(interp_res["importancia"], out / "fig4_shap_top12.png"))
    curvas = {"Línea base": base["scores_test"], **{r["familia"]: r["scores_test"] for r in resultados}}
    rutas.append(fig.curvas_pr(test["churn"].values, curvas, out / "fig5_curvas_pr.png"))
    rutas.append(fig.curvas_roc(test["churn"].values, curvas, out / "fig9_curvas_roc.png"))
    get_run_logger().info("%d figuras generadas en %s", len(rutas), out)
    return rutas


@task(name="registrar-corrida", cache_policy=NO_CACHE)
def registrar(carpeta: str, salida: str, params: dict, datos: dict, dset: dict, base: dict, resultados: list,
              mejor: dict, abl: dict | None, interp_res: dict, por_corte: pd.DataFrame,
              ens: dict | None = None, univariadas: dict | None = None) -> dict:
    out = Path(carpeta)
    tabla = mod.tabla_resultados([base] + resultados + ([ens] if ens else []))
    tabla.to_csv(out / "resultados_modelos.csv", index=False)
    por_corte.to_csv(out / "desempeno_por_corte_prueba.csv", index=False)
    meta = {"flow_run_id": str(flow_run.id), "flow_run_name": flow_run.name, "prefect_ui": flow_run.ui_url,
            "version_paquete": gamepulse_ml.__version__, "commit_git": reg.commit_git(RAIZ),
            "sha256_datos": datos["sha256"], "dataset": dset.get("config"), "parametros": params}
    for r in resultados:
        mod.guardar_modelo(r, out / "modelos", {**meta, "elegido": r is mejor})
    mt = mejor["metricas_test"]
    ficha = {**meta, "versiones": reg.versiones(), "datos": datos, "dataset": dset,
             "modelo_elegido": mejor["familia"], "hiperparametros": mejor["params"],
             "metricas_validacion": mejor["metricas_val"], "metricas_prueba": mt,
             "linea_base_prueba": base["metricas_test"], "ablacion": abl,
             "variables_individuales": univariadas}
    reg.guardar_json(ficha, out / "ficha_corrida.json")
    reg.anexar_historial(salida, {
        "fecha": pd.Timestamp.now().isoformat(timespec="seconds"), "flow_run": flow_run.name,
        "flow_run_id": str(flow_run.id), "carpeta": out.name, "sha256_datos": datos["sha256"][:12],
        "etiqueta": dset.get("config"), "n_folds": params["n_folds"], "modelo": mejor["familia"],
        "params": str(mejor["params"]), "val_PR_AUC": mejor["metricas_val"]["PR_AUC"],
        "test_PR_AUC": mt["PR_AUC"], "test_precision@K": mt["precision@K"], "test_recall@K": mt["recall@K"],
        "base_test_PR_AUC": base["metricas_test"]["PR_AUC"], "tasa_churn_test": mt["tasa_base"]})
    (Path(salida) / "ultima_corrida.txt").write_text(str(out))

    md = [f"# GamePulse · corrida `{flow_run.name}`", "",
          f"- **Modelo elegido (por validación):** {mejor['familia']} `{mejor['params']}`",
          f"- **Prueba:** PR-AUC {mt['PR_AUC']:.3f} (línea base {base['metricas_test']['PR_AUC']:.3f}, azar {mt['tasa_base']:.3f}) · "
          f"precisión@{int(params['k_frac'] * 100)}% {mt['precision@K']:.3f} · recall@{int(params['k_frac'] * 100)}% {mt['recall@K']:.3f}",
          f"- **Matriz de confusión (umbral {mt['umbral']:.3f}):** VP {mt['VP']}, FP {mt['FP']}, FN {mt['FN']}, VN {mt['VN']}",
          f"- **Etiqueta:** `{dset.get('config')}` · {dset.get('filas', 0):,} filas · churn {dset.get('tasa_churn', 0):.1%}"
          f" · {dset.get('reaparicion_60d_%')} % de los churn reaparece en 60 días",
          f"- **Datos:** sha256 `{datos['sha256'][:16]}…` · {datos['filas']:,} filas",
          f"- **Carpeta de la corrida:** `{out}`"]
    if abl:
        md.append(f"- **Ablación sin variables censuradas:** PR-AUC en validación {abl['PR_AUC_cv_sin']:.3f} "
                  f"(con: {abl['PR_AUC_cv_con']:.3f})")
    md += ["", "## Métricas por modelo (prueba)", "", "| Modelo | PR-AUC val (media CV) | PR-AUC prueba | precisión@K | recall@K |",
           "| --- | --- | --- | --- | --- |"]
    for _, r in tabla.iterrows():
        cv = "—" if pd.isna(r["val_PR_AUC_cv_media"]) else f"{r['val_PR_AUC_cv_media']:.3f} ± {r['val_PR_AUC_cv_std']:.3f}"
        md.append(f"| {r['modelo']} | {cv} | {r['test_PR_AUC']:.3f} | {r['test_precision@K']:.3f} | {r['test_recall@K']:.3f} |")
    md += ["", "## Perfiles de riesgo y acción posible", "",
           "| Perfil | Jugadores | Churn real | Prioridad | Rasgos diferenciales | Acción posible (hipótesis) |",
           "| --- | --- | --- | --- | --- | --- |"]
    for _, r in interp_res["matriz"].iterrows():
        md.append(f"| {r['perfil']} | {r['jugadores_prueba']:,} | {r['churn_real']:.1%} | {r['prioridad']} | "
                  f"{r['rasgos_diferenciales']} | {r['accion_posible']} |")
    create_markdown_artifact(key="gamepulse-reporte", markdown="\n".join(md),
                             description="Resumen de la corrida del pipeline de churn")
    return ficha


# ============================================================================ hooks de estado
def al_fallar(flow, flow_run_, state):
    """Deja constancia del fallo también fuera de Prefect (historial local)."""
    try:
        salida = _ruta(flow_run_.parameters.get("salida", "runs"))
        salida.mkdir(parents=True, exist_ok=True)
        with open(salida / "fallos.log", "a", encoding="utf-8") as f:
            f.write(f"{pd.Timestamp.now().isoformat(timespec='seconds')} | {flow_run_.name} | {state.name} | {state.message}\n")
    except Exception:
        traceback.print_exc()


# ============================================================================ flujo
@flow(name="gamepulse-churn", version=gamepulse_ml.__version__, log_prints=True, on_failure=[al_fallar],
      on_crashed=[al_fallar],
      description="Pipeline completo: validación de datos, auditoría, etiqueta, modelos con validación temporal, "
                  "interpretación, recomendaciones y registro.")
def pipeline_churn(
    src: str = "data/plata_wowah.parquet",
    salida: str = "runs",
    horizonte_dias: int = 30,
    eleg_dias: int = 7,
    min_dias_hist: int = 2,
    umbral_cobertura: float = 0.20,
    n_folds: int = 3,
    familias: list[str] = FAMILIAS,
    k_frac: float = 0.10,
    k_segmentos: int = 5,
    ejecutar_auditoria: bool = True,
    ejecutar_ablacion: bool = True,
    reusar_dataset: bool = True,
    exigir_mejora_sobre_base: bool = True,
    oro: str | None = None,
) -> dict:
    log = get_run_logger()
    src_p, salida_p = _ruta(src), _ruta(salida)
    salida_p.mkdir(parents=True, exist_ok=True)
    carpeta = reg.carpeta_corrida(salida_p, flow_run.name or "local")
    log.info("Corrida %s -> %s", flow_run.name, carpeta)
    c_et = ConfigEtiqueta(horizonte_dias=horizonte_dias, eleg_dias=eleg_dias, min_dias_hist=min_dias_hist,
                          umbral_cobertura=umbral_cobertura)
    c_mod = ConfigModelado(n_folds=n_folds, k_frac=k_frac, k_segmentos=k_segmentos, familias=list(familias))
    params = a_dict(c_et, c_mod) | {"src": str(src_p)}
    reg.guardar_json(params, carpeta / "parametros.json")

    datos = validar_datos(str(src_p))
    aud_dir = salida_p / "auditoria"
    if ejecutar_auditoria:
        tarea_auditoria(str(src_p), str(aud_dir), umbral_cobertura)
    # Con `oro`, la tabla de modelado viene de Spark; la plata sigue haciendo falta para la auditoría y el embudo.
    dset = (leer_oro(str(_ruta(oro)), str(salida_p / "cache")) if oro else
            construir_dataset(str(src_p), str(salida_p / "cache"), c_et.__dict__, datos["sha256"], reusar_dataset))
    cfg_m = c_mod.__dict__
    tarea_particion(dset["archivo"], cfg_m)
    base = tarea_linea_base(dset["archivo"], cfg_m)
    resultados = [entrenar(f, dset["archivo"], cfg_m) for f in familias]
    ens = evaluar_ensamble(resultados, dset["archivo"], cfg_m) if len(resultados) > 1 else None
    mejor = seleccionar(resultados, base, exigir_mejora_sobre_base, ens)
    if "LightGBM" not in [r["familia"] for r in resultados]:
        resultados.append(entrenar("LightGBM", dset["archivo"], cfg_m))      # necesario para SHAP
    interp_res = interpretar(dset["archivo"], cfg_m, resultados, mejor, str(carpeta))
    pob_res = analizar_poblacion(str(src_p), dset["archivo"], c_et.__dict__, cfg_m, resultados, mejor, str(carpeta))
    abl = ablacion(dset["archivo"], cfg_m, mejor, str(carpeta)) if ejecutar_ablacion else None
    uni = variables_individuales(dset["archivo"], cfg_m, str(carpeta))
    data = mod.cargar_dataset(dset["archivo"])
    por_corte = mod.desempeno_por_corte(mod.particion(data, c_mod), mejor, base, k_frac)
    figuras(str(carpeta), str(aud_dir), dset["archivo"], cfg_m, base, resultados, interp_res)
    ficha = registrar(str(carpeta), str(salida_p), params, datos, dset, base, resultados, mejor, abl,
                      interp_res, por_corte, ens, uni)
    log.info("Listo. Modelo %s, PR-AUC prueba %.3f. Resultados en %s", mejor["familia"],
             mejor["metricas_test"]["PR_AUC"], carpeta)
    return {"carpeta": str(carpeta), "modelo": ficha["modelo_elegido"], "metricas_prueba": ficha["metricas_prueba"]}


def _cli():
    ap = argparse.ArgumentParser(description="Ejecuta el flujo gamepulse-churn")
    ap.add_argument("--src", default="data/plata_wowah.parquet")
    ap.add_argument("--salida", default="runs")
    ap.add_argument("--horizonte-dias", type=int, default=30)
    ap.add_argument("--eleg-dias", type=int, default=7)
    ap.add_argument("--n-folds", type=int, default=3)
    ap.add_argument("--familias", nargs="+", default=FAMILIAS)
    ap.add_argument("--sin-auditoria", action="store_true")
    ap.add_argument("--sin-ablacion", action="store_true")
    ap.add_argument("--reconstruir-dataset", action="store_true")
    ap.add_argument("--oro", help="carpeta con la copia local del oro de S3 (oro/dataset_modelado/)")
    a = ap.parse_args()
    pipeline_churn(src=a.src, salida=a.salida, horizonte_dias=a.horizonte_dias, eleg_dias=a.eleg_dias,
                   n_folds=a.n_folds, familias=a.familias, ejecutar_auditoria=not a.sin_auditoria,
                   ejecutar_ablacion=not a.sin_ablacion, reusar_dataset=not a.reconstruir_dataset, oro=a.oro)


if __name__ == "__main__":
    _cli()
