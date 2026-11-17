# GamePulse · Predicción de abandono de jugadores

Proyecto Integrador 1 (2026-2) · Ingeniería de Sistemas, Universidad EAFIT.
Juan Alberto Rodríguez, Alejandro Zapata, Vladlen Shatunov, Samuel Alarcón.

**Pregunta de negocio:** ¿hasta qué punto los patrones temporales de actividad y progresión permiten anticipar
la inactividad prolongada de un jugador y orientar posibles acciones de retención?

Este repositorio contiene todo el proyecto:

- **Modelado** (`src/`, `flows/`): auditoría de los datos, etiqueta de inactividad, variables sin fuga temporal,
  modelos con validación temporal, interpretación y matriz de recomendaciones, orquestado con **Prefect**.
- **Ingeniería de datos en AWS** (`aws/`): ingesta desde Kaggle, lago por zonas en S3, Spark en Glue,
  Workflow semanal, Athena y una API de scoring (Lambda + API Gateway).
- **Tablero** (`tablero/`): Streamlit, lee la instantánea exportada de Athena y pide el detalle de cada caso a la API.
- **Documento de entrega** (`docs/documento/`) y los scripts que justifican sus cifras (`analisis_adicional/`).

Tablero publicado: <https://gamepulse-pi.streamlit.app> · Correrlo en local o publicarlo: [docs/DESPLIEGUE_STREAMLIT.md](docs/DESPLIEGUE_STREAMLIT.md).

## Resultado principal (corrida de referencia)

| Modelo | PR-AUC prueba | Precisión en el 10 % de mayor riesgo | Recall en ese 10 % |
|---|---|---|---|
| Azar (tasa de churn en prueba) | 0,196 | 0,196 | 0,10 |
| Regla de recencia (línea base) | 0,450 | 0,504 | 0,258 |
| Regresión logística | 0,585 | 0,680 | 0,350 |
| Random Forest | 0,625 | 0,727 | 0,373 |
| **LightGBM (elegido por validación)** | **0,618** | **0,722** | **0,371** |
| Ensamble (promedio de rangos) | 0,623 | 0,719 | 0,370 |

Prueba = cortes del 27/10 al 01/12/2008, nunca vistos al entrenar ni al elegir el modelo.
Detalle en [`resultados/referencia/`](resultados/referencia/).

**Para entender todo el proceso** (población objetivo, partición, modelos, resultados, qué mejorar y qué falta):
**[docs/proceso_completo.md](docs/proceso_completo.md)**.

## Inicio rápido

```bash
git clone https://github.com/AlarconG03/gamepulse-pi.git && cd gamepulse-pi
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                   # instala el paquete y sus dependencias
# la capa plata ya viene en data/plata_wowah.parquet (ver data/README.md)
pytest -q                                                 # 13 pruebas, ~10 s, no necesitan los datos

prefect server start                                      # terminal 1: dashboard en http://127.0.0.1:4200
prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api   # terminal 2, una sola vez
python flows/pipeline_flow.py                             # corrida completa (~10–15 min)
```

Guía completa, paso a paso, con capturas: **[docs/prefect.md](docs/prefect.md)**.

## Estructura

```
gamepulse-pi/
├── README.md                  # este archivo
├── pyproject.toml             # paquete instalable (pip install -e .) y configuración de pytest
├── requirements.txt           # dependencias (alternativa a pyproject)
├── requirements-lock.txt      # versiones exactas de la corrida de referencia
├── packages.txt               # librería del sistema para LightGBM en Streamlit Cloud
├── LICENSE                    # MIT (código); los datos de WoWAH son CC0 según Kaggle
├── .gitignore                 # excluye corridas, cachés, secretos y ZIPs (la plata sí se versiona)
│
├── data/                      # capa plata (plata_wowah.parquet) y de dónde sale
│   └── README.md              # origen en Kaggle (wowah_data.csv, zones.csv) y cómo verificar la plata
│
├── src/gamepulse_ml/          # lógica del pipeline (independiente de Prefect)
│   ├── config.py              # parámetros por defecto
│   ├── validacion.py          # controles de calidad + SHA-256 de los datos
│   ├── auditoria.py           # auditoría de cobertura temporal
│   ├── dataset.py             # etiqueta + 37 variables por personaje y fecha de corte
│   ├── zonas.py               # clasificación de zonas (ciudad, PvP, raid, mazmorra, mundo)
│   ├── modelos.py             # partición con embargo, origen móvil, modelos, ensamble, métricas
│   ├── interpretacion.py      # SHAP, K-Means, ablación, ranking de variables
│   ├── poblacion.py           # embudo de población, panel, subpoblaciones, modelos especializados
│   ├── recomendaciones.py     # matriz perfil -> señal -> acción posible
│   ├── figuras.py             # gráficas del informe
│   └── registro.py            # trazabilidad local (versiones, historial)
│
├── flows/                     # orquestación con Prefect
│   ├── pipeline_flow.py       # flujo principal gamepulse-churn (16 tareas)
│   ├── sensibilidad_flow.py   # sensibilidad de la etiqueta (21/30/45/60 días) y de la población
│   └── desplegar.py           # deployments: reentrenamiento semanal programado
│
├── tests/                     # 13 pruebas con datos sintéticos (etiqueta, fuga, población, calidad de datos,
│                              # curvas ROC, ranking de variables y ablación)
│
├── docs/
│   ├── prefect.md             # cómo ejecutar, leer el dashboard y desplegar
│   ├── proceso_completo.md    # TODO el proceso: población, partición, modelos, resultados, mejoras, faltantes
│   ├── DESPLIEGUE_STREAMLIT.md # correr el tablero en local o publicarlo en Streamlit Cloud
│   ├── documento/             # documento de entrega (Word y PDF) y sus 13 figuras
│   └── img/                   # capturas del dashboard de Prefect
│
├── analisis_adicional/        # auditoría de la plata (3.2), IC y lift, sesgo/varianza, valor de negocio, diagrama
│                              # (sin reentrenar); qué figura sale de cada script: analisis_adicional/README.md
├── aws/                       # ingeniería de datos en AWS (sección 5 del documento): ver aws/GUIA_PASO_A_PASO.md
│   ├── jobs/                  # Glue: ingesta_bronce (Python Shell), bronce_a_plata y construir_oro (PySpark)
│   ├── infra/                 # scripts de CloudShell: bucket, IAM, Jobs, Workflow semanal, Athena, API, limpieza
│   ├── api/                   # Lambda de scoring (handler.py) y exportación del modelo a modelo.txt + meta.json
│   ├── publicar/              # artefactos publicados en S3 (modelo v1, scores, perfiles, SHAP global)
│   └── pruebas/               # comparación oro (Spark) vs. dataset (pandas)
├── tablero/                   # tablero Streamlit (sección 6): ver tablero/README.md
│   └── datos/                 # instantánea exportada de Athena (versionada, ~4 MB) que lee el tablero
├── resultados/adicional/      # salidas de analisis_adicional (ver su README)
├── resultados/referencia/     # salidas de la corrida de referencia (versionadas, ~2,5 MB)
│   ├── resultados_modelos.csv, desempeno_por_corte_prueba.csv, ficha_corrida.json
│   ├── matriz_recomendaciones.csv, scores_prueba.parquet   # entrada para visualización
│   ├── ablacion_controlada.csv, variables_individuales_cv.csv
│   ├── sensibilidad_etiqueta.csv, auditoria/, figuras/ (fig1–fig9), modelo/lightgbm.joblib
│   └── README.md
│
└── runs/                      # se crea al ejecutar (NO se versiona): una carpeta por corrida
```

## Dos orquestadores, dos pipelines

- **Datos (AWS Glue Workflow `gamepulse-semanal`):** ingesta desde Kaggle, bronce, plata y oro, cada lunes.
- **Modelo (Prefect, en local):** valida, construye la tabla, entrena, selecciona con validación temporal,
  interpreta y registra. `flows/desplegar.py` deja un reentrenamiento semanal programado en el equipo donde
  corre. Con `--oro data/oro` entrena con la tabla que construyó Spark en AWS en lugar de rehacerla con pandas.
- **Publicación:** el modelo `v1` es el de la corrida de referencia; `aws/infra/publicar_modelo.sh vN` sube una
  versión nueva, apunta la API a ella y la registra en `s3://<bucket>/modelos/actual.json`.

Versiones exactas de la corrida de referencia: `requirements-lock.txt`.

## Datos y secretos

- La capa plata está en `data/`. El archivo original (`wowah_data.csv`, 643 MB) y `zones.csv` no se versionan: se descargan de Kaggle (ver `data/README.md`).
- Los datos de WoWAH (O'Neill en Kaggle; Lee et al., 2011) tienen licencia CC0: Public Domain según Kaggle y no
  están bajo la licencia MIT del código.
- Ninguna credencial está en el repositorio. El token de Kaggle vive en AWS Secrets Manager; la URL y la clave
  de la API van en `tablero/.streamlit/secrets.toml` (ignorado por git) o en los secretos de Streamlit Cloud.
