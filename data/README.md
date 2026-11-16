# Datos de entrada

El pipeline de modelado espera **`data/plata_wowah.parquet`**: la capa plata (limpia y validada) del dataset
*World of Warcraft Avatar History* (Lee et al., 2011), publicado en Kaggle por Myles O'Neill:
<https://www.kaggle.com/datasets/mylesoneill/warcraft-avatar-history>.

El archivo está en el repositorio. Los dos archivos originales de Kaggle no se versionan: se descargan de la página del conjunto.

## Archivos de Kaggle que usa el proyecto

| Archivo | Uso |
|---|---|
| `wowah_data.csv` (643 MB, 10.826.734 filas) | Única fuente del modelo: se limpia a la capa plata. |
| `zones.csv` | Sólo en el análisis exploratorio, para la hipótesis H5 (rango de nivel recomendado por zona). **No entra al modelo.** |

El modelo y la capa oro clasifican las zonas (ciudad, PvP, banda, mazmorra, mundo abierto) por su nombre con
una tabla manual propia, `src/gamepulse_ml/zonas.py` (la misma lógica está en `aws/jobs/construir_oro.py`);
no usan `zones.csv`.

## Cómo regenerar la capa plata

**Opción A. En AWS (sección 5 del documento).** El Workflow de Glue descarga `wowah_data.csv` de Kaggle y
escribe la plata en `s3://<bucket>/plata/wowah/` (Parquet particionado por mes); ver `aws/GUIA_PASO_A_PASO.md`.
Para usarla en local, se descarga y se une en un solo archivo:

```bash
aws s3 sync s3://<bucket>/plata/wowah/ plata_s3/
python -c "import pandas as pd; pd.read_parquet('plata_s3').drop(columns='mes', errors='ignore').to_parquet('data/plata_wowah.parquet', index=False)"
```

**Opción B. En local con PySpark** (requiere `pip install pyspark` y Java 17). Descargar `wowah_data.csv` de Kaggle y
correr el mismo Job de plata:

```bash
python aws/jobs/bronce_a_plata.py --entrada <carpeta_con_wowah_data.csv> --plata plata_local/ \
       --rechazadas rechazadas_local/ --resumen cuadre_local/
python -c "import pandas as pd; pd.read_parquet('plata_local').drop(columns='mes', errors='ignore').to_parquet('data/plata_wowah.parquet', index=False)"
```

## Verificación

El flujo valida el archivo antes de usarlo (tarea `validar-datos`). Valores de la plata de la corrida de
referencia:

| Control | Valor |
|---|---|
| SHA-256 (archivo de la corrida de referencia) | `de56570d2136eeac154b8c2c03a2a7528cf19539888ced0fb675707d83367afc` |
| Filas | 10.823.177 |
| Personajes | 37.354 |
| IDs inconsistentes | 819 |
| Rango de fechas | 2008-01-01 00:02:04 a 2008-12-31 23:50:18 |

Columnas: `char_id`, `nivel`, `raza`, `clase`, `zona`, `gremio` (nulo = sin gremio), `ts`, `id_inconsistente`.

Una plata regenerada con la opción A o B tiene los mismos conteos, pero su SHA-256 puede diferir (orden de
filas y metadatos del Parquet). En ese caso el dataset de modelado se reconstruye automáticamente y el hash
nuevo queda registrado en `ficha_corrida.json`. La equivalencia de la tabla de modelado está comprobada:
el oro construido con Spark coincide con la tabla de pandas en todas las columnas (`aws/pruebas/RESULTADO_LOCAL.md`).
