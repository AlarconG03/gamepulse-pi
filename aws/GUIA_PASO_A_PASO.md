# Guía paso a paso: GamePulse en AWS

Se ejecuta en **AWS CloudShell** (región us-east-1) con una cuenta personal. Costo de uso de todo el recorrido, incluidas las pruebas de escalabilidad (paso 6): US$ 1,21 en Glue según AWS Cost Explorer (filtro `RECORD_TYPE = Usage`); en nuestra cuenta lo cubrieron los créditos. Antes de empezar conviene activar una alerta de presupuesto en AWS Budgets. Todos los scripts son idempotentes: se pueden repetir.

## 0. Requisitos
- Cuenta de AWS y un token de la API de Kaggle (`KGAT_...`).
- Comprimir la carpeta `aws/` del repositorio como `aws.zip` (el ZIP debe contener la carpeta `aws/`), subirlo a CloudShell con Actions → Upload file y ejecutar `unzip -o aws.zip` en el directorio de inicio, de modo que exista `~/aws/infra/`. Todos los comandos de esta guía se ejecutan desde `~`.
- El nombre del bucket está en `aws/infra/config.sh` (`gamepulse-pi-2026`); si estuviera ocupado, cambiarlo ahí.

## 1. Infraestructura base y credenciales
```bash
bash aws/infra/paso1_base.sh          # bucket por zonas, cifrado, rol IAM de mínimo privilegio, base de datos del catálogo
read -s TOKEN                         # pegar el token con clic derecho (no se muestra)
aws secretsmanager create-secret --name gamepulse/kaggle --secret-string "{\"token\":\"$TOKEN\"}"; unset TOKEN
```

## 2. Ingesta (Job 1) y prueba de errores
```bash
bash aws/infra/paso2_ingesta.sh       # sube los scripts, crea los tres Jobs y ejecuta la ingesta
bash aws/infra/prueba_error_ingesta.sh  # debe terminar en FAILED y dejar el archivo en landing/errores/
```
Resultado esperado: bronce con 10.826.734 filas y un registro JSON en `bronce/_ingesta_log/`.

## 3. Plata y oro (Jobs 2 y 3) y consulta
```bash
bash aws/infra/ejecutar_job.sh gamepulse-bronce-a-plata
bash aws/infra/ejecutar_job.sh gamepulse-construir-oro
bash aws/infra/paso3_athena.sh        # tablas externas en Athena
bash aws/infra/athena.sh "SELECT count(*) AS filas, count(distinct char_id) AS personajes FROM plata_wowah"
bash aws/infra/athena.sh "SELECT count(*) AS filas, count(distinct char_id) AS personajes, count(distinct fecha_corte) AS cortes, round(avg(churn),4) AS churn FROM dataset_modelado"
```
Esperado: plata 10.823.177 filas y 37.354 personajes (cuadre en `plata/_cuadre/`); oro 156.877 filas, 18.440 personajes, 41 cortes y 0,1408 de abandono.

## 4. Orquestación semanal
```bash
bash aws/infra/paso4_workflow.sh      # Workflow con disparo programado (lunes 03:00, hora de Bogotá)
bash aws/infra/ejecutar_workflow.sh   # ejecución completa a demanda
bash aws/infra/pausar_workflow.sh     # pausa el disparo semanal (con "activar" se reanuda)
```

## 5. Modelo, resultados y API
En el equipo local (con el repositorio y los resultados de referencia), se generan los artefactos en `aws/publicar/` (ya incluidos en el repositorio):
```bash
python aws/api/preparar_publicacion.py
```
En CloudShell:
```bash
bash aws/infra/paso5_publicar.sh      # modelo en S3, tablas scores / perfiles_segmento / shap_global en Athena
bash aws/infra/paso6_api.sh           # capa de dependencias, Lambda y API Gateway; guarda la URL y la clave en ~/.gamepulse_api_*
bash aws/infra/prueba_api.sh          # /health, /predict, solicitud incompleta (400) y sin clave (401)
```

### 5b. Reentrenar con el oro y publicar una versión nueva
En el equipo local, entrenar con la tabla que construyó Spark (la plata sigue haciendo falta para la auditoría y
el embudo de población):
```bash
aws s3 sync s3://<bucket>/oro/dataset_modelado/ data/oro/     # o descargarla desde CloudShell
python flows/pipeline_flow.py --oro data/oro
python aws/api/exportar_modelo.py runs/<corrida>/modelos/lightgbm.joblib aws/publicar/modelos/v2
```
En CloudShell, después de volver a subir `aws.zip` (ya con `aws/publicar/modelos/v2/`) y descomprimirlo:
```bash
bash aws/infra/publicar_modelo.sh v2   # sube modelos/v2/, apunta la Lambda a esa versión, registra modelos/actual.json y prueba la API
```
Para volver atrás: `bash aws/infra/publicar_modelo.sh v1`. `paso6_api.sh` respeta la versión registrada.

## 6. Escalabilidad (opcional)
```bash
nohup bash aws/infra/escalabilidad.sh > escalabilidad.log 2>&1 &
bash aws/infra/escalabilidad_resumen.sh
```

## 7. Limpieza (evita costos al terminar)
```bash
bash aws/infra/limpiar.sh
```

## Si algo falla
- Un Job de Glue: `aws logs tail /aws-glue/jobs/error --since 15m`.
- La API (HTTP 500): `aws logs tail /aws/lambda/gamepulse-api --since 15m`.
- Problemas conocidos al construir la capa de la API: la imagen de Lambda no incluye `libgomp` (el script la copia desde CloudShell) y NumPy necesita sus carpetas `tests`.
