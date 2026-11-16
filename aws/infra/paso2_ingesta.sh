#!/usr/bin/env bash
# Paso 2: sube los Jobs a S3, los crea en Glue y ejecuta la ingesta (Kaggle -> landing -> validacion -> bronce).
# Idempotente: borra y recrea los Jobs en cada ejecucion.
set -euo pipefail
source "$(dirname "$0")/config.sh"
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
ROL_ARN="arn:aws:iam::${CUENTA}:role/${ROL}"

aws secretsmanager describe-secret --secret-id "$SECRETO" --query Name --output text >/dev/null 2>&1 || {
  echo "Falta el secreto ${SECRETO}: crealo con el comando del Paso 1 (punto 3) y repite."; exit 1; }

# 1) Scripts de los Jobs a S3
aws s3 cp "${RAIZ}/jobs/" "s3://${BUCKET}/scripts/" --recursive --exclude "__pycache__/*" --only-show-errors
aws s3 ls "s3://${BUCKET}/scripts/"

# 2) Jobs de Glue (el de ingesta es Python Shell; los otros dos, Spark con Glue 4.0)
crear() { aws glue delete-job --job-name "$1" >/dev/null 2>&1 || true; aws glue create-job --name "$1" --role "$ROL_ARN" "${@:2}" --query Name --output text; }
crear gamepulse-ingesta-bronce --timeout 60 --max-capacity 1 \
  --command "Name=pythonshell,ScriptLocation=s3://${BUCKET}/scripts/ingesta_bronce.py,PythonVersion=3.9" \
  --default-arguments "{\"--bucket\":\"${BUCKET}\",\"--secreto\":\"${SECRETO}\"}"
crear gamepulse-bronce-a-plata --glue-version 4.0 --worker-type G.1X --number-of-workers 5 --timeout 60 \
  --command "Name=glueetl,ScriptLocation=s3://${BUCKET}/scripts/bronce_a_plata.py,PythonVersion=3" \
  --default-arguments "{\"--entrada\":\"s3://${BUCKET}/bronce/wowah/\",\"--plata\":\"s3://${BUCKET}/plata/wowah/\",\"--rechazadas\":\"s3://${BUCKET}/rechazadas/wowah/\",\"--resumen\":\"s3://${BUCKET}/plata/_cuadre/\"}"
crear gamepulse-construir-oro --glue-version 4.0 --worker-type G.1X --number-of-workers 5 --timeout 60 \
  --command "Name=glueetl,ScriptLocation=s3://${BUCKET}/scripts/construir_oro.py,PythonVersion=3" \
  --default-arguments "{\"--plata\":\"s3://${BUCKET}/plata/wowah/\",\"--salida\":\"s3://${BUCKET}/oro/dataset_modelado/\"}"

# 3) Ejecutar la ingesta
bash "$(dirname "$0")/ejecutar_job.sh" gamepulse-ingesta-bronce
echo "--- Bronce:"; aws s3 ls "s3://${BUCKET}/bronce/wowah/" --recursive --human-readable
echo "--- Ultimo registro de ingesta:"
ULTIMO=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix bronce/_ingesta_log/ --query 'sort_by(Contents,&LastModified)[-1].Key' --output text)
aws s3 cp "s3://${BUCKET}/${ULTIMO}" - | head -60
