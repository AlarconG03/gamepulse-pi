#!/usr/bin/env bash
# Prueba de manejo de errores: exige un minimo imposible de filas, de modo que la validacion falle.
# Esperado: el Job termina en FAILED, el archivo queda en landing/errores/ y el registro explica el motivo.
set -uo pipefail
source "$(dirname "$0")/config.sh"
bash "$(dirname "$0")/ejecutar_job.sh" gamepulse-ingesta-bronce '{"--usar_landing":"true","--min_filas":"999999999"}'
echo "--- Cuarentena:"; aws s3 ls "s3://${BUCKET}/landing/errores/" --recursive --human-readable
ULTIMO=$(aws s3api list-objects-v2 --bucket "$BUCKET" --prefix bronce/_ingesta_log/ --query 'sort_by(Contents,&LastModified)[-1].Key' --output text)
aws s3 cp "s3://${BUCKET}/${ULTIMO}" - | grep -E '"estado"|"errores"|solo' 
