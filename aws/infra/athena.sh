#!/bin/bash
# Uso: bash aws/infra/athena.sh "<consulta SQL>"   -> ejecuta en Athena y muestra el resultado como tabla.
source "$(dirname "$0")/config.sh"
SQL="$1"
ID=$(aws athena start-query-execution --query-string "$SQL" \
  --query-execution-context Database=gamepulse \
  --result-configuration OutputLocation=s3://${BUCKET}/athena-results/ \
  --query QueryExecutionId --output text) || exit 1
while :; do
  E=$(aws athena get-query-execution --query-execution-id "$ID" --query 'QueryExecution.Status.State' --output text)
  [[ "$E" == "SUCCEEDED" || "$E" == "FAILED" || "$E" == "CANCELLED" ]] && break
  sleep 2
done
if [[ "$E" != "SUCCEEDED" ]]; then
  aws athena get-query-execution --query-execution-id "$ID" --query 'QueryExecution.Status.StateChangeReason' --output text; exit 1
fi
aws athena get-query-results --query-execution-id "$ID" --output text \
  --query 'ResultSet.Rows[].Data[].VarCharValue' | head -c 4000
echo
aws athena get-query-execution --query-execution-id "$ID" \
  --query 'QueryExecution.Statistics.[EngineExecutionTimeInMillis,DataScannedInBytes]' --output text \
  | awk '{printf "  (motor: %.1f s, datos escaneados: %.1f MB)\n",$1/1000,$2/1048576}'
