#!/usr/bin/env bash
# Mide la escalabilidad del Job construir_oro: 2/5/10 workers con 100 % de los datos y 25/50/100 % con 5 workers.
# Cada corrida escribe en oro_pruebas/<combinacion>/ (el oro real no se toca). Es secuencial; se puede lanzar con nohup.
# Uso: bash aws/infra/escalabilidad.sh [workers:porcentaje ...]  (sin argumentos, las 5 combinaciones)
# Resumen: bash aws/infra/escalabilidad_resumen.sh
source "$(dirname "$0")/config.sh"
JOB=gamepulse-construir-oro
# formato: workers:porcentaje
for C in ${@:-2:100 10:100 5:25 5:50 5:100}; do
  W=${C%%:*}; P=${C##*:}
  echo "$(date +%H:%M:%S) inicia workers=$W porcentaje=$P"
  RUN=$(aws glue start-job-run --job-name $JOB --worker-type G.1X --number-of-workers "$W" \
    --arguments "{\"--porcentaje\":\"$P\",\"--salida\":\"s3://${BUCKET}/oro_pruebas/w${W}_p${P}/\"}" \
    --query JobRunId --output text) || exit 1
  while true; do
    E=$(aws glue get-job-run --job-name $JOB --run-id "$RUN" --query JobRun.JobRunState --output text)
    case "$E" in SUCCEEDED|FAILED|ERROR|TIMEOUT|STOPPED) break;; esac
    sleep 20
  done
  echo "$(date +%H:%M:%S) termina workers=$W porcentaje=$P -> $E"
  [ "$E" = SUCCEEDED ] || exit 1
  sleep 45   # Glue libera la corrida un poco despues de terminar
done
echo "Listo. Ejecuta: bash aws/infra/escalabilidad_resumen.sh"
