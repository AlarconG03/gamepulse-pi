#!/usr/bin/env bash
# Lanza un Job de Glue, espera a que termine y muestra estado, duracion, DPU-segundos y error (si lo hay).
# Uso: bash aws/infra/ejecutar_job.sh <nombre_job> ['{"--parametro":"valor"}']
set -uo pipefail
JOB="$1"; ARGS="${2:-}"
export AWS_DEFAULT_REGION=us-east-1
if [ -n "$ARGS" ]; then
  RUN=$(aws glue start-job-run --job-name "$JOB" --arguments "$ARGS" --query JobRunId --output text)
else
  RUN=$(aws glue start-job-run --job-name "$JOB" --query JobRunId --output text)
fi
echo "Job: $JOB  Run: $RUN"
while true; do
  ESTADO=$(aws glue get-job-run --job-name "$JOB" --run-id "$RUN" --query JobRun.JobRunState --output text)
  echo "  $(date +%H:%M:%S) $ESTADO"
  case "$ESTADO" in SUCCEEDED|FAILED|ERROR|TIMEOUT|STOPPED) break;; esac
  sleep 20
done
aws glue get-job-run --job-name "$JOB" --run-id "$RUN" \
  --query 'JobRun.{Estado:JobRunState,Segundos:ExecutionTime,DPUsegundos:DPUSeconds,Workers:NumberOfWorkers,TipoWorker:WorkerType,Error:ErrorMessage}' --output table
[ "$ESTADO" = "SUCCEEDED" ]
