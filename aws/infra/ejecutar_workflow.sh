#!/usr/bin/env bash
# Lanza el Workflow a demanda y muestra el estado de cada Job hasta que termina.
source "$(dirname "$0")/config.sh"
W=gamepulse-semanal
RUN=$(aws glue start-workflow-run --name $W --query RunId --output text) || exit 1
echo "Workflow: $W  Run: $RUN"
while true; do
  S=$(aws glue get-workflow-run --name $W --run-id "$RUN" --query Run.Status --output text)
  echo "  $(date +%H:%M:%S) $S"
  [[ "$S" == "COMPLETED" || "$S" == "STOPPED" || "$S" == "ERROR" ]] && break
  sleep 30
done
aws glue get-workflow-run --name $W --run-id "$RUN" --include-graph \
  --query 'Run.Graph.Nodes[?Type==`JOB`].[Name,JobDetails.JobRuns[0].JobRunState,JobDetails.JobRuns[0].ExecutionTime]' --output table
aws glue get-workflow-run --name $W --run-id "$RUN" --query 'Run.Statistics' --output table
