#!/usr/bin/env bash
# Tabla con las ultimas corridas del Job construir_oro (workers, % de datos, segundos y DPU-segundos).
source "$(dirname "$0")/config.sh"
aws glue get-job-runs --job-name gamepulse-construir-oro --max-results 12 --output json | python3 -c "
import json,sys
r=json.load(sys.stdin)['JobRuns']
print(f\"{'workers':>8} {'datos %':>8} {'estado':>10} {'segundos':>9} {'DPU-s':>7}\")
for x in sorted(r,key=lambda x:x['StartedOn']):
    a=x.get('Arguments',{})
    print(f\"{x.get('NumberOfWorkers'):>8} {a.get('--porcentaje','100'):>8} {x['JobRunState']:>10} {x.get('ExecutionTime',0):>9} {x.get('DPUSeconds',0):>7}\")
"
