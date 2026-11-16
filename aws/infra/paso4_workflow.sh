#!/usr/bin/env bash
# Crea el Workflow semanal de Glue: ingesta_bronce -> bronce_a_plata -> construir_oro.
# Disparo programado: lunes 08:00 UTC (= 03:00 hora de Bogota). Cada paso arranca solo si el anterior termina bien.
source "$(dirname "$0")/config.sh"
W=gamepulse-semanal
aws glue delete-trigger --name gp-inicio-semanal 2>/dev/null
aws glue delete-trigger --name gp-plata-tras-ingesta 2>/dev/null
aws glue delete-trigger --name gp-oro-tras-plata 2>/dev/null
aws glue delete-workflow --name $W 2>/dev/null
aws glue create-workflow --name $W --description "GamePulse: ingesta -> plata -> oro (semanal)" >/dev/null
aws glue create-trigger --workflow-name $W --name gp-inicio-semanal --type SCHEDULED \
  --schedule "cron(0 8 ? * MON *)" --actions JobName=gamepulse-ingesta-bronce >/dev/null
aws glue create-trigger --workflow-name $W --name gp-plata-tras-ingesta --type CONDITIONAL \
  --predicate '{"Conditions":[{"LogicalOperator":"EQUALS","JobName":"gamepulse-ingesta-bronce","State":"SUCCEEDED"}]}' \
  --actions JobName=gamepulse-bronce-a-plata --start-on-creation >/dev/null
aws glue create-trigger --workflow-name $W --name gp-oro-tras-plata --type CONDITIONAL \
  --predicate '{"Conditions":[{"LogicalOperator":"EQUALS","JobName":"gamepulse-bronce-a-plata","State":"SUCCEEDED"}]}' \
  --actions JobName=gamepulse-construir-oro --start-on-creation >/dev/null
aws glue start-trigger --name gp-inicio-semanal >/dev/null
echo "Workflow creado. Disparadores:"
aws glue get-triggers --query "Triggers[?WorkflowName=='$W'].[Name,Type,State,Schedule]" --output table
