#!/usr/bin/env bash
# Desactiva (o reactiva con "activar") el disparo semanal para no generar costo fuera de las fechas de entrega.
source "$(dirname "$0")/config.sh"
if [ "${1:-}" = "activar" ]; then aws glue start-trigger --name gp-inicio-semanal --query Name --output text
else aws glue stop-trigger --name gp-inicio-semanal --query Name --output text; fi
