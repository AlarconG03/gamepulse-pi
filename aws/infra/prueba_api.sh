#!/usr/bin/env bash
# Prueba funcional de la API: /health, /predict con un jugador real, solicitud incompleta y llamada sin clave.
D="$(dirname "$0")"; URL=$(cat ~/.gamepulse_api_url); CLAVE=$(cat ~/.gamepulse_api_key)
echo "--- GET /health (con clave)";  curl -s -H "x-api-key: $CLAVE" "$URL/health"; echo
echo "--- POST /predict (jugador de ejemplo)"
curl -s -X POST -H "x-api-key: $CLAVE" -H "Content-Type: application/json" --data @"$D/../api/ejemplo_solicitud.json" "$URL/predict" | python3 -m json.tool | head -40
echo "--- POST /predict incompleto (esperado 400)"
curl -s -o /dev/null -w "HTTP %{http_code}\n" -X POST -H "x-api-key: $CLAVE" -d '{"jugadores":[{"nivel":3}]}' "$URL/predict"
echo "--- Sin clave (esperado 401)"
curl -s -o /dev/null -w "HTTP %{http_code}\n" "$URL/health"
