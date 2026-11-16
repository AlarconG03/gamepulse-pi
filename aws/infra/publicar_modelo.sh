#!/usr/bin/env bash
# Publica una versión nueva del modelo y la deja activa en la API.
# Uso: bash aws/infra/publicar_modelo.sh v2
# Antes, en el equipo local, desde una corrida del pipeline:
#   python aws/api/exportar_modelo.py runs/<corrida>/modelos/lightgbm.joblib aws/publicar/modelos/v2
# Sube aws/publicar/modelos/<version>/ a s3://BUCKET/modelos/<version>/, apunta la Lambda a esa carpeta
# (variable MODEL_PREFIX; los contenedores nuevos cargan el modelo nuevo) y registra la versión vigente
# en s3://BUCKET/modelos/actual.json. Las versiones anteriores quedan en S3 para volver atrás.
set -euo pipefail
D="$(dirname "$0")"; source "$D/config.sh"
V="${1:?Indique la versión, por ejemplo v2}"
CARPETA="$D/../publicar/modelos/$V"
[ -f "$CARPETA/modelo.txt" ] && [ -f "$CARPETA/meta.json" ] || { echo "Falta $CARPETA/modelo.txt o meta.json"; exit 1; }
FN=gamepulse-api

aws s3 cp "$CARPETA/" "s3://${BUCKET}/modelos/${V}/" --recursive
# Conserva las demás variables de la Lambda (bucket y clave) y cambia sólo MODEL_PREFIX.
ENV=$(aws lambda get-function-configuration --function-name $FN --query Environment --output json \
      | python3 -c "import json,sys; e=json.load(sys.stdin); e['Variables']['MODEL_PREFIX']='modelos/$V'; print(json.dumps(e))")
aws lambda update-function-configuration --function-name $FN --environment "$ENV" >/dev/null
aws lambda wait function-updated --function-name $FN
printf '{"version": "%s", "publicado": "%s"}\n' "$V" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  | aws s3 cp - "s3://${BUCKET}/modelos/actual.json"
echo "Versión activa: $V"
bash "$D/prueba_api.sh"
