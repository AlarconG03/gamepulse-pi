#!/usr/bin/env bash
# Despliega la API de scoring: capa de dependencias (LightGBM+numpy+scipy), rol, funcion Lambda y HTTP API Gateway.
# Idempotente. Genera una clave de acceso y la guarda en ~/.gamepulse_api_key (no se imprime en el repositorio).
D="$(dirname "$0")"; source "$D/config.sh"
FN=gamepulse-api; ROL_L=gamepulse-lambda-role; CAPA=gamepulse-lightgbm
TMP=$(mktemp -d)

echo "[1/5] Capa de dependencias"
# LightGBM 4.7.0 (la del entrenamiento) sólo trae ruedas manylinux_2_28, compatibles con el runtime python3.12 de Lambda.
pip install -q --target "$TMP/capa/python" --platform manylinux2014_x86_64 --platform manylinux_2_28_x86_64 \
  --python-version 3.12 --implementation cp --only-binary=:all: lightgbm==4.7.0 numpy scipy narwhals > "$TMP/pip.log" 2>&1 \
  || { tail -5 "$TMP/pip.log"; echo "Falló la instalación de la capa: no se publica nada"; exit 1; }
find "$TMP/capa" -name "__pycache__" -type d -prune -exec rm -rf {} + ; find "$TMP/capa/python/scipy" -type d \( -name tests -o -name test \) -prune -exec rm -rf {} +   # NumPy necesita sus tests (los importa)
# LightGBM necesita libgomp (OpenMP), que la imagen de Lambda no incluye: se copia desde CloudShell a lib/ de la capa.
mkdir -p "$TMP/capa/lib"
LG=$(ls /usr/lib64/libgomp.so.1* /usr/lib/x86_64-linux-gnu/libgomp.so.1* 2>/dev/null | head -1)
if [ -z "$LG" ]; then sudo dnf install -y -q libgomp >/dev/null 2>&1; LG=$(ls /usr/lib64/libgomp.so.1* 2>/dev/null | head -1); fi
[ -n "$LG" ] || { echo "No se encontro libgomp.so.1"; exit 1; }
cp -L "$LG" "$TMP/capa/lib/libgomp.so.1"; echo "  libgomp incluida: $LG"
(cd "$TMP/capa" && zip -qr "$TMP/capa.zip" python lib)
aws s3 cp "$TMP/capa.zip" "s3://${BUCKET}/scripts/capa_lightgbm.zip" --only-show-errors
CAPA_ARN=$(aws lambda publish-layer-version --layer-name $CAPA --content S3Bucket=${BUCKET},S3Key=scripts/capa_lightgbm.zip \
  --compatible-runtimes python3.12 --query LayerVersionArn --output text)
echo "  $CAPA_ARN"

echo "[2/5] Rol de la Lambda"
if ! aws iam get-role --role-name $ROL_L >/dev/null 2>&1; then
  aws iam create-role --role-name $ROL_L --assume-role-policy-document \
   '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name $ROL_L --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  aws iam put-role-policy --role-name $ROL_L --policy-name leer-modelo --policy-document \
   "{\"Version\":\"2012-10-17\",\"Statement\":[{\"Effect\":\"Allow\",\"Action\":\"s3:GetObject\",\"Resource\":\"arn:aws:s3:::${BUCKET}/modelos/*\"}]}"
  sleep 12
fi
ROL_ARN=$(aws iam get-role --role-name $ROL_L --query Role.Arn --output text)

echo "[3/5] Funcion Lambda"
[ -f ~/.gamepulse_api_key ] || openssl rand -hex 16 > ~/.gamepulse_api_key
CLAVE=$(cat ~/.gamepulse_api_key)
(cd "$D/../api" && zip -q "$TMP/fn.zip" handler.py)
# Versión vigente del modelo (publicar_modelo.sh la registra); v1 si nunca se publicó otra.
VERSION=$(aws s3 cp "s3://${BUCKET}/modelos/actual.json" - 2>/dev/null | python3 -c "import json,sys; print(json.load(sys.stdin)['version'])" 2>/dev/null || echo v1)
ENV="Variables={MODEL_BUCKET=${BUCKET},MODEL_PREFIX=modelos/${VERSION},API_KEY=${CLAVE}}"
if aws lambda get-function --function-name $FN >/dev/null 2>&1; then
  aws lambda update-function-code --function-name $FN --zip-file fileb://$TMP/fn.zip >/dev/null
  aws lambda wait function-updated --function-name $FN
  aws lambda update-function-configuration --function-name $FN --layers "$CAPA_ARN" --environment "$ENV" >/dev/null
else
  aws lambda create-function --function-name $FN --runtime python3.12 --handler handler.lambda_handler --role "$ROL_ARN" \
    --zip-file fileb://$TMP/fn.zip --timeout 30 --memory-size 512 --layers "$CAPA_ARN" --environment "$ENV" >/dev/null
fi
aws lambda wait function-active --function-name $FN

echo "[4/5] API Gateway (HTTP API)"
API_ID=$(aws apigatewayv2 get-apis --query "Items[?Name=='gamepulse-api'].ApiId | [0]" --output text)
if [ "$API_ID" = "None" ] || [ -z "$API_ID" ]; then
  FN_ARN=$(aws lambda get-function --function-name $FN --query Configuration.FunctionArn --output text)
  API_ID=$(aws apigatewayv2 create-api --name gamepulse-api --protocol-type HTTP --target "$FN_ARN" --query ApiId --output text)
  aws lambda add-permission --function-name $FN --statement-id apigw --action lambda:InvokeFunction \
    --principal apigateway.amazonaws.com --source-arn "arn:aws:execute-api:${AWS_DEFAULT_REGION}:${CUENTA}:${API_ID}/*" >/dev/null
fi
URL="https://${API_ID}.execute-api.${AWS_DEFAULT_REGION}.amazonaws.com"
echo "$URL" > ~/.gamepulse_api_url
echo "[5/5] Listo"
echo "  URL: $URL   (guardada en ~/.gamepulse_api_url; la clave, en ~/.gamepulse_api_key)"
rm -rf "$TMP"
