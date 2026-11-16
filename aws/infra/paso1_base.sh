#!/usr/bin/env bash
# Paso 1: infraestructura base. Se ejecuta en AWS CloudShell (us-east-1). Es idempotente: se puede repetir.
# Crea: bucket con zonas (landing, bronce, plata, rechazadas, oro, modelos, scripts, athena-results),
#       rol IAM de los Jobs de Glue (minimo privilegio) y la base de datos del catalogo de Glue.
set -euo pipefail
source "$(dirname "$0")/config.sh"
echo "Cuenta: ${CUENTA}  Bucket: ${BUCKET}"

# 1) Bucket privado con zonas
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  aws s3api create-bucket --bucket "$BUCKET" || { echo "El nombre ${BUCKET} esta ocupado: cambialo en aws/infra/config.sh"; exit 1; }
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
for z in landing bronce/wowah plata/wowah rechazadas/wowah oro/dataset_modelado oro/scores modelos scripts athena-results; do
  aws s3api put-object --bucket "$BUCKET" --key "${z}/" >/dev/null
done

# 2) Rol de los Jobs de Glue: politica administrada de Glue + acceso solo a este bucket y a un secreto
cat > /tmp/confianza.json <<'J'
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"glue.amazonaws.com"},"Action":"sts:AssumeRole"}]}
J
aws iam get-role --role-name "$ROL" >/dev/null 2>&1 || aws iam create-role --role-name "$ROL" \
  --assume-role-policy-document file:///tmp/confianza.json >/dev/null
aws iam attach-role-policy --role-name "$ROL" --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
cat > /tmp/politica.json <<J
{"Version":"2012-10-17","Statement":[
 {"Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket","s3:GetBucketLocation"],
  "Resource":["arn:aws:s3:::${BUCKET}","arn:aws:s3:::${BUCKET}/*"]},
 {"Effect":"Allow","Action":["secretsmanager:GetSecretValue"],
  "Resource":"arn:aws:secretsmanager:us-east-1:${CUENTA}:secret:${SECRETO}-*"}]}
J
aws iam put-role-policy --role-name "$ROL" --policy-name gamepulse-acceso --policy-document file:///tmp/politica.json

# 3) Base de datos del catalogo (Glue Data Catalog)
aws glue get-database --name gamepulse >/dev/null 2>&1 || aws glue create-database --database-input '{"Name":"gamepulse","Description":"GamePulse: capas plata y oro"}'

echo "OK. Bucket: s3://${BUCKET}  Rol: ${ROL}"
aws s3 ls "s3://${BUCKET}/"
