#!/usr/bin/env bash
# Elimina los recursos de GamePulse (Workflow, Jobs, API, Lambda, capas, tablas, rol, secreto y bucket).
# DESTRUCTIVO: borra todos los datos del bucket. Pide confirmacion.
source "$(dirname "$0")/config.sh"
read -r -p "Se borrara TODO el proyecto en AWS (bucket ${BUCKET} incluido). Escribe BORRAR para continuar: " R
[ "$R" = "BORRAR" ] || { echo "Cancelado"; exit 1; }
for t in gp-inicio-semanal gp-plata-tras-ingesta gp-oro-tras-plata; do aws glue delete-trigger --name $t 2>/dev/null; done
aws glue delete-workflow --name gamepulse-semanal 2>/dev/null
for j in gamepulse-ingesta-bronce gamepulse-bronce-a-plata gamepulse-construir-oro; do aws glue delete-job --job-name $j 2>/dev/null; done
for t in plata_wowah rechazadas_wowah dataset_modelado scores perfiles_segmento shap_global; do aws glue delete-table --database-name gamepulse --name $t 2>/dev/null; done
aws glue delete-database --name gamepulse 2>/dev/null
API_ID=$(aws apigatewayv2 get-apis --query "Items[?Name=='gamepulse-api'].ApiId | [0]" --output text)
[ "$API_ID" != "None" ] && aws apigatewayv2 delete-api --api-id "$API_ID"
aws lambda delete-function --function-name gamepulse-api 2>/dev/null
for v in $(aws lambda list-layer-versions --layer-name gamepulse-lightgbm --query 'LayerVersions[].Version' --output text 2>/dev/null); do
  aws lambda delete-layer-version --layer-name gamepulse-lightgbm --version-number $v; done
for r in gamepulse-lambda-role gamepulse-glue-role; do
  for p in $(aws iam list-attached-role-policies --role-name $r --query 'AttachedPolicies[].PolicyArn' --output text 2>/dev/null); do aws iam detach-role-policy --role-name $r --policy-arn $p; done
  for p in $(aws iam list-role-policies --role-name $r --query 'PolicyNames[]' --output text 2>/dev/null); do aws iam delete-role-policy --role-name $r --policy-name $p; done
  aws iam delete-role --role-name $r 2>/dev/null
done
aws secretsmanager delete-secret --secret-id "$SECRETO" --force-delete-without-recovery 2>/dev/null
aws s3 rb "s3://${BUCKET}" --force
rm -f ~/.gamepulse_api_key ~/.gamepulse_api_url
echo "Listo."
