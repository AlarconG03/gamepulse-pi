#!/usr/bin/env bash
# Publica el modelo y las tablas de resultados en S3 y los registra en Athena.
D="$(dirname "$0")"; source "$D/config.sh"
P="$D/../publicar"
aws s3 cp "$P/modelos/v1/"   "s3://${BUCKET}/modelos/v1/"   --recursive
aws s3 cp "$P/scores/"       "s3://${BUCKET}/oro/scores/"    --recursive
aws s3 cp "$P/perfiles/"     "s3://${BUCKET}/oro/perfiles/"  --recursive
aws s3 cp "$P/shap_global/"  "s3://${BUCKET}/oro/shap_global/" --recursive
bash "$D/paso3_athena.sh" || exit 1
bash "$D/athena.sh" "SELECT count(*) AS filas, round(avg(riesgo_lightgbm),4) AS riesgo_medio, round(avg(churn),4) AS churn FROM scores"
