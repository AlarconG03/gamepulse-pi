# Configuracion comun de los scripts de AWS (se carga con: source aws/infra/config.sh)
export AWS_DEFAULT_REGION=us-east-1
export BUCKET="gamepulse-pi-2026"          # nombre propio; si estuviera ocupado en AWS, cambiarlo aqui
export ROL="gamepulse-glue-role"
export SECRETO="gamepulse/kaggle"
export CUENTA=$(aws sts get-caller-identity --query Account --output text)
