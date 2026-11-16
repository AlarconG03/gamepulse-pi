#!/usr/bin/env bash
# Exporta desde Athena la instantánea que lee el tablero (sección 6) y la deja en ~/tablero_datos.zip.
# Uso en CloudShell:  bash aws/infra/exportar_tablero.sh   y luego Actions > Download file > tablero_datos.zip
# En el equipo local: descomprimir en tablero/datos/ del repositorio  (quedan scores/, perfiles/, shap_global/,
# casos/ y modelo/).  Sólo lectura sobre las tablas; escribe en s3://BUCKET/tablero/ (se sobrescribe).
D="$(dirname "$0")"
source "$D/config.sh"
DESTINO=s3://${BUCKET}/tablero
PRIMER_CORTE=$(bash "$D/athena.sh" "SELECT cast(min(fecha_corte) AS date) FROM scores" | head -1 | awk '{print $NF}')
echo "Primer corte de prueba: ${PRIMER_CORTE}"
[[ -n "$PRIMER_CORTE" ]] || { echo "No se pudo leer la tabla scores"; exit 1; }

declare -A CONSULTAS=(
  [scores]="SELECT * FROM scores"
  [perfiles]="SELECT * FROM perfiles_segmento"
  [shap_global]="SELECT * FROM shap_global"
  [casos]="SELECT * FROM dataset_modelado WHERE fecha_corte >= timestamp '${PRIMER_CORTE} 00:00:00'"
)
for T in scores perfiles shap_global casos; do
  aws s3 rm --recursive --quiet "${DESTINO}/${T}/"          # UNLOAD exige carpeta vacía
  echo ">> UNLOAD ${T}"
  bash "$D/athena.sh" "UNLOAD (${CONSULTAS[$T]}) TO '${DESTINO}/${T}/' WITH (format = 'PARQUET')" > /dev/null \
    || { echo "FALLO en ${T}"; exit 1; }
done

rm -rf ~/tablero_datos ~/tablero_datos.zip
aws s3 sync --quiet "${DESTINO}/" ~/tablero_datos/
aws s3 cp --quiet "s3://${BUCKET}/modelos/v1/modelo.txt" ~/tablero_datos/modelo/modelo.txt
aws s3 cp --quiet "s3://${BUCKET}/modelos/v1/meta.json" ~/tablero_datos/modelo/meta.json
# Athena nombra los archivos sin extensión: se la agregamos para que pandas los reconozca.
find ~/tablero_datos -type f ! -name '*.*' -exec mv {} {}.parquet \;
(cd ~/tablero_datos && zip -qr ~/tablero_datos.zip .)
echo "Archivos exportados:"
find ~/tablero_datos -type f | sed "s|$HOME/tablero_datos/||" | sort | awk -F/ '{n[$1]++} END {for (k in n) print "  " k ": " n[k] " archivo(s)"}'
ls -lh ~/tablero_datos.zip
