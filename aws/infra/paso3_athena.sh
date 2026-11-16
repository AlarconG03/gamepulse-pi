#!/usr/bin/env bash
# Crea (si no existen) las tablas externas de Athena definidas en ddl_athena.sql.
D="$(dirname "$0")"
source "$D/config.sh"
TMP=$(mktemp)
sed "s/BUCKET/${BUCKET}/g" "$D/ddl_athena.sql" | python3 -c "
import sys
txt='\n'.join(l for l in sys.stdin.read().splitlines() if not l.strip().startswith('--'))
for s in txt.split(';'):
    s=' '.join(s.split())
    if s: print(s)
" > "$TMP"
while IFS= read -r q; do
  echo ">> ${q:0:70}..."
  bash "$D/athena.sh" "$q" > /dev/null < /dev/null || { echo "FALLO"; rm -f "$TMP"; exit 1; }
done < "$TMP"
rm -f "$TMP"
echo "Tablas creadas:"
aws glue get-tables --database-name gamepulse --query 'TableList[].Name' --output text
