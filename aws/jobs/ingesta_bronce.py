"""Job de AWS Glue (Python Shell): fuente -> landing -> validacion -> BRONCE.

1. Descarga el dataset de Kaggle con su API (credenciales en AWS Secrets Manager: {"username","key"} o {"token"}; nunca en el codigo).
   Si el archivo ya esta en `landing/` (`--usar_landing true`) se omite la descarga.
2. Valida el CSV (encabezado, filas, nulos, formato de fecha) leyendolo por bloques.
3. Si pasa: lo copia a `bronce/wowah/fecha_ingesta=AAAA-MM-DD/` y escribe un registro JSON en `bronce/_ingesta_log/`.
   Si falla: mueve el archivo a `landing/errores/`, escribe el registro con el motivo y termina con error
   (el Workflow se detiene y no se ejecutan las etapas siguientes).

Parametros: --bucket --secreto [--dataset mylesoneill/warcraft-avatar-history] [--archivo wowah_data.csv]
            [--usar_landing false] [--min_filas 1000000]
"""
import argparse
import base64
import csv
import io
import json
import os
import re
import sys
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone

import boto3

ESPERADAS = ["char", "level", "race", "charclass", "zone", "guild", "timestamp"]
FORMATO_TS = re.compile(r"^\d{1,2}/\d{1,2}/\d{2} \d{2}:\d{2}:\d{2}$")
MAX_PCT_NULOS = 0.01                  # 1 % de nulos por columna como maximo
MUESTRA_FORMATO = 50000               # filas sobre las que se revisa el formato de fecha


def parsear():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--secreto", default="gamepulse/kaggle")
    ap.add_argument("--dataset", default="mylesoneill/warcraft-avatar-history")
    ap.add_argument("--archivo", default="wowah_data.csv")
    ap.add_argument("--usar_landing", default="false")
    ap.add_argument("--min_filas", type=int, default=1_000_000)
    return ap.parse_known_args()[0]


class _SinRedireccion(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None                                                   # la redireccion se sigue a mano, sin credenciales


def descargar_kaggle(a, destino):
    """Descarga el zip con la API REST de Kaggle (sin dependencias). Credenciales del secreto:
    {"token": ...} (token nuevo, Bearer) o {"username": ..., "key": ...} (kaggle.json, Basic)."""
    sm = boto3.client("secretsmanager")
    cred = {k: str(v).strip() for k, v in json.loads(sm.get_secret_value(SecretId=a.secreto)["SecretString"]).items()}
    req = urllib.request.Request(f"https://www.kaggle.com/api/v1/datasets/download/{a.dataset}")
    if "token" in cred:
        req.add_header("Authorization", f"Bearer {cred['token']}")
    else:
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{cred['username']}:{cred['key']}".encode()).decode())
    opener = urllib.request.build_opener(_SinRedireccion)
    try:
        resp = opener.open(req)
    except urllib.error.HTTPError as e:
        if e.code not in (301, 302, 303, 307, 308):
            raise RuntimeError(f"Kaggle respondio HTTP {e.code}: revisa el token y que el dataset sea accesible")
        resp = urllib.request.urlopen(e.headers["Location"])          # URL firmada: sin cabecera de autorizacion
    ruta_zip = os.path.join(destino, "dataset.zip")
    with resp, open(ruta_zip, "wb") as f:
        shutil.copyfileobj(resp, f, 1 << 20)
    with zipfile.ZipFile(ruta_zip) as z:
        nombres = {os.path.basename(n): n for n in z.namelist()}
        if a.archivo not in nombres:
            raise RuntimeError(f"{a.archivo} no esta en el zip: {list(nombres)}")
        z.extract(nombres[a.archivo], destino)
        return os.path.join(destino, nombres[a.archivo])


def validar(ruta, min_filas):
    """Devuelve (metricas, errores). Lee el CSV por bloques: no carga el archivo completo en memoria."""
    errores, nulos = [], dict.fromkeys(ESPERADAS, 0)
    filas, malas_fecha, malas_ancho = 0, 0, 0
    with open(ruta, newline="", encoding="utf-8") as f:
        lector = csv.reader(f)
        encabezado = [c.strip() for c in next(lector)]
        if encabezado != ESPERADAS:
            return {"encabezado": encabezado}, [f"encabezado inesperado: {encabezado}"]
        for fila in lector:
            filas += 1
            if len(fila) != len(ESPERADAS):
                malas_ancho += 1
                continue
            for c, v in zip(ESPERADAS, fila):
                if v.strip() == "":
                    nulos[c] += 1
            if filas <= MUESTRA_FORMATO and not FORMATO_TS.match(fila[6].strip()):
                malas_fecha += 1
    m = {"filas": filas, "filas_con_ancho_incorrecto": malas_ancho, "nulos_por_columna": nulos,
         "fechas_con_formato_incorrecto_en_muestra": malas_fecha, "tamano_bytes": os.path.getsize(ruta)}
    if filas < min_filas:
        errores.append(f"solo {filas} filas (minimo {min_filas})")
    if malas_ancho > 0.001 * max(filas, 1):
        errores.append(f"{malas_ancho} filas con numero de columnas incorrecto")
    for c, n in nulos.items():
        if filas and n / filas > MAX_PCT_NULOS:
            errores.append(f"columna {c}: {n / filas:.2%} de nulos (maximo {MAX_PCT_NULOS:.0%})")
    if malas_fecha > 0.001 * MUESTRA_FORMATO:
        errores.append(f"{malas_fecha} fechas con formato distinto de MM/dd/yy HH:mm:ss en la muestra")
    return m, errores


def main():
    a = parsear()
    s3 = boto3.client("s3")
    hoy = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log = {"inicio": datetime.now(timezone.utc).isoformat(), "dataset": a.dataset, "estado": "ERROR"}
    clave_landing = f"landing/{a.archivo}"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            if a.usar_landing.lower() == "true":
                local = os.path.join(tmp, a.archivo)
                s3.download_file(a.bucket, clave_landing, local)
                log["fuente"] = f"s3://{a.bucket}/{clave_landing}"
            else:
                local = descargar_kaggle(a, tmp)
                s3.upload_file(local, a.bucket, clave_landing)           # copia cruda en landing
                log["fuente"] = f"kaggle:{a.dataset}"
            metricas, errores = validar(local, a.min_filas)
            log["validacion"], log["errores"] = metricas, errores
            if errores:
                destino = f"landing/errores/{hoy}/{a.archivo}"
                s3.copy_object(Bucket=a.bucket, CopySource={"Bucket": a.bucket, "Key": clave_landing}, Key=destino)
                log["archivo_en_cuarentena"] = f"s3://{a.bucket}/{destino}"
            else:
                destino = f"bronce/wowah/fecha_ingesta={hoy}/{a.archivo}"
                s3.upload_file(local, a.bucket, destino)
                log.update(estado="OK", destino=f"s3://{a.bucket}/{destino}")
                # bronce conserva solo la ultima carga (el historial queda en el registro): evita filas duplicadas
                viejos = [o["Key"] for p in s3.get_paginator("list_objects_v2").paginate(Bucket=a.bucket, Prefix="bronce/wowah/")
                          for o in p.get("Contents", []) if not o["Key"].endswith("/") and f"fecha_ingesta={hoy}/" not in o["Key"]]
                for k in viejos:
                    s3.delete_object(Bucket=a.bucket, Key=k)
                log["particiones_antiguas_eliminadas"] = len(viejos)
        except Exception as e:                                           # cualquier fallo queda registrado
            log["errores"] = log.get("errores", []) + [f"{type(e).__name__}: {e}"]
    log["fin"] = datetime.now(timezone.utc).isoformat()
    clave_log = f"bronce/_ingesta_log/{hoy}_{datetime.now(timezone.utc).strftime('%H%M%S')}.json"
    s3.put_object(Bucket=a.bucket, Key=clave_log, Body=json.dumps(log, indent=2, ensure_ascii=False).encode())
    print(json.dumps(log, indent=2, ensure_ascii=False))
    if log["estado"] != "OK":
        sys.exit(f"Ingesta fallida: {log['errores']}")


if __name__ == "__main__":
    main()
