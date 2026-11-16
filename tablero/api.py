"""Detalle de un caso: riesgo y los 5 factores que más aportan, pedidos a la API de scoring.

Con URL y clave en `.streamlit/secrets.toml` ([api] url, clave) se llama a la API real (Lambda + API Gateway).
Sin ellas, o si la API no responde, se usa el MISMO código de la Lambda (`aws/api/handler.py`) en local con el
modelo de la instantánea; el tablero indica siempre cuál de las dos fuentes respondió.
"""
import json
import math
import os
import sys
from pathlib import Path

import requests

RAIZ = Path(__file__).resolve().parents[1]


def _limpio(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):  # escalares de numpy
        v = v.item()
    return v


def solicitud(fila: dict, columnas: list) -> dict:
    return {"jugadores": [{c: _limpio(fila[c]) for c in columnas}]}


def _local(cuerpo: dict, carpeta_modelo: Path) -> dict:
    os.environ["MODEL_LOCAL_DIR"] = str(carpeta_modelo)
    sys.path.insert(0, str(RAIZ / "aws" / "api"))
    import handler  # noqa: E402  (mismo código desplegado en la Lambda)
    return handler.predecir(cuerpo["jugadores"])[0]


def explicar(fila: dict, columnas: list, carpeta_modelo: Path, url: str | None, clave: str | None) -> tuple[dict, str]:
    cuerpo = solicitud(fila, columnas)
    if url and clave:
        try:
            r = requests.post(url.rstrip("/") + "/predict", json=cuerpo, timeout=15,
                              headers={"x-api-key": clave, "Content-Type": "application/json"})
            r.raise_for_status()
            return r.json()["predicciones"][0], "API en AWS (Lambda + API Gateway)"
        except (requests.RequestException, KeyError, ValueError, json.JSONDecodeError) as e:
            return _local(cuerpo, carpeta_modelo), f"modo sin conexión (la API no respondió: {type(e).__name__})"
    return _local(cuerpo, carpeta_modelo), "modo sin conexión (mismo código de la API, en local)"
