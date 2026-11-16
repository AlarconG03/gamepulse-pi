"""API de scoring (AWS Lambda + HTTP API Gateway).

GET  /health   -> estado y metadatos del modelo cargado.
POST /predict  -> {"jugadores": [{<37 variables>}, ...]} devuelve riesgo de abandono, alerta (riesgo >= umbral)
                  y los 5 factores que más aportan al riesgo (contribuciones de LightGBM, equivalentes a SHAP).

Variables de entorno: MODEL_BUCKET y MODEL_PREFIX (p. ej. modelos/v1/); API_KEY (si existe, se exige el
encabezado x-api-key); MODEL_LOCAL_DIR para pruebas locales. Dependencias: solo numpy y lightgbm.
"""
import hmac
import json
import os

import lightgbm as lgb
import numpy as np

_CACHE = {}
MAX_JUGADORES = 5000


def _modelo():
    """Carga el modelo una vez por contenedor (arranque en frío) y lo reutiliza."""
    if "booster" not in _CACHE:
        carpeta = os.environ.get("MODEL_LOCAL_DIR")
        if not carpeta:
            import boto3
            carpeta = "/tmp/modelo"
            os.makedirs(carpeta, exist_ok=True)
            s3, pre = boto3.client("s3"), os.environ["MODEL_PREFIX"].rstrip("/")
            for f in ("modelo.txt", "meta.json"):
                s3.download_file(os.environ["MODEL_BUCKET"], f"{pre}/{f}", f"{carpeta}/{f}")
        _CACHE["booster"] = lgb.Booster(model_file=f"{carpeta}/modelo.txt")
        _CACHE["meta"] = json.load(open(f"{carpeta}/meta.json", encoding="utf-8"))
    return _CACHE["booster"], _CACHE["meta"]


def _resp(codigo, cuerpo):
    return {"statusCode": codigo, "headers": {"Content-Type": "application/json"},
            "body": json.dumps(cuerpo, ensure_ascii=False)}


def _matriz(jugadores, cols, cats):
    """Matriz numerica en el orden del modelo; las categoricas se codifican con la posicion de su nivel (como pandas)."""
    X = np.full((len(jugadores), len(cols)), np.nan)
    for i, j in enumerate(jugadores):
        for k, c in enumerate(cols):
            v = j[c]
            if v is None:
                continue
            if c in cats:
                X[i, k] = cats[c].index(v) if v in cats[c] else np.nan
            else:
                X[i, k] = float(v)
    return X


def predecir(jugadores: list) -> list:
    booster, meta = _modelo()
    cols, cats = meta["columnas"], meta["categoricas"]
    faltan = sorted({c for j in jugadores for c in cols if c not in j})
    if faltan:
        raise ValueError(f"faltan variables: {faltan}")
    try:
        X = _matriz(jugadores, cols, cats)
    except (TypeError, ValueError) as e:
        raise ValueError(f"valor no numerico en una variable: {e}")
    riesgo = booster.predict(X)
    contrib = booster.predict(X, pred_contrib=True)[:, :-1]  # ultima columna = valor base
    salida = []
    for i in range(len(X)):
        top = np.argsort(-np.abs(contrib[i]))[:5]
        salida.append({"riesgo": round(float(riesgo[i]), 4), "alerta": bool(riesgo[i] >= meta["umbral"]),
                       "factores": [{"variable": cols[k], "valor": jugadores[i][cols[k]],
                                     "aporte_logodds": round(float(contrib[i][k]), 4)} for k in top]})
    return salida


def _autorizado(event) -> bool:
    clave = os.environ.get("API_KEY")
    if not clave:
        return True
    cab = {k.lower(): v for k, v in (event.get("headers") or {}).items()}
    return hmac.compare_digest(cab.get("x-api-key") or "", clave)  # tiempo constante


def lambda_handler(event, context=None):
    if not _autorizado(event):
        return _resp(401, {"error": "clave de acceso invalida"})
    try:
        metodo = event.get("requestContext", {}).get("http", {}).get("method", event.get("httpMethod", "GET"))
        ruta = event.get("rawPath", event.get("path", "/"))
        if ruta.endswith("/health"):
            _, meta = _modelo()
            return _resp(200, {"estado": "ok", "modelo": meta["familia"], "umbral": meta["umbral"],
                               "n_variables": len(meta["columnas"]), "dataset": meta["dataset"],
                               "sha256_modelo": meta["sha256_modelo"][:12]})
        if ruta.endswith("/predict") and metodo == "POST":
            cuerpo = json.loads(event.get("body") or "{}")
            jugadores = cuerpo.get("jugadores")
            if not isinstance(jugadores, list) or not jugadores:
                return _resp(400, {"error": "se esperaba {'jugadores': [ {...}, ... ]}"})
            if len(jugadores) > MAX_JUGADORES:
                return _resp(400, {"error": f"máximo {MAX_JUGADORES} jugadores por llamada"})
            return _resp(200, {"predicciones": predecir(jugadores)})
        return _resp(404, {"error": "ruta no encontrada"})
    except ValueError as e:
        return _resp(400, {"error": str(e)})
    except Exception as e:  # no exponer detalles internos
        print("ERROR", repr(e))
        return _resp(500, {"error": "error interno"})
