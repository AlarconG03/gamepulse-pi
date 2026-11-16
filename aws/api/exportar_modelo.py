"""Exporta el LightGBM entrenado a un formato portable: `modelo.txt` (booster en texto) y `meta.json`.
La API solo necesita LightGBM (sin scikit-learn ni joblib). Opcionalmente sube la carpeta a S3.

Uso: python aws/api/exportar_modelo.py <modelo.joblib> <carpeta_salida> [s3://bucket/modelos/v1/]
"""
import hashlib
import json
import sys
from pathlib import Path

import joblib


def exportar(joblib_path: str, salida: str) -> Path:
    d = joblib.load(joblib_path)
    booster = d["modelo"].booster_
    cat_names = ["clase", "raza"]
    niveles = dict(zip(cat_names, booster.pandas_categorical))
    salida = Path(salida)
    salida.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(salida / "modelo.txt"))
    meta = {"familia": "LightGBM", "columnas": d["columnas"], "categoricas": niveles, "umbral": d["umbral"],
            "params": d["params"], "dataset": d["meta"].get("dataset"), "sha256_datos": d["meta"].get("sha256_datos"),
            "version_paquete": d["meta"].get("version_paquete"),
            "sha256_modelo": hashlib.sha256((salida / "modelo.txt").read_bytes()).hexdigest()}
    (salida / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return salida


if __name__ == "__main__":
    out = exportar(sys.argv[1], sys.argv[2])
    print("Modelo exportado en", out)
    if len(sys.argv) > 3:
        import subprocess
        subprocess.run(["aws", "s3", "cp", str(out), sys.argv[3], "--recursive"], check=True)
