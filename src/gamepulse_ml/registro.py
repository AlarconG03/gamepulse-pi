"""Trazabilidad local, independiente de Prefect: cada corrida deja su carpeta con configuración,
versiones, hash de datos, métricas y modelos, y una fila en el historial común."""
import json, platform, subprocess, sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
import pandas as pd

LIBRERIAS = ["pandas", "numpy", "pyarrow", "scikit-learn", "lightgbm", "shap", "prefect"]


def versiones() -> dict:
    v = {"python": sys.version.split()[0], "sistema": platform.platform()}
    for lib in LIBRERIAS:
        try:
            v[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            v[lib] = None
    return v


def commit_git(carpeta) -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=carpeta, capture_output=True,
                              text=True, timeout=5).stdout.strip() or None
    except Exception:
        return None


def carpeta_corrida(base, nombre: str) -> Path:
    d = Path(base) / f"{datetime.now():%Y%m%d_%H%M%S}_{nombre}"
    (d / "modelos").mkdir(parents=True, exist_ok=True)
    (d / "figuras").mkdir(exist_ok=True)
    return d


def guardar_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))
    return str(path)


def anexar_historial(base, fila: dict) -> Path:
    path = Path(base) / "registro_ejecuciones.csv"
    df = pd.DataFrame([fila])
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path
