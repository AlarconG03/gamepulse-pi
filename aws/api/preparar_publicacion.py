"""Prepara, a partir de la corrida de referencia, los artefactos que se publican en S3:
  - modelo portable (modelo.txt + meta.json)           -> modelos/v1/
  - tablas de resultados en Parquet (nombres de columna aptos para Athena) -> oro/scores/, oro/perfiles/, oro/shap_global/
  - ejemplo de solicitud para la API (un jugador real de la prueba con riesgo alto)
Uso: python aws/api/preparar_publicacion.py [carpeta_referencia] [dataset.parquet]
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from exportar_modelo import exportar  # noqa: E402

REF = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "resultados/referencia"
DATASET = Path(sys.argv[2]) if len(sys.argv) > 2 else RAIZ / "runs/cache/dataset_H30_e7_m2_c20.parquet"
SALIDA = RAIZ / "aws/publicar"


def main():
    exportar(REF / "modelo/lightgbm.joblib", SALIDA / "modelos/v1")
    s = pd.read_parquet(REF / "scores_prueba.parquet").rename(columns={
        "riesgo_regresion-logistica": "riesgo_regresion_logistica", "riesgo_random-forest": "riesgo_random_forest"})
    (SALIDA / "scores").mkdir(parents=True, exist_ok=True)
    s.to_parquet(SALIDA / "scores/scores.parquet", index=False)
    (SALIDA / "perfiles").mkdir(exist_ok=True)
    pd.read_csv(REF / "matriz_recomendaciones.csv").rename(columns={"%_en_top_riesgo": "pct_en_top_riesgo"}).to_parquet(SALIDA / "perfiles/perfiles.parquet", index=False)
    (SALIDA / "shap_global").mkdir(exist_ok=True)
    sh = pd.read_csv(REF / "importancia_shap_prueba.csv")
    sh.columns = ["variable", "mean_abs_shap"]
    sh.to_parquet(SALIDA / "shap_global/shap_global.parquet", index=False)
    meta = json.loads((SALIDA / "modelos/v1/meta.json").read_text())
    d = pd.read_parquet(DATASET)
    ej = s.sort_values("riesgo_lightgbm", ascending=False).iloc[0]
    fila = d[(d.char_id == ej.char_id) & (d.fecha_corte == ej.fecha_corte)].iloc[0]
    jug = {c: (None if pd.isna(fila[c]) else (fila[c].item() if hasattr(fila[c], "item") else fila[c]))
           for c in meta["columnas"]}
    (RAIZ / "aws/api/ejemplo_solicitud.json").write_text(json.dumps({"jugadores": [jug]}, ensure_ascii=False, indent=1))
    print("Listo:", sorted(p.relative_to(SALIDA).as_posix() for p in SALIDA.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
