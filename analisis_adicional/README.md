# Análisis adicional

Scripts que justifican cifras y figuras del documento sin reentrenar el modelo. Se ejecutan desde la raíz del
repositorio; los que leen datos necesitan `data/plata_wowah.parquet` (ver `data/README.md`). `ic_y_lift.py`,
`diagnostico_nivel_maximo.py` y `diagnostico_y_negocio.py` leen además `runs/cache/dataset_H30_e7_m2_c20.parquet`,
que deja el pipeline (`python flows/pipeline_flow.py`).

| Script | Qué produce | En el documento |
|---|---|---|
| `auditoria_plata.py` | Auditoría de la capa plata: cobertura, silencios, vida de los personajes, cambio de régimen, tasa de la etiqueta y señales preliminares (H1–H5). Salida de texto en `auditoria_resultados.txt` y tabla `ventanas_semanales.csv`. El bloque 9c de la salida (H5) necesita `zones.csv` de Kaggle. | 3.2 y Figura 1 |
| `ic_y_lift.py`, `figura_ic_lift.py` | Intervalos de confianza por bootstrap, lift por corte y calibración. | 3.3.4 y 4.5 (mejoras 1 y 2) |
| `diagnostico_y_negocio.py` | Búsqueda de hiperparámetros, sesgo y varianza, valor de negocio y casos aplicados. | 3.3.3 a 3.3.6 y Figura 3 |
| `diagrama_arquitectura.py` | Diagrama de la arquitectura implementada. | Figura 5 |
| `diagnostico_nivel_maximo.py` | Diagnóstico de la subpoblación en nivel máximo y del nivel 70 tras la expansión (`resultados/adicional/nivel_maximo*.csv`). | 3.3.4 y 4.5 (mejora 3) |

Las figuras 2 y 4 las genera el pipeline de modelado (`flows/pipeline_flow.py`; copias en
`resultados/referencia/figuras/`). Las figuras 6 a 11 son capturas de la consola de AWS y las 12 y 13, del tablero.
En `docs/documento/figuras/` cada archivo lleva el número de su figura; su `README.md` dice de qué salida viene cada uno.

```bash
python analisis_adicional/auditoria_plata.py data/ salida_auditoria/ [ruta/a/zones.csv]
python analisis_adicional/ic_y_lift.py
python analisis_adicional/diagnostico_nivel_maximo.py
python analisis_adicional/figura_ic_lift.py
python analisis_adicional/diagnostico_y_negocio.py
python analisis_adicional/diagrama_arquitectura.py docs/documento/figuras/fig05_arquitectura.png
```
