# Tablero GamePulse (sección 6)

Tablero en Streamlit para el equipo de retención: dónde se concentra el riesgo, qué señales lo acompañan,
lista de acción priorizada y detalle de cada personaje (riesgo y 5 factores pedidos a la API).

## Datos: instantánea exportada de Athena
1. En CloudShell (con la carpeta `aws/` del repositorio en `~/aws/`, ver `aws/GUIA_PASO_A_PASO.md`):  `bash aws/infra/exportar_tablero.sh`
   (UNLOAD a Parquet de `scores`, `perfiles_segmento`, `shap_global` y `dataset_modelado` del período de prueba,
   más `modelos/v1/`). Descargar `~/tablero_datos.zip` con *Actions > Download file*.
2. Descomprimir en `tablero/datos/` (deben quedar `scores/`, `perfiles/`, `shap_global/`, `casos/`, `modelo/`).

Sin AWS (desarrollo): `python tablero/preparar_datos_locales.py` copia los mismos archivos desde el repositorio
(necesita `runs/cache/dataset_H30_e7_m2_c20.parquet`, que deja el pipeline). `tablero/datos/` ya viene en el repositorio.

## API (detalle del caso)
Copiar `tablero/.streamlit/secrets.toml.example` como `tablero/.streamlit/secrets.toml` y completar URL y
clave (el archivo no se versiona). Sin ese archivo, o si la API no responde, el detalle usa el mismo código de
la Lambda (`aws/api/handler.py`) en local y el tablero lo indica.

## Ejecutar (desde la raíz del repositorio)
```
pip install -r tablero/requirements.txt
streamlit run tablero/app.py
```
