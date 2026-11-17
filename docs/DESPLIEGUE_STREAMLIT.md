# Tablero: correrlo en local o publicarlo en Streamlit Community Cloud

Tablero publicado: <https://gamepulse-pi.streamlit.app>

El tablero vive en `tablero/` y trae lo que necesita:

- `tablero/app.py`: el archivo de entrada.
- `tablero/requirements.txt`: las dependencias.
- `packages.txt` (en la raíz y en `tablero/`): `libgomp1`, la librería del sistema que necesita LightGBM en Streamlit Cloud.
- `tablero/.streamlit/config.toml`: el tema.
- `tablero/datos/`: la instantánea de datos (scores, perfiles, SHAP global, variables de la prueba y el modelo `v1`).

El detalle de cada personaje se pide a la API de la sección 5. La URL y la clave de la API no están en el
repositorio. Sin ellas el tablero funciona igual en "modo sin conexión": usa el mismo código de la API
(`aws/api/handler.py`) con el modelo de la instantánea, y la franja superior lo indica.

## 1. Correrlo en local

Requiere Python 3.10 o superior.

1. Obtenga el código. Con git:

   ```bash
   git clone https://github.com/AlarconG03/gamepulse-pi.git
   cd gamepulse-pi
   ```

   O sin git: en GitHub, **Code → Download ZIP**, descomprima `gamepulse-pi-main.zip` y entre a la carpeta
   `gamepulse-pi-main`.

2. Cree un entorno e instale las dependencias del tablero:

   ```bash
   python -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   pip install -r tablero/requirements.txt
   ```

3. Opcional, para pedir el detalle a una API desplegada: copie `tablero/.streamlit/secrets.toml.example` como
   `tablero/.streamlit/secrets.toml` y complete `url` y `clave`. Ese archivo está en `.gitignore`.
4. Desde la raíz del repositorio:

   ```bash
   streamlit run tablero/app.py
   ```

   Se abre en <http://localhost:8501>.

## 2. Publicar su propia copia en Streamlit Community Cloud

Requiere una cuenta de GitHub. Costo: gratis.

1. Tenga el repositorio en su cuenta de GitHub: un *fork* de `AlarconG03/gamepulse-pi` (botón **Fork**) o
   el repositorio original si es suyo.
2. Entre a <https://share.streamlit.io> con su cuenta de GitHub y autorice el acceso.
3. **Create app** → **Deploy a public app from GitHub**.
4. Repository: `<su-cuenta>/<nombre-del-repo>` · Branch: `main` · Main file path: `tablero/app.py`.
5. Elija la dirección de la app (queda `https://<nombre-que-elija>.streamlit.app`).
6. **Advanced settings** → Python version: 3.12 o 3.13.

### Secretos (opcional: detalle por la API)

En **Advanced settings → Secrets** (o después en **Settings → Secrets**) pegue, con sus valores:

```toml
[api]
url = "https://<id>.execute-api.us-east-1.amazonaws.com"
clave = "<clave de la API>"
```

Quien desplegó la API con `aws/infra/paso6_api.sh` encuentra ambos valores en CloudShell, en `~/.gamepulse_api_url`
y `~/.gamepulse_api_key`. Nunca los escriba en un archivo del repositorio.

### Desplegar y probar

1. **Deploy**. La primera vez tarda de 3 a 5 minutos.
2. Compruebe que el bloque 1 dice "72 %" frente a "20 %" y que la franja dice "22.380 puntuaciones en 6 cortes".
3. Con secretos, al seleccionar un personaje la franja dice "API en AWS (respondió)"; sin ellos, "modo sin conexión".

Cada cambio que llegue a `main` se publica solo en 1 o 2 minutos.

## Problemas frecuentes

| Síntoma | Causa y solución |
|---|---|
| `OSError: libgomp.so.1` | Falta `packages.txt` con `libgomp1` en la raíz del repositorio. |
| `Falta .../tablero/datos/scores` | La carpeta `tablero/datos/` está incompleta. |
| La franja dice "modo sin conexión" con secretos puestos | Revise `[api]`, `url` y `clave` en los secretos. |
| La app "se durmió" | Pasa tras días sin visitas. Clic en **Yes, get this app back up** (unos 30 s). |
