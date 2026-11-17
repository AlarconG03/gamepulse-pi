# Resultados de referencia

Salidas de la corrida `pistachio-impala` (10/11/2026) del flujo `gamepulse-churn` con los parámetros por
defecto (etiqueta de 30 días con cobertura, 3 bloques de origen móvil, 3 familias de modelos), y de la corrida
`vanilla-hippo` del flujo de sensibilidad (6 escenarios de etiqueta y población). Se versionan para que el informe y la visualización trabajen
sobre números fijos. `ablacion_controlada.csv`, `variables_individuales_cv.csv` y `figuras/fig9_curvas_roc.png` salen
de la corrida `humble-peccary` (17/11/2026), con el mismo código más la mejora 5 de 4.5; sus métricas son idénticas
a las de `pistachio-impala`. Una corrida nueva queda en `runs/` y no sobrescribe esta carpeta.

| Archivo | Contenido | Úsenlo para |
|---|---|---|
| `resultados_modelos.csv` | Métricas de validación (media y desviación en origen móvil) y prueba de cada modelo, línea base y ensamble | Sección 3.3.4 del documento |
| `desempeno_por_corte_prueba.csv` | PR-AUC, precisión y recall del 10 % por fecha de corte en prueba | Estabilidad / deriva |
| `ficha_corrida.json` | Parámetros, hash de datos, versiones de librerías, id de Prefect, métricas, ablación | Reproducibilidad |
| `scores_prueba.parquet` | Por jugador y corte: riesgo de cada modelo, segmento, 3 señales, acción sugerida, si entra al top 10 % | **Visualización** |
| `matriz_recomendaciones.csv` | Perfil, tamaño, churn, prioridad, rasgos, acción posible y cómo evaluarla | Matriz de recomendaciones |
| `catalogo_senales_acciones.csv` | Señal -> descripción -> acción -> métrica de evaluación | Matriz de recomendaciones |
| `importancia_shap_prueba.csv` | Importancia global de variables (SHAP) | Interpretación |
| `perfiles_kmeans_prueba.csv`, `kmeans_diagnostico_k.csv` | Perfiles K-Means (k = 5) y codo/silueta para k = 3..8 | No supervisado |
| `sensibilidad_etiqueta.csv` | Horizontes 21/30/45/60 días y 2 poblaciones alternativas: churn, reaparición, PR-AUC propia y sobre la población base | Justificar etiqueta y población |
| `embudo_poblacion.csv` | Por corte: personajes con historia, excluidos por motivo y elegibles | Población objetivo |
| `estructura_panel.json` | Cortes por personaje, % de jugadores de prueba ya vistos al entrenar | Partición |
| `desempeno_por_subpoblacion.csv` | PR-AUC, ROC-AUC, lift y precisión/recall@10 % por subpoblación en prueba | Evaluación por población |
| `modelos_especializados.csv` | Un LightGBM por subpoblación frente al modelo global | ¿Poblaciones específicas? |
| `ablacion_controlada.csv` | LightGBM con y sin las 3 variables censuradas, mismos hiperparámetros: PR-AUC en validación (0,513 → 0,497) y en prueba (0,618 → 0,611) | Sección 3.3.2 y mejora 5 de 4.5 |
| `variables_individuales_cv.csv` | Las 37 variables evaluadas por separado (regresión logística, bloques de validación): ROC-AUC y PR-AUC medias | Mejora 5 de 4.5 |
| `auditoria/` | Cobertura diaria, huecos del colector, saltos, tasa de retorno | Sección 3.2.3 del documento |
| `figuras/` | fig1–fig9 para informe y presentación (fig7: ventanas del corte; fig8: embudo de la población; fig9: curvas ROC) | Informe / presentación |
| `modelo/lightgbm.joblib` | Modelo elegido con sus columnas, hiperparámetros y umbral | Despliegue (`aws/api/exportar_modelo.py`) |
| `registro_ejecuciones.csv` | Historial de corridas hasta la de referencia | Trazabilidad |

Cargar el modelo:

```python
import joblib, pandas as pd
m = joblib.load("resultados/referencia/modelo/lightgbm.joblib")
X = pd.read_parquet("runs/cache/dataset_H30_e7_m2_c20.parquet")
for c in ["clase", "raza"]:
    X[c] = X[c].astype("category")
riesgo = m["modelo"].predict_proba(X[m["columnas"]])[:, 1]
alerta = riesgo >= m["umbral"]
```
