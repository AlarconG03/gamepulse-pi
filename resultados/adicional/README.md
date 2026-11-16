# Resultados adicionales

Salidas de los scripts de `analisis_adicional/`, calculadas sobre `resultados/referencia/` sin reentrenar el
modelo de referencia y sin modificar esa carpeta. Los programas comprueban que las métricas recalculadas
coinciden con `resultados_modelos.csv` antes de seguir.

## Intervalos, lift y calibración (mejoras 1 y 2 de 4.5; 3.3.4)

`analisis_adicional/ic_y_lift.py` y `analisis_adicional/figura_ic_lift.py`.

| Archivo | Contenido |
|---|---|
| `ic_bootstrap.csv` | PR-AUC, ROC-AUC, precisión@10 %, recall@10 % y lift de cada modelo en prueba, con IC 95 % por bootstrap agrupado por personaje (2.000 remuestreos, semilla 42) |
| `ic_diferencias.csv` | Diferencias entre pares de modelos con su IC 95 % e indicador de si el intervalo excluye el cero |
| `lift.csv` | Lift de la PR-AUC en validación (tasa base 13,2 %) y en prueba (19,6 %), y lift@10 % en prueba |
| `calibracion_por_corte.csv` | Riesgo medio estimado por LightGBM frente al churn observado en cada corte de prueba (mejora 5, trabajo futuro) |
| `fig_ic_lift.png` | Figura de los intervalos y del lift |
| `salida_consola.txt` | Salida completa de la ejecución |

## Diagnóstico del modelo y valor de negocio (3.3.3 a 3.3.6, Figura 3)

`analisis_adicional/diagnostico_y_negocio.py`.

| Archivo | Contenido |
|---|---|
| `busqueda_hiperparametros.csv` | PR-AUC de cada configuración en los 3 bloques de origen móvil, su media, desviación y cuál se eligió |
| `curva_aprendizaje.csv` | PR-AUC en entrenamiento y validación según la fracción de entrenamiento usada |
| `curva_validacion_arboles.csv` | PR-AUC en entrenamiento y validación según el número de árboles de LightGBM |
| `valor_negocio.csv` | Intervenciones, abandonos alcanzados y beneficio neto de cada política |
| `beneficio_vs_K.csv` | Beneficio neto según la capacidad de intervención K: modelo, regla de recencia y azar |
| `valor_negocio_sensibilidad.csv` | Beneficio neto con otros supuestos de efecto y costo, y la precisión de equilibrio |
| `casos_aplicados.csv` | Cinco personajes del corte 10/11/2008 con su riesgo, percentil y factores SHAP |
| `fig_diagnostico_negocio.png` | Figura 3 del documento |

## Nivel máximo (mejora 3 de 4.5; 3.3.4)

`analisis_adicional/diagnostico_nivel_maximo.py`.

| Archivo | Contenido |
|---|---|
| `nivel_maximo.csv` | Métricas de la subpoblación en nivel máximo y del nivel 70 tras la expansión, con su nota |
| `nivel_maximo_por_corte.csv` | Filas, positivos y tasa de abandono del grupo en nivel máximo por corte de prueba |

Uso (desde la raíz del repositorio, después de correr el pipeline, que genera `runs/cache/dataset_*.parquet`):

```bash
python analisis_adicional/ic_y_lift.py
python analisis_adicional/figura_ic_lift.py
python analisis_adicional/diagnostico_y_negocio.py
python analisis_adicional/diagnostico_nivel_maximo.py
```
