"""GamePulse - componente de aprendizaje automático.

Paquete con las etapas del pipeline como funciones puras (sin dependencia de Prefect):

    validacion       -> controles de calidad sobre la capa plata
    auditoria        -> auditoría de cobertura temporal (sustenta la etiqueta)
    dataset          -> etiqueta + variables por personaje y fecha de corte (sin fuga)
    modelos          -> partición temporal con embargo, entrenamiento, selección y evaluación
    interpretacion   -> SHAP, perfiles K-Means, ablación de variables censuradas
    recomendaciones  -> matriz perfil/señal -> acción posible, y señales por jugador
    figuras          -> gráficas para el informe
    registro         -> trazabilidad local (config, versiones, hash de datos, historial de corridas)

El orquestador (Prefect) vive en ``flows/`` y llama a estas funciones.
"""
import logging

__version__ = "2.0.0"
logger = logging.getLogger("gamepulse")
logger.setLevel(logging.INFO)
