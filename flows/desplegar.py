"""
GamePulse - Despliegue de los flujos en el servidor Prefect (escenario de implementación real).

Registra dos deployments y deja un proceso escuchando para ejecutarlos:
  * gamepulse-churn/semanal         -> todos los lunes 03:00 (hora de Bogotá), como se reentrenaría en producción
  * gamepulse-sensibilidad-etiqueta/manual -> sin calendario; se lanza desde el dashboard con "Run"

    python flows/desplegar.py          # dejar esta terminal abierta (Ctrl+C para detener)

Desde el dashboard (Deployments) se puede lanzar cualquiera con "Quick run" o "Custom run" cambiando parámetros.
"""
import sys
from pathlib import Path
from prefect import serve
from prefect.schedules import Cron

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "flows"))

from pipeline_flow import pipeline_churn
from sensibilidad_flow import sensibilidad_etiqueta

if __name__ == "__main__":
    semanal = pipeline_churn.to_deployment(
        name="semanal", schedules=[Cron("0 3 * * 1", timezone="America/Bogota")],
        tags=["gamepulse", "churn", "produccion-simulada"],
        description="Reentrena y evalúa el modelo de churn cada lunes, como en un despliegue real.")
    manual = pipeline_churn.to_deployment(name="manual", tags=["gamepulse", "churn"],
                                          description="Ejecución a demanda con parámetros personalizados.")
    sens = sensibilidad_etiqueta.to_deployment(name="manual", tags=["gamepulse", "etiqueta"])
    serve(semanal, manual, sens)
