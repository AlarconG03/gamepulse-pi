"""Parámetros del pipeline. Todos tienen un valor por defecto y se pueden cambiar desde Prefect."""
from dataclasses import dataclass, asdict, field
import pandas as pd

FIN_DATOS = pd.Timestamp("2008-12-31")
INICIO_DATOS = pd.Timestamp("2008-01-01")
EXPANSION = pd.Timestamp("2008-11-13")      # tope de nivel 70 -> 80 (Wrath of the Lich King)


@dataclass
class ConfigEtiqueta:
    horizonte_dias: int = 30          # días CON COBERTURA sin aparecer para churn = 1
    eleg_dias: int = 7                # elegible si fue visto en los últimos N días
    min_dias_hist: int = 2            # mínimo de días activos en el historial
    umbral_cobertura: float = 0.20    # día sin cobertura si registros < 20 % de la mediana
    min_dias_cubiertos_eleg: int = 5  # días cubiertos mínimos en la semana previa al corte
    primer_corte: str = "2008-02-04"  # primer lunes con 28 días de historia

    def clave(self) -> str:
        return (f"H{self.horizonte_dias}_e{self.eleg_dias}_m{self.min_dias_hist}"
                f"_c{int(self.umbral_cobertura * 100)}")


@dataclass
class ConfigModelado:
    val_ini: str = "2008-08-25"       # bloque de validación (selección de hiperparámetros y umbral)
    val_fin: str = "2008-09-22"
    test_ini: str = "2008-10-27"      # bloque de prueba (se evalúa una sola vez)
    n_folds: int = 3                  # bloques de validación con origen móvil (1 = solo el bloque val)
    semanas_por_fold: int = 5
    k_frac: float = 0.10              # capacidad de intervención: % de mayor riesgo por corte
    k_segmentos: int = 5              # número de perfiles K-Means
    seed: int = 42
    familias: list = field(default_factory=lambda: ["Regresión logística", "Random Forest", "LightGBM"])


def a_dict(*cfgs) -> dict:
    out = {}
    for c in cfgs:
        out.update(asdict(c))
    return out
