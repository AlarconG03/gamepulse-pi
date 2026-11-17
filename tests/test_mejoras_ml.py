"""Pruebas de las mejoras de ML: curvas ROC, variables individuales sin fuga y ablación controlada."""

import numpy as np
import pandas as pd

from gamepulse_ml import figuras, modelos, interpretacion
from gamepulse_ml.config import ConfigModelado


def datos_sinteticos():
    """Genera datos pequeños para probar los análisis de ML."""
    filas = []

    fechas = pd.date_range(
        "2008-02-04", "2008-12-01", freq="W-MON"
    )

    for i, fecha in enumerate(fechas):
        for j in range(24):
            filas.append({
                "char_id": j,
                "fecha_corte": fecha,
                "fin_ventana_etiqueta": fecha + pd.Timedelta(days=30),
                "churn": int((j + i) % 4 == 0),
                "clase": "Warrior" if j % 2 else "Mage",
                "raza": "Orc" if j % 3 else "Troll",
                "frac_dias_activos_28": float((j + 2 * i) % 20) / 20,
                "dias_desde_ultima": float(j % 7),
                "antiguedad_censurada": float(i == 0),
                "dias_activos_hist": float(j + i),
                "antiguedad_dias": float(10 + i * 7),
            })

    return pd.DataFrame(filas)


def test_generacion_curvas_roc(tmp_path):
    """Comprueba que la gráfica ROC se genera correctamente."""
    y = np.array([0, 1, 0, 1, 0, 1])

    scores = {
        "Regla": [0.1, 0.8, 0.3, 0.9, 0.5, 0.6],
        "Modelo": [0.3, 0.9, 0.1, 0.7, 0.4, 0.8],
    }

    ruta = tmp_path / "fig9_curvas_roc.png"

    resultado = figuras.curvas_roc(y, scores, ruta)

    assert resultado == str(ruta)
    assert ruta.exists()
    assert ruta.stat().st_size > 2000


def test_variables_individuales_sin_fuga():
    """Comprueba el ranking temporal sin utilizar etiquetas de prueba."""
    data = datos_sinteticos()
    cfg = ConfigModelado(n_folds=2, semanas_por_fold=3)

    ranking = interpretacion.evaluar_variables_individuales(
        data, cfg
    )

    assert len(ranking) == 7
    assert ranking["ROC_AUC_cv_media"].between(0, 1).all()
    assert ranking["ROC_AUC_cv_media"].is_monotonic_decreasing

    # Cambiar solo las etiquetas de prueba
    modificados = data.copy()
    mask = modificados["fecha_corte"] >= pd.Timestamp(cfg.test_ini)

    modificados.loc[mask, "churn"] = (
        1 - modificados.loc[mask, "churn"]
    )

    nuevo_ranking = interpretacion.evaluar_variables_individuales(
        modificados, cfg
    )

    pd.testing.assert_frame_equal(ranking, nuevo_ranking)


def test_ablacion_con_parametros_fijos():
    """Comprueba la comparación usando los parametros originales."""
    data = datos_sinteticos()
    cfg = ConfigModelado(n_folds=2, semanas_por_fold=3)

    original = modelos.entrenar_familia(
        "Regresión logística", data, cfg
    )

    resultado = interpretacion.ablacion_controlada(
        data, cfg, original
    )

    assert resultado["hiperparametros"] == original["params"]
    assert resultado["n_folds"] == cfg.n_folds

    cv_original = next(
        item["PR_AUC_media"]
        for item in original["cv"]
        if item["params"] == original["params"]
    )

    assert resultado["PR_AUC_cv_con"] == cv_original

    diferencia = round(
        resultado["PR_AUC_cv_sin"] - resultado["PR_AUC_cv_con"],
        4
    )

    assert resultado["diferencia_cv"] == diferencia
    assert 0 <= resultado["ROC_AUC_test_sin"] <= 1
