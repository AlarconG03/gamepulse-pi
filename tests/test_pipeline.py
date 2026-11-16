"""Pruebas rápidas con datos sintéticos (no requieren data/plata_wowah.parquet).

    pytest -q
"""
import numpy as np
import pandas as pd
import pytest

from gamepulse_ml import validacion, dataset as ds, modelos as mod, recomendaciones as rec
from gamepulse_ml.config import ConfigEtiqueta, ConfigModelado


# ----------------------------------------------------------------------------- utilidades
def plata_sintetica(dias_a, dias_b=None):
    """Personaje 1 (relleno) aparece todos los días -> todos los días con cobertura.
    Personaje 2 aparece en `dias_a`. Personaje 3 aparece en `dias_b`."""
    filas = []
    for d in pd.date_range("2008-01-01", "2008-12-31"):
        for h in (10, 20):
            filas.append((1, d + pd.Timedelta(hours=h)))
    for d in dias_a:
        filas.append((2, pd.Timestamp(d) + pd.Timedelta(hours=12)))
    for d in ([] if dias_b is None else dias_b):
        filas.append((3, pd.Timestamp(d) + pd.Timedelta(hours=12)))
    df = pd.DataFrame(filas, columns=["char_id", "ts"])
    df["nivel"] = 10
    df["raza"], df["clase"], df["zona"] = "Orc", "Warrior", "Durotar"
    df["gremio"] = np.nan
    df["id_inconsistente"] = False
    return df[["char_id", "nivel", "raza", "clase", "zona", "gremio", "ts", "id_inconsistente"]]


def dataset_sintetico(cortes, h=30):
    filas = []
    for i, t in enumerate(cortes):
        for c in range(50):
            filas.append({"char_id": c, "fecha_corte": t, "fin_ventana_etiqueta": t + pd.Timedelta(days=h),
                          "churn": int((c + i) % 5 == 0), "dias_desde_ultima": 1 + c % 7,
                          "frac_dias_activos_28": (c % 10) / 10})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------- validación de datos
def test_validacion_archivo_inexistente(tmp_path):
    with pytest.raises(validacion.ErrorCalidadDatos):
        validacion.validar_plata(tmp_path / "no_existe.parquet")


def test_validacion_detecta_duplicados(tmp_path):
    df = plata_sintetica(["2008-02-01"])
    df = pd.concat([df, df.tail(1)])
    p = tmp_path / "plata.parquet"; df.to_parquet(p)
    with pytest.raises(validacion.ErrorCalidadDatos, match="duplicados"):
        validacion.validar_plata(p)


def test_validacion_detecta_nivel_fuera_de_rango(tmp_path):
    df = plata_sintetica(["2008-02-01"]); df.loc[df.index[0], "nivel"] = 95
    p = tmp_path / "plata.parquet"; df.to_parquet(p)
    with pytest.raises(validacion.ErrorCalidadDatos, match="nivel"):
        validacion.validar_plata(p)


# ----------------------------------------------------------------------------- etiqueta
def test_ventana_cuenta_solo_dias_con_cobertura():
    cal = pd.date_range("2008-01-01", "2008-12-31")
    cubierto = pd.Series(True, index=cal)
    cubierto.loc["2008-03-05":"2008-03-08"] = False           # 4 días caídos
    fin = ds._fin_ventana(pd.Timestamp("2008-03-03"), cubierto, 30)
    assert (fin - pd.Timestamp("2008-03-03")).days == 34       # 30 días cubiertos + 4 caídos


def test_etiqueta_y_variables_sin_fuga(tmp_path):
    # Personaje 2: activo todos los días de enero y febrero, y nunca más -> churn en el corte del 25/2.
    # Personaje 3: activo todos los días del año -> no churn.
    dias_a = pd.date_range("2008-01-02", "2008-02-29")
    dias_b = pd.date_range("2008-01-02", "2008-12-31")
    p = tmp_path / "plata.parquet"; plata_sintetica(dias_a, dias_b).to_parquet(p)
    out = tmp_path / "dataset.parquet"
    ds.construir_dataset(p, out, ConfigEtiqueta(primer_corte="2008-02-25"))
    d = pd.read_parquet(out)
    fila_a = d[(d.char_id == 2) & (d.fecha_corte == "2008-02-25")].iloc[0]
    fila_b = d[(d.char_id == 3) & (d.fecha_corte == "2008-02-25")].iloc[0]
    assert fila_a.churn == 0                 # aún aparece el 25-29/2, dentro de la ventana
    fila_a2 = d[(d.char_id == 2) & (d.fecha_corte == "2008-03-03")].iloc[0]
    assert fila_a2.churn == 1                # desde el 3/3 no vuelve a aparecer
    assert fila_a2.dias_desde_ultima == 3    # última aparición 29/2: solo usa el pasado
    assert fila_b.churn == 0
    assert (d.groupby("fecha_corte")["fin_ventana_etiqueta"].first()
            - d.groupby("fecha_corte")["fecha_corte"].first()).dt.days.min() >= 30


# ----------------------------------------------------------------------------- partición temporal
def test_particion_con_embargo():
    cortes = pd.date_range("2008-02-04", "2008-12-01", freq="W-MON")
    data = dataset_sintetico(cortes)
    cfg = ConfigModelado()
    p = mod.particion(data, cfg)
    assert p["entrenamiento"].fin_ventana_etiqueta.max() <= p["validacion"].fecha_corte.min()
    assert p["entren_valid"].fin_ventana_etiqueta.max() <= p["prueba"].fecha_corte.min()
    for f in mod.folds_temporales(data, cfg):
        assert f["train"].fin_ventana_etiqueta.max() <= f["val"].fecha_corte.min()
        assert len(f["val"]) > 0


def test_precision_recall_at_k():
    df = pd.DataFrame({"fecha_corte": ["a"] * 10, "churn": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]})
    p, r = mod.precision_recall_at_k(df, np.arange(10)[::-1], k_frac=0.2)   # mayor score = primeras filas
    assert p == 1.0 and r == 1.0


# ----------------------------------------------------------------------------- recomendaciones
def test_senales_por_jugador_agrupa_variables():
    sv = pd.DataFrame({"dias_desde_ultima": [0.5, -0.1], "tiene_gremio": [0.1, 0.8], "nivel": [-0.2, 0.0]})
    s = rec.senales_por_jugador(sv, n=2)
    assert s.loc[0, "senal_1"] == "ausencia_reciente"
    assert s.loc[1, "senal_1"] == "vinculo_social"
    assert s["accion_sugerida"].notna().all()


# ----------------------------------------------------------------------------- población
def test_embudo_cuadra_con_dataset(tmp_path):
    from gamepulse_ml import poblacion as pob
    dias_a = pd.date_range("2008-01-02", "2008-02-29")          # deja de jugar a fin de febrero
    dias_b = pd.date_range("2008-01-02", "2008-12-31")          # juega todo el año
    p = tmp_path / "plata.parquet"; plata_sintetica(dias_a, dias_b).to_parquet(p)
    out = tmp_path / "dataset.parquet"
    ds.construir_dataset(p, out, ConfigEtiqueta(primer_corte="2008-02-25"))
    d = pd.read_parquet(out)
    e = pob.embudo_poblacion(p, d)
    assert (e["elegibles"].values == d.groupby("fecha_corte").size().values).all()
    abril = e[e["fecha_corte"] == pd.Timestamp("2008-04-07").date()].iloc[0]
    assert abril["fuera_ausente_mas_de_30d"] == 1               # el personaje 2 ya no es población en abril


def test_leer_oro_ordena_columnas_y_rechaza_esquema_distinto(tmp_path):
    oro = pd.DataFrame({c: [1, 2] for c in reversed(ds.COLUMNAS)}).astype("int32")
    oro["fecha_corte"] = pd.to_datetime(["2008-02-11", "2008-02-04"])
    oro.to_parquet(tmp_path / "oro.parquet")
    res = ds.leer_oro(tmp_path / "oro.parquet", tmp_path / "t.parquet")
    t = pd.read_parquet(res["archivo"])
    assert list(t.columns) == ds.COLUMNAS and t["fecha_corte"].is_monotonic_increasing and res["filas"] == 2
    assert t["nivel"].dtype == "int64"
    oro.drop(columns="nivel").to_parquet(tmp_path / "malo.parquet")
    with pytest.raises(ValueError, match="nivel"):
        ds.leer_oro(tmp_path / "malo.parquet", tmp_path / "t2.parquet")
