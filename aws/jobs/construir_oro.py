"""Job de AWS Glue (PySpark): capa PLATA -> capa ORO (tabla de modelado).

Una fila por personaje x fecha de corte (lunes) con 37 variables y la etiqueta `churn`.
Es la misma logica de `src/gamepulse_ml/dataset.py` (pandas), expresada con DataFrames de Spark:
  * cobertura diaria y minutos por pasada (agregados pequenos, al driver: 366 dias),
  * tabla diaria por personaje (D) y por zona (Z), distribuidas,
  * variables por corte mediante joins con la tabla de cortes y agregaciones/ventanas.
Las variables usan solo filas con ts < corte; la etiqueta, solo filas con ts >= corte (sin fuga).

Parametros: --plata s3://.../plata/wowah/  --salida s3://.../oro/dataset_modelado/  [--porcentaje 100]
`--porcentaje` toma una muestra determinista de personajes (solo para medir escalabilidad).
"""
import argparse
import json
import time

import pandas as pd
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

HORIZONTE, ELEG_DIAS, MIN_DIAS_HIST, UMBRAL_COB, MIN_DIAS_CUB = 30, 7, 2, 0.20, 5
PRIMER_CORTE, INICIO, FIN = "2008-02-04", pd.Timestamp("2008-01-01"), pd.Timestamp("2008-12-31")
EXPANSION = pd.Timestamp("2008-11-13")
SESION_MIN = 30
TIPOS = ["ciudad", "pvp", "raid", "mazmorra", "mundo"]
CIUDADES = [
    'Dalaran',
    'Deeprun Tram',
    'Ironforge',
    'Orgrimmar',
    'Shattrath City',
    'Silvermoon City',
    'Stormwind City',
    'The Exodar',
    'Thunder Bluff',
    'Undercity',
]
PVP = [
    'Alterac Valley',
    'Arathi Basin',
    "Blade's Edge Arena",
    'Dalaran競技場',
    'Eye of the Storm',
    'Hall of Legends',
    'Nagrand Arena',
    'Ruins of Lordaeron',
    'Strand of the Ancients',
    'The Ring of Valor',
    'Warsong Gulch',
    'Wintergrasp',
]
RAIDS = [
    "Ahn'Qiraj",
    'Black Temple',
    'Blackwing Lair',
    "Gruul's Lair",
    'Hyjal',
    'Karazhan',
    "Magtheridon's Lair",
    'Molten Core',
    'Naxxramas',
    "Onyxia's Lair",
    "Ruins of Ahn'Qiraj",
    'Serpentshrine Cavern',
    'Sunwell Plateau',
    'Tempest Keep',
    'The Eye of Eternity',
    'The Obsidian Sanctum',
    'Vault of Archavon',
    "Zul'Aman",
    "Zul'Gurub",
]
MAZMORRAS = [
    "Ahn'kahet: The Old Kingdom",
    'Auchenai Crypts',
    'Auchindoun: Shadow Labyrinth',
    'Azjol-Nerub',
    'Blackfathom Deeps',
    'Blackrock Depths',
    'Blackrock Spire',
    'Coilfang: The Slave Pens',
    'Deadmines',
    'Dire Maul',
    "Drak'Tharon Keep",
    'Gnomeregan',
    'Gundrak',
    'Halls of Lightning',
    'Halls of Stone',
    'Hellfire Ramparts',
    "Magisters' Terrace",
    'Mana-Tombs',
    'Maraudon',
    'Old Hillsbrad Foothills',
    'Ragefire Chasm',
    'Razorfen Downs',
    'Razorfen Kraul',
    'Scarlet Monastery',
    'Scholomance',
    'Sethekk Halls',
    'Shadowfang Keep',
    'Stratholme',
    'The Arcatraz',
    'The Black Morass',
    'The Blood Furnace',
    'The Botanica',
    'The Culling of Stratholme',
    'The Mechanar',
    'The Nexus',
    'The Oculus',
    'The Shattered Halls',
    'The Steamvault',
    "The Temple of Atal'Hakkar",
    'The Underbog',
    'The Violet Hold',
    'Uldaman',
    'Utgarde Keep',
    'Utgarde Pinnacle',
    'Wailing Caverns',
    "Zul'Farrak",
]


def parsear():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plata", required=True)
    ap.add_argument("--salida", required=True)
    ap.add_argument("--porcentaje", type=int, default=100)
    return ap.parse_known_args()[0]


def tipo_zona():
    z = F.col("zona")
    return (F.when(z.isin(CIUDADES), "ciudad").when(z.isin(PVP), "pvp").when(z.isin(RAIDS), "raid")
             .when(z.isin(MAZMORRAS), "mazmorra").otherwise("mundo"))


def cobertura(spark, df):
    """(cubierto, min_pasada) como Series de pandas indexadas por dia. Sobre TODA la plata (incluye IDs inconsistentes)."""
    cal = pd.date_range(INICIO, FIN, freq="D")
    reg = df.groupBy(F.to_date("ts").alias("dia")).count().toPandas()
    reg = pd.Series(reg["count"].values, index=pd.to_datetime(reg["dia"])).reindex(cal, fill_value=0)
    cubierto = reg >= UMBRAL_COB * reg.median()

    seg = F.unix_timestamp("ts")
    w = Window.orderBy("ts")
    u = df.select("ts").distinct().withColumn("dif", seg - F.lag(seg).over(w))
    pas = u.filter(F.col("dif").isNull() | (F.col("dif") > 120)).select("ts")      # inicio de cada pasada
    gap = pas.withColumn("gap_min", (seg - F.lag(seg).over(Window.orderBy("ts"))) / 60)
    g = gap.groupBy(F.to_date("ts").alias("dia")).agg(F.expr("percentile(gap_min, 0.5)").alias("m")).toPandas()
    mp = pd.Series(g["m"].values, index=pd.to_datetime(g["dia"])).reindex(cal).fillna(10).clip(5, 20)
    return cubierto, mp


def tabla_cortes(cubierto):
    filas, log = [], []
    for t in pd.date_range(PRIMER_CORTE, FIN, freq="W-MON"):
        acum = cubierto.loc[t:].cumsum()
        idx = acum[acum >= HORIZONTE].index
        e = None if len(idx) == 0 else idx[0] + pd.Timedelta(days=1)
        cub_eleg = int(cubierto.loc[t - pd.Timedelta(days=ELEG_DIAS): t - pd.Timedelta(days=1)].sum())
        if e is None or e > FIN + pd.Timedelta(days=1):
            log.append((str(t.date()), "descartado: sin seguimiento futuro completo")); continue
        if cub_eleg < min(MIN_DIAS_CUB, ELEG_DIAS):
            log.append((str(t.date()), f"descartado: solo {cub_eleg} dias cubiertos antes del corte")); continue
        cov = {w: max(int(cubierto.loc[t - pd.Timedelta(days=w): t - pd.Timedelta(days=1)].sum()), 1) for w in (7, 14, 28)}
        filas.append((t.date(), e.date(), cov[7], cov[14], cov[28], 80 if t >= EXPANSION else 70))
        log.append((str(t.date()), "ok"))
    return filas, log


def main():
    a = parsear()
    t0 = time.time()
    spark = SparkSession.builder.appName("gamepulse-construir-oro").getOrCreate()
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.sparkContext.setLogLevel("WARN")

    df = spark.read.parquet(a.plata).select("char_id", "nivel", "clase", "raza", "zona", "gremio", "ts", "id_inconsistente")
    if a.porcentaje < 100:
        df = df.filter(F.abs(F.hash("char_id")) % 100 < a.porcentaje)
    cubierto, min_pasada = cobertura(spark, df)

    # ---- tabla diaria por personaje (D) y por zona (Z), sin IDs inconsistentes
    df = df.filter(~F.col("id_inconsistente")).drop("id_inconsistente")
    df = df.withColumn("dia", F.to_date("ts")).withColumn("tipo_zona", tipo_zona())
    seg = F.unix_timestamp("ts")
    wc = Window.partitionBy("char_id").orderBy("ts")
    df = df.withColumn("nueva_sesion", F.when(F.coalesce((seg - F.lag(seg).over(wc)) / 60, F.lit(1e9)) > SESION_MIN, 1).otherwise(0))
    aggs = [F.count("*").alias("obs"), F.sum("nueva_sesion").alias("sesiones"), F.max("nivel").alias("nivel"),
            F.expr("max_by(clase, ts)").alias("clase"), F.expr("max_by(raza, ts)").alias("raza"),
            F.max(F.when(F.col("gremio").isNotNull(), F.struct("ts", "gremio")))["gremio"].alias("gremio")]
    aggs += [F.sum(F.when(F.col("tipo_zona") == tp, 1).otherwise(0)).alias(f"obs_{tp}") for tp in TIPOS]
    mp = spark.createDataFrame(pd.DataFrame({"dia": [d.date() for d in min_pasada.index], "min_pasada": min_pasada.values}))
    D = df.groupBy("char_id", "dia").agg(*aggs).join(F.broadcast(mp), "dia", "left")
    D = D.withColumn("horas", F.col("obs") * F.col("min_pasada") / 60).drop("min_pasada")
    wd = Window.partitionBy("char_id").orderBy("dia")
    g_norm = F.coalesce(F.col("gremio"), F.lit(-1))
    D = (D.withColumn("subio_nivel", F.coalesce((F.col("nivel") - F.lag("nivel").over(wd)) > 0, F.lit(False)))
          .withColumn("cambio_gremio", F.coalesce(g_norm != F.lag(g_norm).over(wd), F.lit(False)))
          .withColumn("finde", F.dayofweek("dia").isin(1, 7)))
    D = D.cache()
    Z = df.groupBy("char_id", "dia", "zona").agg(F.count("*").alias("n"))

    # ---- cortes validos
    filas, log = tabla_cortes(cubierto)
    C = F.broadcast(spark.createDataFrame(filas, "t date, e date, cov7 int, cov14 int, cov28 int, tope int"))

    # ---- historia por (personaje, corte) y elegibles
    Hh = D.join(C, D["dia"] < C["t"])
    hist = Hh.groupBy("char_id", "t").agg(
        F.max("dia").alias("ult"), F.count("*").alias("ndias"), F.min("dia").alias("primer"),
        F.expr("max_by(nivel, dia)").alias("nivel_ult"), F.expr("max_by(gremio, dia)").alias("gremio_ult"),
        F.expr("max_by(clase, dia)").alias("clase_ult"), F.expr("max_by(raza, dia)").alias("raza_ult"),
        F.max(F.when(F.col("subio_nivel"), F.col("dia"))).alias("sub"))
    E = hist.filter((F.col("ult") >= F.date_sub("t", ELEG_DIAS)) & (F.col("ndias") >= MIN_DIAS_HIST)).cache()

    # ---- ventana de 28 dias de los elegibles
    W = (Hh.filter(F.col("dia") >= F.date_sub("t", 28)).join(E.select("char_id", "t"), ["char_id", "t"]))
    en = lambda w: F.col("dia") >= F.date_sub(F.col("t"), w)
    a3 = []
    for w in (7, 14, 28):
        a3 += [F.sum(F.when(en(w), 1).otherwise(0)).alias(f"d{w}"),
               F.sum(F.when(en(w), F.col("horas")).otherwise(0.0)).alias(f"h{w}"),
               F.sum(F.when(en(w), F.col("sesiones")).otherwise(0)).alias(f"s{w}")]
    a3 += [F.sum(F.when(F.col("finde"), F.col("horas")).otherwise(0.0)).alias("h_finde"),
           F.max("nivel").alias("nmax"), F.min("nivel").alias("nmin"), F.sum("obs").alias("obs_tot"),
           F.sum(F.col("cambio_gremio").cast("int")).alias("cambios_gremio_28")]
    a3 += [F.sum(f"obs_{tp}").alias(f"o_{tp}") for tp in TIPOS]
    A = W.groupBy("char_id", "t").agg(*a3)
    ws = Window.partitionBy("char_id", "t").orderBy("dia")
    S = (W.select("char_id", "t", (F.datediff("dia", F.lag("dia").over(ws))).alias("salto"))
          .groupBy("char_id", "t").agg(F.max("salto").alias("salto_max_w"), F.stddev_samp("salto").alias("salto_std_28")))

    # ---- zonas de los ultimos 28 dias
    Zw = (Z.join(C, (Z["dia"] < C["t"]) & (Z["dia"] >= F.date_sub(C["t"], 28)))
            .join(E.select("char_id", "t"), ["char_id", "t"])
            .groupBy("char_id", "t", "zona").agg(F.sum("n").alias("n")))
    Zw = Zw.withColumn("tot", F.sum("n").over(Window.partitionBy("char_id", "t")))
    ZA = Zw.groupBy("char_id", "t").agg(
        F.count("*").alias("n_zonas_28"),
        F.greatest(-F.sum((F.col("n") / F.col("tot")) * F.log(F.col("n") / F.col("tot"))), F.lit(0.0)).alias("entropia_zonas_28"))

    # ---- tamano del gremio activo (7 dias): personajes distintos por gremio y corte
    GR = (Hh.filter((F.col("dia") >= F.date_sub("t", 7)) & F.col("gremio").isNotNull())
            .groupBy(F.col("t").alias("t_g"), F.col("gremio").alias("gremio_g")).agg(F.countDistinct("char_id").alias("tamano_gremio_activo_7")))

    # ---- etiqueta: aparece algun dia en [corte, fin de ventana)
    fut = (D.join(C, (D["dia"] >= C["t"]) & (D["dia"] < C["e"])).select("char_id", "t").distinct().withColumn("activo", F.lit(1)))

    X = (E.join(A, ["char_id", "t"]).join(S, ["char_id", "t"], "left").join(ZA, ["char_id", "t"], "left")
          .join(GR, (F.col("t") == F.col("t_g")) & (F.col("gremio_ult") == F.col("gremio_g")), "left").drop("t_g", "gremio_g")
          .join(fut, ["char_id", "t"], "left").join(C, "t"))
    c = lambda w: F.col(f"cov{w}")
    cov_prev = F.greatest(c(28) - c(7), F.lit(1))
    out = X.select(
        "char_id", F.col("t").cast("timestamp").alias("fecha_corte"),
        F.datediff("t", "ult").alias("dias_desde_ultima"), F.datediff("t", "primer").alias("antiguedad_dias"),
        (F.col("primer") <= F.lit("2008-01-07").cast("date")).cast("int").alias("antiguedad_censurada"),
        F.col("ndias").alias("dias_activos_hist"),
        *[F.least(F.col(f"d{w}") / c(w), F.lit(1.0)).alias(f"frac_dias_activos_{w}") for w in (7, 14, 28)],
        *[x for w in (7, 14, 28) for x in (
            (F.col(f"h{w}") * w / c(w)).alias(f"horas_{w}"), (F.col(f"s{w}") * w / c(w)).alias(f"sesiones_{w}"))],
        ((F.col("h7") / c(7) + 0.1) / ((F.col("h28") - F.col("h7")) / cov_prev + 0.1)).alias("tendencia_horas"),
        ((F.col("d7") / c(7) + 0.05) / ((F.col("d28") - F.col("d7")) / cov_prev + 0.05)).alias("tendencia_dias"),
        (F.col("h28") / F.greatest(F.col("s28"), F.lit(1))).alias("horas_por_sesion_28"),
        (F.col("h28") / F.col("d28")).alias("horas_por_dia_activo_28"),
        (F.col("h_finde") / F.col("h28")).alias("frac_finde_28"),
        F.greatest(F.coalesce(F.col("salto_max_w"), F.lit(0)), F.datediff("t", "ult")).alias("salto_max_28"),
        F.coalesce(F.col("salto_std_28"), F.lit(0.0)).alias("salto_std_28"),
        F.col("nivel_ult").alias("nivel"), (F.col("tope") - F.col("nivel_ult")).alias("distancia_al_tope"),
        (F.col("nivel_ult") >= F.col("tope")).cast("int").alias("en_tope"),
        (F.col("nmax") - F.col("nmin")).alias("niveles_ganados_28"),
        F.coalesce(F.datediff("t", "sub"), F.datediff("t", "primer")).alias("dias_desde_subida_nivel"),
        F.coalesce(F.col("n_zonas_28"), F.lit(0)).alias("n_zonas_28"),
        F.coalesce(F.col("entropia_zonas_28"), F.lit(0.0)).alias("entropia_zonas_28"),
        *[(F.col(f"o_{tp}") / F.col("obs_tot")).alias(f"frac_{tp}_28") for tp in TIPOS],
        F.col("gremio_ult").isNotNull().cast("int").alias("tiene_gremio"), F.col("cambios_gremio_28"),
        F.coalesce(F.col("tamano_gremio_activo_7"), F.lit(0)).alias("tamano_gremio_activo_7"),
        F.col("clase_ult").alias("clase"), F.col("raza_ult").alias("raza"),
        F.when(F.col("activo").isNull(), 1).otherwise(0).alias("churn"),
        F.col("e").cast("timestamp").alias("fin_ventana_etiqueta"))

    out.coalesce(1).write.mode("overwrite").parquet(a.salida)
    res = spark.read.parquet(a.salida)
    n = res.count()
    resumen = {"filas": n, "personajes": res.select("char_id").distinct().count(),
               "cortes_validos": res.select("fecha_corte").distinct().count(),
               "tasa_churn": round(res.agg(F.avg("churn")).first()[0], 4), "variables": len(res.columns) - 4,
               "porcentaje_muestra": a.porcentaje, "segundos": round(time.time() - t0, 1),
               "cortes": [{"fecha_corte": f, "estado": s} for f, s in log]}
    print(json.dumps({k: v for k, v in resumen.items() if k != "cortes"}, indent=2))
    spark.stop()


if __name__ == "__main__":
    main()
