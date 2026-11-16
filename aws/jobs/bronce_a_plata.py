"""Job de AWS Glue (PySpark): capa BRONCE -> capa PLATA.

Lee el CSV de bronce (todas las columnas como texto), tipa, valida y escribe:
  * plata      : Parquet particionado por mes (yyyy-MM), con la marca `id_inconsistente`
  * rechazadas : las filas que incumplen una regla, con su motivo
  * resumen    : JSON con el cuadre bronce = plata + rechazadas

Reglas (3.2.2 del documento):
  R1 char_id nulo o no numerico | R2 nivel nulo o fuera de 1..80 | R3 ts no interpretable
  R4 zona, raza o clase vacias  | R5 duplicado por (char_id, ts): se rechaza el excedente
  T1 gremio = -1 pasa a NULL    | M1 ID con mas de una raza o clase: se MARCA, no se rechaza

Uso en Glue (parametros del Job):  --entrada s3://.../bronce/wowah/  --plata s3://.../plata/wowah/
                                   --rechazadas s3://.../rechazadas/wowah/  --resumen s3://.../plata/_cuadre/
Uso local:  python aws/jobs/bronce_a_plata.py --entrada ... --plata ... --rechazadas ... --resumen ...
"""
import argparse
import json
import sys

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

FORMATO_TS = "MM/dd/yy HH:mm:ss"
COLUMNAS_ORIGEN = {"char": "char_id", "level": "nivel", "race": "raza", "charclass": "clase",
                   "zone": "zona", "guild": "gremio", "timestamp": "ts"}


def parsear():
    ap = argparse.ArgumentParser()
    for a in ("entrada", "plata", "rechazadas", "resumen"):
        ap.add_argument(f"--{a}", required=True)
    return ap.parse_known_args()[0]          # Glue agrega --JOB_NAME y otros parametros


def leer_bronce(spark, ruta):
    raw = spark.read.option("header", "true").option("encoding", "UTF-8").csv(ruta)   # todo como texto
    raw = raw.toDF(*[c.strip() for c in raw.columns])
    for viejo, nuevo in COLUMNAS_ORIGEN.items():
        raw = raw.withColumnRenamed(viejo, nuevo)
    return raw


def transformar(raw):
    """Devuelve (plata, rechazadas). `raw` tiene las 7 columnas como texto."""
    t = raw.select(
        F.trim("char_id").cast("long").alias("char_id"), F.trim("nivel").cast("int").alias("nivel"),
        F.trim("raza").alias("raza"), F.trim("clase").alias("clase"), F.trim("zona").alias("zona"),
        F.trim("gremio").cast("int").alias("gremio"),
        F.to_timestamp(F.trim("ts"), FORMATO_TS).alias("ts"),
        *[F.col(c).alias(f"_{c}") for c in COLUMNAS_ORIGEN.values()])        # texto original, para rechazadas

    vacio = lambda c: F.col(c).isNull() | (F.col(c) == "")
    motivo = (F.when(F.col("char_id").isNull(), "R1 char_id nulo o no numerico")
               .when(F.col("nivel").isNull() | ~F.col("nivel").between(1, 80), "R2 nivel nulo o fuera de 1..80")
               .when(F.col("ts").isNull(), "R3 ts no interpretable")
               .when(vacio("zona") | vacio("raza") | vacio("clase"), "R4 zona, raza o clase vacias"))
    t = t.withColumn("motivo", motivo)

    # R5: duplicado exacto por (char_id, ts) entre las filas validas; se conserva una
    validas = t.filter(F.col("motivo").isNull())
    orden = Window.partitionBy("char_id", "ts").orderBy("nivel", "raza", "clase", "zona", "gremio")
    validas = validas.withColumn("_rn", F.row_number().over(orden))
    dups = validas.filter(F.col("_rn") > 1).withColumn("motivo", F.lit("R5 duplicado por (char_id, ts)"))
    limpio = validas.filter(F.col("_rn") == 1)

    # T1: centinela de "sin gremio"
    limpio = limpio.withColumn("gremio", F.when(F.col("gremio") == -1, F.lit(None).cast("int")).otherwise(F.col("gremio")))

    # M1: el ID aparece con mas de una raza o clase
    incons = (limpio.groupBy("char_id").agg(F.countDistinct("raza").alias("n_raza"), F.countDistinct("clase").alias("n_clase"))
                    .filter((F.col("n_raza") > 1) | (F.col("n_clase") > 1)).select("char_id").withColumn("id_inconsistente", F.lit(True)))
    plata = (limpio.join(F.broadcast(incons), "char_id", "left")
                   .withColumn("id_inconsistente", F.coalesce(F.col("id_inconsistente"), F.lit(False)))
                   .withColumn("mes", F.date_format("ts", "yyyy-MM"))
                   .select("char_id", "nivel", "raza", "clase", "zona", "gremio", "ts", "id_inconsistente", "mes"))

    rechazadas = (t.filter(F.col("motivo").isNotNull()).unionByName(dups.drop("_rn"))
                   .select(*[F.col(f"_{c}").alias(c) for c in COLUMNAS_ORIGEN.values()], "motivo"))
    return plata, rechazadas


def main():
    a = parsear()
    spark = SparkSession.builder.appName("gamepulse-bronce-a-plata").getOrCreate()
    spark.conf.set("spark.sql.ansi.enabled", "false")
    spark.conf.set("spark.sql.session.timeZone", "UTC")      # dias calendario identicos en cualquier entorno
    spark.sparkContext.setLogLevel("WARN")
    raw = leer_bronce(spark, a.entrada)
    plata, rechazadas = transformar(raw)

    plata.write.mode("overwrite").partitionBy("mes").parquet(a.plata)
    rechazadas.write.mode("overwrite").parquet(a.rechazadas)

    # cuadre leyendo lo escrito (no lo calculado en memoria)
    n_bronce = raw.count()
    p = spark.read.parquet(a.plata)
    r = spark.read.parquet(a.rechazadas)
    n_plata, n_rech = p.count(), r.count()
    resumen = {
        "filas_bronce": n_bronce, "filas_plata": n_plata, "filas_rechazadas": n_rech,
        "cuadra": n_bronce == n_plata + n_rech,
        "personajes_plata": p.select("char_id").distinct().count(),
        "ids_inconsistentes": p.filter("id_inconsistente").select("char_id").distinct().count(),
        "filas_sin_gremio": p.filter(F.col("gremio").isNull()).count(),
        "rechazadas_por_motivo": {row["motivo"]: row["n"] for row in r.groupBy("motivo").agg(F.count("*").alias("n")).collect()},
        "rango_ts": [str(x) for x in p.agg(F.min("ts"), F.max("ts")).first()],
    }
    print(json.dumps(resumen, indent=2, ensure_ascii=False))
    spark.createDataFrame([(json.dumps(resumen, ensure_ascii=False),)], ["json"]).coalesce(1).write.mode("overwrite").text(a.resumen)
    if not resumen["cuadra"]:
        sys.exit("El cuadre bronce = plata + rechazadas no se cumple")
    spark.stop()


if __name__ == "__main__":
    main()
