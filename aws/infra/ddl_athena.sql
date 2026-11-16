-- Tablas externas de Athena sobre las capas plata y oro (Parquet en S3). Reemplaza BUCKET por gamepulse-<cuenta>.
-- La particion `mes` de la plata usa "partition projection": Athena la calcula sin crawlers ni MSCK REPAIR.

CREATE EXTERNAL TABLE IF NOT EXISTS gamepulse.plata_wowah (
  char_id bigint, nivel int, raza string, clase string, zona string,
  gremio int, ts timestamp, id_inconsistente boolean)
PARTITIONED BY (mes string)
STORED AS PARQUET
LOCATION 's3://BUCKET/plata/wowah/'
TBLPROPERTIES (
  'projection.enabled'='true',
  'projection.mes.type'='date', 'projection.mes.format'='yyyy-MM',
  'projection.mes.range'='2008-01,2008-12', 'projection.mes.interval'='1', 'projection.mes.interval.unit'='MONTHS');

CREATE EXTERNAL TABLE IF NOT EXISTS gamepulse.rechazadas_wowah (
  char_id string, nivel string, raza string, clase string, zona string, gremio string, ts string, motivo string)
STORED AS PARQUET
LOCATION 's3://BUCKET/rechazadas/wowah/';

CREATE EXTERNAL TABLE IF NOT EXISTS gamepulse.dataset_modelado (
  char_id bigint, fecha_corte timestamp, dias_desde_ultima int, antiguedad_dias int, antiguedad_censurada int,
  dias_activos_hist int,
  frac_dias_activos_7 double, frac_dias_activos_14 double, frac_dias_activos_28 double,
  horas_7 double, sesiones_7 double, horas_14 double, sesiones_14 double, horas_28 double, sesiones_28 double,
  tendencia_horas double, tendencia_dias double, horas_por_sesion_28 double, horas_por_dia_activo_28 double,
  frac_finde_28 double, salto_max_28 int, salto_std_28 double,
  nivel int, distancia_al_tope int, en_tope int, niveles_ganados_28 int, dias_desde_subida_nivel int,
  n_zonas_28 int, entropia_zonas_28 double,
  frac_ciudad_28 double, frac_pvp_28 double, frac_raid_28 double, frac_mazmorra_28 double, frac_mundo_28 double,
  tiene_gremio int, cambios_gremio_28 int, tamano_gremio_activo_7 int, clase string, raza string,
  churn int, fin_ventana_etiqueta timestamp)
STORED AS PARQUET
LOCATION 's3://BUCKET/oro/dataset_modelado/';

-- Controles (resultados esperados: 10.823.177 / 156.877)
-- SELECT count(*), count(distinct char_id), count_if(id_inconsistente and mes is not null) FROM gamepulse.plata_wowah;
-- SELECT count(*), count(distinct char_id), count(distinct fecha_corte), avg(churn) FROM gamepulse.dataset_modelado;

-- Resultados del modelo (publicados desde la corrida de referencia, ver aws/api/preparar_publicacion.py)
CREATE EXTERNAL TABLE IF NOT EXISTS gamepulse.scores (
  char_id bigint, fecha_corte timestamp, churn int, riesgo double, segmento int,
  riesgo_regresion_logistica double, riesgo_random_forest double, riesgo_lightgbm double,
  senal_1 string, senal_2 string, senal_3 string, accion_sugerida string, prioridad_top_k boolean)
STORED AS PARQUET
LOCATION 's3://BUCKET/oro/scores/';

CREATE EXTERNAL TABLE IF NOT EXISTS gamepulse.perfiles_segmento (
  segmento bigint, perfil string, jugadores_prueba bigint, churn_real double, riesgo_medio double,
  pct_en_top_riesgo double, prioridad string, senal_principal_modelo string, rasgos_diferenciales string,
  accion_posible string, como_evaluarla string, naturaleza string)
STORED AS PARQUET
LOCATION 's3://BUCKET/oro/perfiles/';

CREATE EXTERNAL TABLE IF NOT EXISTS gamepulse.shap_global (variable string, mean_abs_shap double)
STORED AS PARQUET
LOCATION 's3://BUCKET/oro/shap_global/';
