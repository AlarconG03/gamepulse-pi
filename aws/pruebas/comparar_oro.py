"""Compara la tabla de modelado de Spark (oro) con la de pandas, columna a columna (tolerancia 1e-9).
Uso: python aws/pruebas/comparar_oro.py <carpeta_oro_spark> <dataset_pandas.parquet>"""
import sys, glob
import pandas as pd, numpy as np
W=sys.argv[1]
a=pd.read_parquet(glob.glob(W+'/*.parquet')[0])
b=pd.read_parquet(sys.argv[2])
print('filas spark/pandas',len(a),len(b)); print('personajes',a.char_id.nunique(),b.char_id.nunique(),'cortes',a.fecha_corte.nunique(),b.fecha_corte.nunique(),'churn',a.churn.mean().round(4),b.churn.mean().round(4))
print('columnas iguales:',set(a.columns)==set(b.columns), [c for c in b.columns if c not in a.columns], [c for c in a.columns if c not in b.columns])
a['fecha_corte']=pd.to_datetime(a['fecha_corte']).astype('datetime64[ns]'); b['fecha_corte']=pd.to_datetime(b['fecha_corte']).astype('datetime64[ns]'); a['char_id']=a.char_id.astype('int64'); b['char_id']=b.char_id.astype('int64')
k=['char_id','fecha_corte']
a=a.sort_values(k).reset_index(drop=True); b=b.sort_values(k).reset_index(drop=True)
print('llaves iguales:', a[k].equals(b[k]))
if a[k].equals(b[k]):
    peor={}
    for c in b.columns:
        if c in k: continue
        if b[c].dtype==object or str(b[c].dtype) in ('category','string','str'):
            d=(a[c].astype(str)!=b[c].astype(str)).sum()
        elif 'datetime' in str(b[c].dtype):
            d=(pd.to_datetime(a[c])!=pd.to_datetime(b[c])).sum()
        else:
            x=a[c].astype(float).values; y=b[c].astype(float).values
            d=(~np.isclose(x,y,rtol=1e-9,atol=1e-9,equal_nan=True)).sum()
        if d: peor[c]=int(d)
    print('columnas con diferencias (n filas):',peor or 'NINGUNA')
