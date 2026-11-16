# Validación local de los Jobs (PySpark 4 en 4 núcleos, 15 GB)

- `bronce_a_plata.py` sobre un CSV de bronce de 10.823.289 filas (la plata real más 112 anomalías inyectadas): 10.823.177 en plata, 112 rechazadas (100 duplicados R5, 5 nivel R2, 3 zona vacía R4, 2 fecha R3, 2 ID R1), 37.354 personajes, 819 IDs inconsistentes y 1.693.240 filas sin gremio, igual que la plata original. Cuadre: cumple. Tiempo: 157 s.
- `construir_oro.py` sobre esa plata: 156.877 filas, 18.440 personajes, 41 cortes, 14,08 % de `y = 1` y 37 variables; **idéntica a la tabla de pandas en todas las columnas** (`comparar_oro.py`, tolerancia 1e-9). Tiempo: 90 s.
