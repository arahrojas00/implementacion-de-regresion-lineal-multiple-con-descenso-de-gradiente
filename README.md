# implementacion-de-regresion-lineal-multiple-con-descenso-de-gradiente
Regresión lineal múltiple desde cero (descenso de gradiente) en Python. Entrena y predice en UCI Parkinson’s UPDRS.
## Tabla de contenidos
- [Dataset](#dataset)
- [Uso](#uso)
- [Salida esperada](#salida-esperada)
- [Cómo funciona (resumen técnico)](#como-funciona-resumen-tecnico)

## Dataset
Parkinson’s Telemonitoring (UPDRS) — UCI Machine Learning Repository.
Archivo principal: parkinsons_updrs.data
URL de referencia:
https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring

## Uso
Ejecutar desde la carpeta del proyecto:

python3 ev_reg.py \
  --data parkinsons_updrs.data \
  --target total_UPDRS \
  --features "age" "test_time" "Jitter(%)" "Jitter(Abs)" "Jitter:RAP" "Jitter:PPQ5" "Jitter:DDP" \
             "Shimmer" "Shimmer(dB)" "Shimmer:APQ3" "Shimmer:APQ5" "Shimmer:APQ11" "Shimmer:DDA" \
             "NHR" "HNR" "RPDE" "DFA" "PPE" \
  --alpha 0.02 \
  --epochs 12000 \
  --seed 7 \
  --test-size 0.3

Parámetros principales:
--data: ruta al .data/.csv con encabezados
--target: nombre de la columna objetivo (ej. total_UPDRS)
--features: nombres de columnas para X (en comillas si tienen ( o :)
--alpha: learning rate del gradiente (ej. 0.02)
--epochs: número de iteraciones (ej. 12000)
--seed: semilla para partición reproducible
--test-size: fracción para test (ej. 0.3)
