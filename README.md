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

## Salida esperada
El script imprime en consola:

==== Resultados ====
  R2_train: 0.157904
   R2_test: 0.179825
RMSE_train: 9.845184
 RMSE_test: 9.625026

Ejemplos de predicciones (test):
y_true=18.00800 | y_pred=32.30054
y_true=33.71400 | y_pred=22.57472
y_true=25.75200 | y_pred=33.14459
y_true=27.57600 | y_pred=39.45165
y_true=24.65700 | y_pred=37.79853
y_true=18.18000 | y_pred=23.71207
y_true=42.67300 | y_pred=25.87697
y_true=19.62100 | y_pred=23.85219
y_true=21.89300 | y_pred=32.14614
y_true=26.86800 | y_pred=25.21116
