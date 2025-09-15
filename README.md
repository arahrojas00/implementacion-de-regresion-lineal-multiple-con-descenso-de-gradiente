# Implementacion de regresion lineal multiple con descenso de gradiente
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
Ejecutar desde la consola:

python3 ev_reg.py

Parámetros principales:

--alpha: learning rate del gradiente (ej. 0.02)
--epochs: número de iteraciones (ej. 12000)

## Salida esperada
El script imprime en consola:


## Resultados
Desempeño del modelo inicial:

- **R² (train):** 0.180
- **R² (val):** 0.123 
- **R² (test):** 0.151 
- **RMSE (train):** 9.726
- **RMSE (val):** 9.799 
- **RMSE (test):** 9.959  

## Gráfica Residuos vs Predicciones (Validación)

<img width="641" height="542" alt="Captura de pantalla 2025-09-14 a la(s) 9 48 24 p m" src="https://github.com/user-attachments/assets/bddcad1e-896f-4baa-85c5-20def71dd990" />

# Implementacion de regresion lineal multiple con descenso de gradiente mejorado

## Salida esperada
El script imprime en consola:


## Resultados
Desempeño del modelo mejorado (polinómico + L2):

- **R² (train):** 0.205 
- **R² (val):** 0.154
- **R² (test):** 0.169 
- **RMSE (train):** 9.577
- **RMSE (val):** 9.623
- **RMSE (test):** 9.851

## Gráfica Predicciones vs Valores Reales (Test)
<img width="642" height="546" alt="Captura de pantalla 2025-09-14 a la(s) 11 31 58 p m" src="https://github.com/user-attachments/assets/e0ae769b-7205-4bce-b46e-8ca6b5431e85" />

## Gráfica Comparación de desempeño (R^2)
<img width="643" height="544" alt="Captura de pantalla 2025-09-14 a la(s) 11 15 36 p m" src="https://github.com/user-attachments/assets/e13e07c7-419b-42b1-bdc0-82c52c8a04cb" />

## Gráfica Comparación de desempeño (RMSE)
<img width="640" height="542" alt="Captura de pantalla 2025-09-14 a la(s) 11 33 02 p m" src="https://github.com/user-attachments/assets/f19ea225-b043-4777-90e1-2097e5c8c4df" />
