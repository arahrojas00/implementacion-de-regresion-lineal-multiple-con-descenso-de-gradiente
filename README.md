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

