import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

# Cargar datos
df = pd.read_csv("parkinsons_updrs.data")  # archivo con encabezados
target = "total_UPDRS"
# Excluir columnas no deseadas (ID, género y otras puntuaciones UPDRS)
cols_excluir = ["subject#", "motor_UPDRS", "sex", target]
features = [c for c in df.columns if c not in cols_excluir]
X = df[features].to_numpy(dtype=float)
y = df[target].to_numpy(dtype=float)

# División en train/validation/test (60/20/20) usando scikit-learn
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=7
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=7
)

# Estandarizar características usando medias y desviaciones de entrenamiento
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# Entrenar modelo de regresión lineal (Mínimos cuadrados ordinarios)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# (Opcional) Entrenar un modelo Ridge con regularización L2
# model_ridge = Ridge(alpha=1.0)
# model_ridge.fit(X_train_scaled, y_train)

# Predicciones
y_pred_train = model.predict(X_train_scaled)
y_pred_val   = model.predict(X_val_scaled)
y_pred_test  = model.predict(X_test_scaled)

# Métricas de desempeño
rmse = lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred))

print("Desempeño del modelo con scikit-learn:")
print(f"R2_train = {r2_score(y_train, y_pred_train):.3f}   RMSE_train = {rmse(y_train, y_pred_train):.3f}")
print(f"R2_val   = {r2_score(y_val,   y_pred_val):.3f}   RMSE_val   = {rmse(y_val,   y_pred_val):.3f}")
print(f"R2_test  = {r2_score(y_test,  y_pred_test):.3f}   RMSE_test  = {rmse(y_test,  y_pred_test):.3f}")

# Diagnóstico de sesgo: gráfico de residuos vs predicciones en validación
residuos = y_val - y_pred_val
plt.scatter(y_pred_val, residuos, s=12, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicciones (ŷ)")
plt.ylabel("Residuos (y - ŷ)")
plt.title("Residuos vs Predicciones (Validación)")
plt.show()
