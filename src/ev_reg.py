import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar datos y seleccionar características
df = pd.read_csv("parkinsons_updrs.data")  # archivo con encabezados
target = "total_UPDRS"
# Excluir columnas no deseadas
cols_excluir = ["subject#", "motor_UPDRS", "sex", target]
features = [c for c in df.columns if c not in cols_excluir]
X = df[features].to_numpy(dtype=float)
y = df[target].to_numpy(dtype=float)

# Dividir en train/validation/test (60/20/20)
np.random.seed(7)
n = X.shape[0]
indices = np.arange(n)
np.random.shuffle(indices)
train_end = int(0.6 * n)
val_end = int(0.8 * n)
train_idx, val_idx, test_idx = indices[:train_end], indices[train_end:val_end], indices[val_end:]
X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

# Estandarizar características (usar media/desvío de entrenamiento)
class StandardScaler:
    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.std_ = np.where(std == 0, 1.0, std)
        return self
    def transform(self, X):
        return (X - self.mean_) / self.std_
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Modelo de Regresión Lineal con descenso de gradiente
class LinearRegressionGD:
    def __init__(self, alpha=0.01, epochs=5000, l2=0.0):
        self.alpha = float(alpha)
        self.epochs = int(epochs)
        self.l2 = float(l2)     # parámetro de regularización L2 (Ridge)
        self.w = None
        self.b = 0.0
    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            err = y_pred - y
            # Gradientes (incluye término de regularización L2 en w)
            grad_w = (X.T @ err) / n + self.l2 * self.w / n
            grad_b = np.sum(err) / n
            # Actualizar parámetros
            self.w -= self.alpha * grad_w
            self.b -= self.alpha * grad_b
        return self

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        return X @ self.w + self.b

# Funciones de métricas
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    y_mean = np.mean(y_true)
    ss_tot = np.sum((y_true - y_mean) ** 2)
    ss_res = np.sum((y_true - y_pred) ** 2)
    return 1.0 - ss_res/ss_tot if ss_tot != 0 else 0.0

# Entrenar modelo lineal inicial (sin regularización, l2=0)
model_initial = LinearRegressionGD(alpha=0.02, epochs=12000, l2=0.0)
model_initial.fit(X_train_scaled, y_train)

# Evaluar en train, validation y test
y_pred_train = model_initial.predict(X_train_scaled)
y_pred_val = model_initial.predict(X_val_scaled)
y_pred_test = model_initial.predict(X_test_scaled)

print("Desempeño del modelo inicial:")
print(f"R2_train = {r2_score(y_train, y_pred_train):.3f}   RMSE_train = {rmse(y_train, y_pred_train):.3f}")
print(f"R2_val   = {r2_score(y_val, y_pred_val):.3f}   RMSE_val   = {rmse(y_val, y_pred_val):.3f}")
print(f"R2_test  = {r2_score(y_test, y_pred_test):.3f}   RMSE_test  = {rmse(y_test, y_pred_test):.3f}")

# Diagnóstico de sesgo: gráfico de residuos
residuos = y_val - y_pred_val
plt.scatter(y_pred_val, residuos, s=12, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicciones (ŷ)")
plt.ylabel("Residuos (y - ŷ)")
plt.title("Residuos vs Predicciones (Validación)")
plt.show()
