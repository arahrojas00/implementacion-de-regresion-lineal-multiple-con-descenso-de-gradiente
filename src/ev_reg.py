#!/usr/bin/env python3
# coding: utf-8
"""
Regresión lineal múltiple DESDE CERO (descenso de gradiente)
- Sin frameworks de ML ni de estadística avanzada.

Ejemplo de entrada:
python3 ev_reg.py \
  --data parkinsons_updrs.data \
  --target total_UPDRS \
  --features "age" "test_time" "Jitter(%)" "Jitter(Abs)" "Jitter:RAP" "Jitter:PPQ5" "Jitter:DDP" \
             "Shimmer" "Shimmer(dB)" "Shimmer:APQ3" "Shimmer:APQ5" "Shimmer:APQ11" "Shimmer:DDA" \
             "NHR" "HNR" "RPDE" "DFA" "PPE" \
  --alpha 0.02 --epochs 12000 --seed 7 --test-size 0.3
"""

import argparse
import numpy as np
import pandas as pd

# --------- Limpieza de Datos ---------

def load_with_pandas(filepath, feature_names, target_name):
    df = pd.read_csv(filepath)
    if target_name not in df.columns:
        raise ValueError(f"Target '{target_name}' no está en el archivo. Columnas: {list(df.columns)}")
    if not feature_names:
        feature_names = [c for c in df.columns if c != target_name]
    else:
        faltan = [c for c in feature_names if c not in df.columns]
        if faltan:
            raise ValueError(f"Features inexistentes: {faltan}")
    cols = feature_names + [target_name]
    df = df[cols].copy()
    for c in cols:  # fuerza a numérico y limpia
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()
    X = df[feature_names].to_numpy(dtype=float)
    y = df[target_name].to_numpy(dtype=float)
    return X, y

def train_test_split(X, y, test_size=0.2, seed=42):
    n = X.shape[0]
    idx = np.arange(n)
    np.random.default_rng(seed).shuffle(idx)
    t = max(1, int(round(test_size * n)))
    test_idx, train_idx = idx[:t], idx[t:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# --------- StandardScaler ---------

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

# --------- Modelo desde cero ---------

class LinearRegressionGD:
    def __init__(self, alpha=0.01, epochs=5000):
        self.alpha = float(alpha)
        self.epochs = int(epochs)
        self.w = None   # pesos (d,)
        self.b = 0.0    # bias escalar

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d, dtype=float)
        self.b = 0.0
        for _ in range(self.epochs):
            y_pred = X @ self.w + self.b
            err = y_pred - y
            grad_w = (X.T @ err) / n
            grad_b = np.sum(err) / n
            self.w -= self.alpha * grad_w
            self.b -= self.alpha * grad_b
        return self

    def predict(self, X):
        if self.w is None:
            raise RuntimeError("Modelo no entrenado. Llama a fit() antes.")
        return X @ self.w + self.b

# --------- Métricas ---------

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2_score(y_true, y_pred):
    y_mean = float(np.mean(y_true))
    ss_tot = float(np.sum((y_true - y_mean) ** 2))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0

# --------- CLI ---------

def parse_args():
    p = argparse.ArgumentParser(description="Regresión lineal múltiple con descenso de gradiente (desde cero).")
    p.add_argument("--data", required=True, type=str, help="Ruta al archivo .data/.csv con encabezados")
    p.add_argument("--target", required=True, type=str, help="Nombre de la columna objetivo")
    p.add_argument("--features", nargs="*", help="Nombres de columnas para X")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()

    # Datos
    X, y = load_with_pandas(args.data, args.features, args.target)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, seed=args.seed
    )

    # Estandarización
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Modelo (LinearRegressionGD)
    model = LinearRegressionGD(alpha=args.alpha, epochs=args.epochs)
    model.fit(X_train, y_train)

    # Evaluación
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    print("\n=== Resultados ===")
    print(f"  R2_train: {r2_score(y_train, y_pred_train):.6f}")
    print(f"   R2_test: {r2_score(y_test, y_pred_test):.6f}")
    print(f"RMSE_train: {rmse(y_train, y_pred_train):.6f}")
    print(f" RMSE_test: {rmse(y_test, y_pred_test):.6f}")

    print("\nEjemplos de predicciones (test):")
    for yt, yp in list(zip(y_test, y_pred_test))[:10]:
        print(f"y_true={yt:.5f} | y_pred={yp:.5f}")

if __name__ == "__main__":
    main()
