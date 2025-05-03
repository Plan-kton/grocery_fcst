# --- ols_engine.py ---

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 1. Fit OLS Model

def fit_linear_regression(X, y):
    """
    Fits an Ordinary Least Squares (OLS) model.
    Returns beta coefficients.
    """
    X = np.array(X)
    y = np.array(y)

    ones = np.ones((X.shape[0], 1))
    X_augmented = np.hstack((ones, X))

    X_transpose = X_augmented.T
    XtX = X_transpose @ X_augmented
    XtX_inv = np.linalg.inv(XtX)
    Xty = X_transpose @ y
    beta = XtX_inv @ Xty

    return beta

# 2. Predict using OLS Model

def predict_linear_regression(X, beta):
    """
    Predicts using fitted OLS beta coefficients.
    """
    X = np.array(X)
    ones = np.ones((X.shape[0], 1))
    X_augmented = np.hstack((ones, X))

    y_fitted = X_augmented @ beta

    return y_fitted

# 3. Evaluate OLS Model

def evaluate_model(y, y_fitted):
    """
    Calculates and prints model evaluation metrics (MSE, MAE, R2, MAPE).
    """
    mse = np.mean((y - y_fitted) ** 2)
    mae = np.mean(np.abs(y - y_fitted))
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_fitted) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    mape = np.mean(np.abs((y - y_fitted) / y)) * 100

    print(f"Model Evaluation:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R² Score): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return mse, mae, r2, mape

# 4. Add Fitted Values and Residuals

def add_fitted_and_residuals(df, y_true, y_fitted):
    """
    Adds 'y_fitted' and 'residuals' columns to a DataFrame.
    """
    df = df.copy()

    y_fitted_series = pd.Series(y_fitted, index=y_true.index)
    df['y_fitted'] = y_fitted_series
    df['residuals'] = y_true - y_fitted_series

    return df
