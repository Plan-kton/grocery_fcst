# --- ols_engine.py ---

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression
import statsmodels.api as sm

# --- 2. Statsmodels OLS ---
def fit_ols_statsmodels(X, y):
    """
    Fits an OLS model using statsmodels.
    Returns the fitted model and predicted values.
    """
    import statsmodels.api as sm
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    y_fitted = model.predict(X)
    return model, y_fitted

def predict_ols_statsmodels(X, model):
    X = sm.add_constant(X, has_constant='add')  # <-- Force it
    return model.predict(X)

# --- 3. Scikit-learn OLS ---
def fit_ols_sklearn(X, y):
    """
    Fits a linear regression model using scikit-learn.
    Returns the fitted model and predictions.
    """
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(X, y)
    y_fitted = model.predict(X)

    return model, y_fitted


def predict_ols_sklearn(X, model):
    return model.predict(X)

# --- 4. Evaluation ---
def evaluate_model(y, y_fitted, model_name=None):
    """
    Calculates and prints model evaluation metrics (MSE, MAE, R2, MAPE).
    Optionally prints the model name.
    """
    mse = np.mean((y - y_fitted) ** 2)
    mae = np.mean(np.abs(y - y_fitted))
    ss_total = np.sum((y - np.mean(y)) ** 2)
    ss_residual = np.sum((y - y_fitted) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    mape = np.mean(np.abs((y - y_fitted) / np.where(y == 0, np.nan, y))) * 100

    if model_name:
        print(f"\n📈 Model Evaluation: {model_name}")
    else:
        print("\n📈 Model Evaluation:")

    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R² Score): {r2:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")

    return mse, mae, r2, mape


