# --- bootstrap_engine.py ---

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 1. Check Stationarity (Dickey-Fuller Test)

def check_residual_stationarity(residuals, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller test to check if residuals are stationary.
    Prints result.
    """
    adf_stat, p_value, _, _, _, _ = adfuller(residuals)
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    if p_value < alpha:
        print("Residuals are stationary (reject null hypothesis). ✅")
    else:
        print("Residuals are NOT stationary (fail to reject null hypothesis). ⚠️")

# 2. Simulate Bootstrap Forecasts

def simulate_bootstrap_forecasts(X_future, beta, residuals, n_simulations=4000):
    """
    Simulate future forecasts by randomly sampling historical residuals.
    """
    n_forecasts = X_future.shape[0]
    forecasts = np.zeros((n_simulations, n_forecasts))

    X_augmented = np.hstack((np.ones((n_forecasts, 1)), X_future))
    y_pred_base = X_augmented @ beta

    for i in range(n_simulations):
        sampled_residuals = np.random.choice(residuals, size=n_forecasts, replace=True)
        forecasts[i, :] = y_pred_base + sampled_residuals

    return forecasts

# 3. Summarize Bootstrap Simulations

def summarize_bootstrap_distribution(simulations):
    """
    Summarizes bootstrap forecast simulations into mean, p5, and p95.
    Returns a DataFrame.
    """
    mean = np.mean(simulations, axis=0)
    p5 = np.percentile(simulations, 5, axis=0)
    p95 = np.percentile(simulations, 95, axis=0)

    summary = pd.DataFrame({
        'mean': mean,
        'p5': p5,
        'p95': p95
    })

    return summary
