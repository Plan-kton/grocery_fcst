import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# 1. Check Stationarity (Dickey-Fuller Test)

def check_residual_stationarity(residuals, alpha=0.05):
    """
    Perform Augmented Dickey-Fuller test to check if residuals are stationary.
    Prints result and returns True/False.
    """
    adf_stat, p_value, _, _, _, _ = adfuller(residuals)
    print(f"ADF Statistic: {adf_stat:.4f}")
    print(f"p-value: {p_value:.4f}")

    is_stationary = p_value < alpha
    if is_stationary:
        print("Residuals are stationary (reject null hypothesis). ✅")
    else:
        print("Residuals are NOT stationary (fail to reject null hypothesis). ⚠️")
    
    return is_stationary

# 2. Simulate Bootstrap Forecasts

def simulate_bootstrap_forecasts(X_future, beta, residuals, n_simulations=4000, random_state=None):
    """
    Simulate future forecasts by randomly sampling historical residuals.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_forecasts = X_future.shape[0]
    
    # ✅ Shape check for X and beta
    assert X_future.shape[1] + 1 == beta.shape[0], \
        f"Shape mismatch: beta expects {beta.shape[0]-1} features, but got {X_future.shape[1]}."

    forecasts = np.zeros((n_simulations, n_forecasts))

    # Add constant term
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
