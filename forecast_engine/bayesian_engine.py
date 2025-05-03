# --- bayesian_engine.py ---

import pandas as pd
import pymc as pm
import numpy as np
import aesara.tensor as at 

# 1. Fit Bayesian Regression Model
def fit_bayesian_regression(X, y):
    """
    Fits a Bayesian linear regression model using PyMC v4.
    Returns the trace of sampled posterior distributions.
    """
    X = np.array(X)
    y = np.array(y)
    n, k = X.shape

    with pm.Model() as model:
        # Priors
        intercept = pm.Normal("intercept", mu=0, sigma=10)
        coefs = pm.Normal("coefs", mu=0, sigma=10, shape=k)
        sigma = pm.HalfNormal("sigma", sigma=2)

        # Linear predictor
        mu = intercept + at.dot(X, coefs)

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Sampling
        trace = pm.sample(
            draws=2000,
            tune=1000,
            target_accept=0.95,
            chains=2,
            cores=1
        )

    return trace

# 2. Simulate Forecasts from Bayesian Model
def simulate_bayesian_forecasts(X_future_array, trace):
    """
    Simulates forecasts from the Bayesian posterior samples.
    Returns a matrix of simulated future forecasts.
    """
    intercept_samples = trace['intercept']
    coefs_samples = trace['coefs']
    
    n_simulations = len(intercept_samples)
    n_forecast_periods = X_future_array.shape[0]
    
    simulated_forecasts = np.zeros((n_simulations, n_forecast_periods))

    for i in range(n_simulations):
        simulated_forecasts[i, :] = intercept_samples[i] + np.dot(X_future_array, coefs_samples[i, :])
    
    return simulated_forecasts

# 3. Summarize Bayesian Forecast Distribution
def summarize_bayesian_distribution(simulated_forecasts):
    """
    Summarizes simulated forecasts into mean, 5th percentile, and 95th percentile.
    Returns a pandas DataFrame.
    """
    summary = pd.DataFrame({
        'mean': simulated_forecasts.mean(axis=0),
        'p5': np.percentile(simulated_forecasts, 5, axis=0),
        'p95': np.percentile(simulated_forecasts, 95, axis=0)
    })
    return summary
