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

def simulate_bootstrap_forecasts(X_future, model_or_beta, residuals, n_simulations=4000, random_state=None):
    """
    Simulate future forecasts by randomly sampling historical residuals.
    """
    if random_state is not None:
        np.random.seed(random_state)
        
    # --- Extract beta from model if needed ---
    if hasattr(model_or_beta, "params"):
        beta = model_or_beta.params.values
    else:
        beta = model_or_beta

    n_forecasts = X_future.shape[0] #is the number of rows or periods
    
    # ✅ Shape check for X and beta - compares the number of columns (ind vars) with the number of beta coefficients
    assert X_future.shape[1] + 1 == beta.shape[0], \
        f"Shape mismatch: beta expects {beta.shape[0]-1} features, but got {X_future.shape[1]}."

    forecasts = np.zeros((n_simulations, n_forecasts)) # this just initializes a matrix based on sims and periods and fills them with )

    # Add constant term
    X_augmented = np.hstack((np.ones((n_forecasts, 1)), X_future)) # this augments 1s to the matrix for the constant. Stack merges the vector to the left
    y_pred_base = X_augmented @ beta # This is matrix multiplication to the betas and values

    for i in range(n_simulations): #4000 simulation to loop through
        sampled_residuals = np.random.choice(residuals, size=n_forecasts, replace=True) #randomnly chooses the residuals, one for each period
        forecasts[i, :] = y_pred_base + sampled_residuals # add the residual to the base to create the simulated y. 
        # This results in a (4000, 8) matrix which is 4000 different y estimates for each period.  These 4000 estimates give us the distribution
        # the forecasts[i, :] - the i is for each simulation and the :] means all columns. The output is a forecast for each row and period
        # These rows are what is then used to build the distribution
        
    return forecasts


#3. Aggregate Bootstrap forecast across the n periods

def aggregate_forecast_distribution(forecast_matrix, agg_func=np.sum):
    """
    Aggregates forecast simulations across all future periods to produce a distribution of totals.

    Args:
        forecast_matrix (ndarray): Shape (n_simulations, n_periods)
        agg_func (function): Aggregation function (default is np.sum for yearly total)

    Returns:
        pd.Series: Aggregated total for each simulation (length = n_simulations)
    """
    # Apply aggregation function row-wise
    aggregated = agg_func(forecast_matrix, axis=1)
    
    # Return as Series for easier summary or plotting
    return pd.Series(aggregated, name='y_fcst_total')



