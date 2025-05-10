# --- utils.py ---

import numpy as np
import pandas as pd

# 1. Get Evaluation Metrics

def get_evaluation_metrics(y_true, y_pred):
    """
    Calculates evaluation metrics (MSE, MAE, R², MAPE).
    Returns a dictionary.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MSE': mse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }

# 2. Summarize Forecast Table (basic)

def summarize_forecast_table(df_combined, forecast_years):
    """
    Summarizes OLS, Bayes, and Bootstrap forecasts for selected years.
    """
    summary_table = df_combined.loc[forecast_years, [
        'y_comb',
        'y_fcst_bayes_mean', 'y_fcst_bayes_p5', 'y_fcst_bayes_p95',
        'y_fcst_bootstrap', 'y_fcst_bootstrap_p5', 'y_fcst_bootstrap_p95'
    ]].copy()

    summary_table.rename(columns={
        'y_comb': 'OLS Forecast',
        'y_fcst_bayes_mean': 'Bayes Mean',
        'y_fcst_bayes_p5': 'Bayes P5',
        'y_fcst_bayes_p95': 'Bayes P95',
        'y_fcst_bootstrap': 'Bootstrap Mean',
        'y_fcst_bootstrap_p5': 'Bootstrap P5',
        'y_fcst_bootstrap_p95': 'Bootstrap P95'
    }, inplace=True)

    summary_table = summary_table.round(1)

    return summary_table

# 3. Summarize Forecast Table with Colors

def summarize_forecast_table_with_colors(df_combined, forecast_years):
    """
    Summarizes forecasts and applies color styling for better visualization.
    """
    summary_table = summarize_forecast_table(df_combined, forecast_years)

    styled_table = summary_table.style.background_gradient(
        cmap='Greens', axis=0, subset=['OLS Forecast', 'Bayes Mean', 'Bootstrap Mean']
    ).background_gradient(
        cmap='Reds_r', axis=0, subset=['Bayes P5', 'Bayes P95', 'Bootstrap P5', 'Bootstrap P95']
    ).format("{:,.1f}")

    return styled_table
