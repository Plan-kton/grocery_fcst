# --- utils.py ---

import numpy as np
import pandas as pd

# Get Evaluation Metrics

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
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100


    return {
        'MSE': mse,
        'MAE': mae,
        'R²': r2,
        'MAPE (%)': mape
    }

# Summarize Forecast Table (basic)

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

# calculate yoy values

def calculate_yoy(df, value_cols, periods=12, suffix='_yoy'):
    """
    Adds Year-over-Year (YOY) percent change columns to the DataFrame.

    Args:
        df (pd.DataFrame): Time-indexed DataFrame (monthly or fiscal).
        value_cols (list): List of column names to compute YOY for.
        periods (int): Number of periods to look back (12 for months, 13 for fiscal periods).
        suffix (str): Suffix to append to YOY columns.

    Returns:
        pd.DataFrame: DataFrame with added YOY columns.
    """
    df = df.copy()

    for col in value_cols:
        yoy_col = f"{col}{suffix}"
        df[yoy_col] = df[col].pct_change(periods=periods) * 100

    return df

