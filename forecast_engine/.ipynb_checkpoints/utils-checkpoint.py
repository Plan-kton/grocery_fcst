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
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, np.nan, y_true))) * 100


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
    Summarizes forecast results with renamed columns and applies color styling.
    """
    # Extract only the forecast rows
    df_forecast = df_combined.loc[forecast_years, [
        'y_fcst_ols_matrix', 
        'y_fcst_ols_statsmodels', 
        'y_fcst_bayes_mean', 'y_fcst_bayes_p5', 'y_fcst_bayes_p95',
        'y_fcst_bootstrap', 'y_fcst_bootstrap_p5', 'y_fcst_bootstrap_p95'
    ]].copy()

    # Rename for clarity
    df_forecast.columns = [
        'OLS Forecast (Matrix)', 
        'OLS Forecast (Statsmodels)', 
        'Bayes Mean', 'Bayes P5', 'Bayes P95',
        'Bootstrap Mean', 'Bootstrap P5', 'Bootstrap P95'
    ]

    # Apply color styling
    styled_table = df_forecast.style.background_gradient(
        cmap='Greens', axis=0, subset=[
            'OLS Forecast (Matrix)', 'OLS Forecast (Statsmodels)', 'Bayes Mean', 'Bootstrap Mean'
        ]
    ).background_gradient(
        cmap='Reds_r', axis=0, subset=[
            'Bayes P5', 'Bayes P95', 'Bootstrap P5', 'Bootstrap P95'
        ]
    ).format("{:,.1f}")

    return styled_table

# 4. Add y_comb and is_forecast

def add_comb_and_flag(df, dep, forecast_col='y_fcst_ols', fitted_col='y_fitted'):
    """
    Adds derived columns for modeling and analysis:
    - y_estimated: fitted + forecast values (for visualizing the model's estimated signal)
    - y_actual_or_forecast: actuals + forecast (for aggregations and YOY calcs)
    - is_forecast: boolean flag where forecast values begin
    """
    df = df.copy()

    # Fitted + Forecast = Estimated model output
    df['y_estimated'] = df[forecast_col]
    df.loc[df[forecast_col].isna(), 'y_estimated'] = df[fitted_col]

    # Actuals + Forecast = Used for aggregation and YOY
    df['y_actual_or_forecast'] = df[dep]
    df.loc[df[dep].isna(), 'y_actual_or_forecast'] = df[forecast_col]

    # Boolean flag to mark the forecast period
    df['is_forecast'] = df[fitted_col].isna() & df[forecast_col].notna()

    return df


#5. Converts weekly data to monthly

def convert_weekly_to_monthly(df, value_cols):
    """
    Aggregates weekly data into calendar months using the datetime index.

    Args:
        df (pd.DataFrame): Weekly time series with a datetime index.
        value_cols (list): Columns to sum or average.

    Returns:
        pd.DataFrame: Monthly aggregated DataFrame.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df_monthly = df[value_cols].resample('M').sum()
    return df_monthly

#6. converts_weekly to fiscal periods

def convert_weekly_to_fiscal(df, value_cols, calendar_df):
    """
    Aggregates weekly data into fiscal periods using a lookup calendar.

    Args:
        df (pd.DataFrame): Weekly data with datetime index.
        value_cols (list): Columns to aggregate.
        calendar_df (pd.DataFrame): Calendar with 'date' as index and 'fis_period_id' column.

    Returns:
        pd.DataFrame: Aggregated DataFrame by fiscal period.
    """
    df = df.copy()
    calendar_df = calendar_df.copy()

    # Ensure both are datetime indexed
    df.index = pd.to_datetime(df.index)
    calendar_df.index = pd.to_datetime(calendar_df.index)

    # Join calendar to add fiscal period mapping
    df = df.join(calendar_df[['fis_period_id']], how='left')

    # Group by fiscal period and aggregate
    df_fiscal = df.groupby('fis_period_id')[value_cols].sum()

    return df_fiscal

#7 calculate yoy values

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

