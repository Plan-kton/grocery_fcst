# --- plotting.py ---

import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from scipy.stats import zscore

# 1. Plot True vs Predicted (Scatter)

def plot_true_vs_predicted(y, y_fitted, title="True vs Predicted"):
    """
    Scatter plot of true vs predicted values with R-squared.
    """
    from sklearn.linear_model import LinearRegression
    import matplotlib.pyplot as plt
    import pandas as pd

    y = pd.Series(y)
    y_fitted = pd.Series(y_fitted)

    model = LinearRegression()
    model.fit(y.values.reshape(-1, 1), y_fitted.values)
    r2 = model.score(y.values.reshape(-1, 1), y_fitted.values)
    y_fit_line = model.predict(y.values.reshape(-1, 1))

    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_fitted, color='blue', label='Predicted vs True')
    plt.plot(y, y_fit_line, color='green', linewidth=2, label=f'Fit Line (R² = {r2:.3f})')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 2. Plot Actuals vs Fitted vs Forecast

def plot_actual_vs_fitted_vs_forecast(df_combined, dep):
    """
    Line plot of Actuals, Fitted, and Forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_combined.index, df_combined[dep], label='Actuals')
    plt.plot(df_combined.index, df_combined['y_fitted'], label='Fitted', linestyle='--')

    last_fitted_date = df_combined['y_fitted'].dropna().index.max()
    plt.axvline(x=last_fitted_date, color='red', linestyle='--', label='Forecast Start')

    plt.title('Actuals, Fitted, and Forecast')
    plt.xlabel('Year')
    plt.ylabel(dep)
    plt.grid(True)
    plt.legend()
    plt.show()

# 3. Plot All Forecasts Together

def plot_selected_forecasts(df_combined, dep):
    """
    Plots Actuals, OLS Fitted (matrix), and OLS/Bayes/Bootstrap forecasts for the last 2 years.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    # Ensure datetime index
    df_combined = df_combined.copy()
    df_combined.index = pd.to_datetime(df_combined.index)

    # Filter to last 2 years
    last_2_years = df_combined.index.max() - pd.DateOffset(years=2)
    df_plot = df_combined[df_combined.index >= last_2_years]

    # Safely select expected columns
    cols = [dep, 'y_fitted', 'y_fcst_ols', 'y_fcst_bayes_mean', 'y_fcst_bootstrap', 'y_fcst_ols_statsmodels']
    df_plot = df_plot[[col for col in cols if col in df_plot.columns]]

    # Begin plotting
    plt.figure(figsize=(12, 6))

    if dep in df_plot:
        plt.plot(df_plot.index, df_plot[dep], label='Actuals', color='black')
    if 'y_fitted' in df_plot:
        plt.plot(df_plot.index, df_plot['y_fitted'], label='OLS Fitted', linestyle='--', color='gray')
    if 'y_fcst_ols' in df_plot:
        plt.plot(df_plot.index, df_plot['y_fcst_ols'], label='OLS Forecast', linestyle='-', color='blue')
    if 'y_fcst_bayes_mean' in df_plot:
        plt.plot(df_plot.index, df_plot['y_fcst_bayes_mean'], label='Bayes Forecast', linestyle='-', color='orange')
    if 'y_fcst_bootstrap' in df_plot:
        plt.plot(df_plot.index, df_plot['y_fcst_bootstrap'], label='Bootstrap Forecast', linestyle='-', color='green')
    if 'y_fcst_ols_statsmodels' in df_plot:
        plt.plot(df_plot.index, df_plot['y_fcst_ols_statsmodels'], label='Statsmodels Forecast', linestyle='-', color='purple')

    # Forecast start marker
    forecast_start = df_combined['y_fitted'].dropna().index.max()
    if forecast_start >= last_2_years:
        plt.axvline(x=forecast_start, color='red', linestyle='--', label='Forecast Start')

    # Final formatting
    plt.title('Actuals, OLS Fitted (Matrix), and Forecasts (Last 2 Years)')
    plt.xlabel('Date')
    plt.ylabel(dep)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

# 4. Plot Bootstrap Forecasts
def plot_bootstrap_forecast(df_combined, dep):
    """
    Plots Bootstrap mean and prediction interval.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_combined.index, df_combined[dep], label='Actuals', marker='o')
    plt.plot(df_combined.index, df_combined['y_fcst_bootstrap'], label='Bootstrap Forecast (Mean)', linestyle='--', color='green')
    plt.fill_between(
        df_combined.index,
        df_combined['y_fcst_bootstrap_p5'],
        df_combined['y_fcst_bootstrap_p95'],
        color='lightgreen', alpha=0.4, label='Bootstrap 90% PI')
    forecast_start_date = df_combined['y_fitted'].dropna().index.max()
    plt.axvline(x=forecast_start_date, color='red', linestyle='--', label='Forecast Start')
    plt.title('Residual Bootstrap Forecast')
    plt.xlabel('Year')
    plt.ylabel(dep)
    plt.grid(True)
    plt.legend()
    plt.show()

# 5. Plot Input Variables (X's)

def plot_input_variables(df, input_vars, start_date=None, end_date=None):
    """
    Creates a grid of time series plots for input variables.
    """
    if start_date:
        df = df[df.index >= start_date]
    if end_date:
        df = df[df.index <= end_date]

    n_vars = len(input_vars)
    ncols = 2
    nrows = math.ceil(n_vars / ncols)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, var in enumerate(input_vars):
        ax = axes[i]
        ax.plot(df.index, df[var], marker='o')
        ax.set_title(var)
        ax.set_xlabel('Year')
        ax.grid(True)

    for j in range(len(input_vars), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

# New: Plot Aggregate Forecast Distribution
def plot_aggregate_forecast_distribution(
    total_mean,
    total_p5,
    total_p95,
    yoy_mean=None,
    yoy_p5=None,
    yoy_p95=None,
    title="Aggregate Forecast Distribution",
    unit_label="Units",
    yoy_label="YOY (%)"
):
    """
    Plots aggregate forecast summary with error bars for total and optional YOY.

    Args:
        total_mean (float): Mean of total forecast (e.g., units)
        total_p5 (float): 5th percentile of total forecast
        total_p95 (float): 95th percentile of total forecast
        yoy_mean (float): Mean YOY percent (optional)
        yoy_p5 (float): 5th percentile YOY percent (optional)
        yoy_p95 (float): 95th percentile YOY percent (optional)
        title (str): Plot title
        unit_label (str): Label for total units axis
        yoy_label (str): Label for YOY axis
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar positions
    x = [0]
    labels = [unit_label]
    values = [total_mean]
    yerr = [[total_mean - total_p5], [total_p95 - total_mean]]

    if yoy_mean is not None:
        x.append(1)
        labels.append(yoy_label)
        values.append(yoy_mean)
        yerr[0].append(yoy_mean - yoy_p5)
        yerr[1].append(yoy_p95 - yoy_mean)

    ax.bar(x, values, yerr=yerr, capsize=10, color=['steelblue', 'orange'][:len(values)])

    # Label each bar with exact value
    for i, val in enumerate(values):
        ax.text(i, val, f"{val:,.1f}" + ("%" if i == 1 else ""), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

