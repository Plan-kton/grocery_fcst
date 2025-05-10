# --- plotting.py ---

import matplotlib.pyplot as plt
import pandas as pd
import math

# 1. Plot True vs Predicted (Scatter)

def plot_true_vs_predicted(y, y_fitted):
    """
    Scatter plot of true vs predicted values with R-squared.
    """
    from sklearn.linear_model import LinearRegression

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
    plt.title('True vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.show()

# 2. Plot Actuals vs Fitted vs Forecast

def plot_actual_vs_fitted_vs_forecast(df_combined, dep):
    """
    Line plot of Actuals, Fitted, and Forecast.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(df_combined.index, df_combined['y_comb'], label='Forecast', linestyle='-', linewidth=2)
    plt.plot(df_combined.index, df_combined[dep], label='Actuals', marker='o')
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

def plot_all_forecasts(df_combined, dep):
    """
    Plots Actuals, OLS, Bayes, Bootstrap forecasts together.
    """
    plt.figure(figsize=(14, 7))

    plt.plot(df_combined.index, df_combined[dep], label='Actuals', marker='o', color='black')
    plt.plot(df_combined.index, df_combined['y_comb'], label='OLS (Fitted + Forecast)', linestyle='--', color='blue')
    plt.plot(df_combined.index, df_combined['y_fcst_bayes_mean'], label='Bayes Forecast (Mean)', linestyle='-', color='orange')
    plt.fill_between(
        df_combined.index,
        df_combined['y_fcst_bayes_p5'],
        df_combined['y_fcst_bayes_p95'],
        color='orange', alpha=0.3, label='Bayes 90% CI')
    plt.plot(df_combined.index, df_combined['y_fcst_bootstrap'], label='Bootstrap Forecast (Mean)', linestyle='-', color='green')
    plt.fill_between(
        df_combined.index,
        df_combined['y_fcst_bootstrap_p5'],
        df_combined['y_fcst_bootstrap_p95'],
        color='lightgreen', alpha=0.4, label='Bootstrap 90% PI')

    forecast_start_date = df_combined['y_fitted'].dropna().index.max()
    plt.axvline(x=forecast_start_date, color='red', linestyle='--', label='Forecast Start')

    plt.title('Actuals, OLS, Bayes, and Bootstrap Forecasts')
    plt.xlabel('Year')
    plt.ylabel(dep)
    plt.grid(True)
    plt.legend()
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
