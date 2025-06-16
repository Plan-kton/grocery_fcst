# --- plotting.py ---

import matplotlib.pyplot as plt
import seaborn as sns
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


# 2. Plot Actuals vs Fitted (Lineplot)

def plot_actual_vs_fitted_series(y_actual, y_fitted):
    plt.figure(figsize=(10, 6))
    plt.plot(y_actual.index, y_actual, label='Actuals', color='black')
    plt.plot(y_fitted.index, y_fitted, label='Fitted', linestyle='--', color='red')
    plt.title("Actuals vs Fitted")  # <-- static title
    plt.xlabel('Month')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 2. Plot Actuals vs Fitted vs Forecast (Lineplot)

def plot_actual_vs_fitted_vs_forecast(df_combined, dep, fitted_col, forecast_col):
    """
    Line plot of Actuals, Fitted, and Forecast using standardized column names:
    - Actuals: from dep
    - fitted_col: y_fitted column name
    - forecast_col: extension of fitted using y_estimated
    """
    plt.figure(figsize=(10, 6))

    # Plot actuals where y_fitted is available (training period)
    actuals = df_combined[df_combined[fitted_col].notna()]
    plt.plot(actuals.index, actuals[dep], label='Actuals')

    # Plot fitted values
    plt.plot(df_combined.index, df_combined[fitted_col], label='Fitted', linestyle='--')

    # Forecast = estimated values after the last fitted date
    forecast_start = df_combined[fitted_col].dropna().index.max()
    forecast = df_combined[df_combined.index > forecast_start]
    plt.plot(forecast.index, forecast[forecast_col], label='Forecast', linestyle=':')

    # Vertical line marking start of forecast
    plt.axvline(x=forecast_start, color='red', linestyle='--', label='Forecast Start')

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

def plot_input_variables(df, input_vars, years=None):
    """
    Creates a grid of time series plots for input variables.
    
    Parameters:
    - df: DataFrame with datetime index.
    - input_vars: list of column names to plot.
    - years: number of years back from the most recent date in the index.
    """
    if years:
        latest_date = df.index.max()
        start_date = latest_date - pd.DateOffset(years=years)
        df = df[df.index >= start_date]

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


# Plot Aggregate Forecast Distribution Histogram
def plot_aggregate_forecast_distribution(
    simulations_total,
    unit_label="Total Forecast",
    title="Forecast Distribution (Simulated)",
    bins=50
):
    """
    Plots the histogram of total simulated forecast values (e.g., sales or units).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(simulations_total, bins=bins, alpha=0.7, color='steelblue', label=unit_label)
    ax.set_title(title)
    ax.set_xlabel('Simulated Values')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Plot Aggregate Forecast Distribution Histogram for YOY values
def plot_aggregate_forecast_distribution_yoy(df, yoy_col, summary_yoy):
    
    # plot results
    plt.figure(figsize=(6, 5))
    plt.hist(df[yoy_col], bins=30, color='dimgrey', edgecolor='white')

    # Title and labels
    plt.title("Simulated Grocery Sales YOY Growth: June–Nov 2025 vs 2024")
    plt.xlabel("YOY Growth (%)")
    plt.ylabel("Simulation Count")

    # Add vertical lines for percentiles and mean
    plt.axvline(summary_yoy['p5'],  color='green', linestyle='--', label='5th Percentile')
    plt.axvline(summary_yoy['p95'], color='green', linestyle='--', label='95th Percentile')
    plt.axvline(summary_yoy['mean'], color='red', linestyle='--', label='Mean')

    # Display the grid and legend
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
# plot any two lines
def lineplot_list(df_combined, cols_list, title, ylabel):
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

    # Begin plotting
    plt.figure(figsize=(12, 6))

    for col in cols_list:
        plt.plot(df_plot.index, df_plot[col], label=f'{col}')
    
    # Final formatting
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
# plot final forecast chart
def plot_final_chart(df, color_map, y_bounds=None):
    """
    Plot Actuals and forecast lines using explicit legend handles and a custom color map.

    Parameters:
        df (pd.DataFrame): Time-indexed DataFrame with columns to plot.
        color_map (dict): Dictionary mapping column names to colors.
        y_bounds (tuple): Optional (ymin, ymax) for y-axis range.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 5))

    for col in df.columns:
        sns.lineplot(
            data=df,
            x=df.index,
            y=col,
            label=col,
            color=color_map.get(col, 'pink'),
            linewidth=2.0
        )

    plt.xlabel('Month')
    plt.ylabel('Forecasted Values')
    plt.title('CPI Food-at-Home Forecast')
    
    if y_bounds:
        plt.ylim(y_bounds)

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    