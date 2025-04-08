# lagged_correlation.py

import pandas as pd
import matplotlib.pyplot as plt

def plot_lagged_correlations(df, cpi_col, ppi_cols, max_lag=12):
    """
    Plots correlation of CPI YoY with lagged PPI YoY variables.
    
    Parameters:
        df: pandas DataFrame with CPI and PPI data
        cpi_col: str, name of the CPI column
        ppi_cols: list of str, names of PPI columns
        max_lag: int, number of months to lag for analysis
    """
    # Step 0: Compute YoY % change for CPI
    df[f'{cpi_col}_yoy'] = df[cpi_col].pct_change(periods=12)
    
    # Initialize plot
    plt.figure(figsize=(12, 7))

    for ppi in ppi_cols:
        # Step 1: Compute YoY for PPI
        ppi_yoy_col = f'{ppi}_yoy'
        df[ppi_yoy_col] = df[ppi].pct_change(periods=12)

        # Step 2: Create lagged columns
        lagged_cols = []
        for lag in range(1, max_lag + 1):
            lag_col = f'{ppi}_yoy_lag{lag}'
            df[lag_col] = df[ppi_yoy_col].shift(lag)
            lagged_cols.append(lag_col)

        # Step 3: Compute correlations with CPI YoY
        corrs = df[[f'{cpi_col}_yoy'] + lagged_cols].corr()
        corr_values = corrs.loc[f'{cpi_col}_yoy', lagged_cols]

        # Step 4: Plot
        plt.plot(range(1, max_lag + 1), corr_values.values, marker='o', label=ppi)

    plt.title('Correlation of CPI YoY with Lagged PPI YoY Variables')
    plt.xlabel('Lag (months)')
    plt.ylabel('Correlation Coefficient')
    plt.grid(True)
    plt.legend(title='PPI Variables')
    plt.xticks(range(1, max_lag + 1))
    plt.tight_layout()
    plt.show()
