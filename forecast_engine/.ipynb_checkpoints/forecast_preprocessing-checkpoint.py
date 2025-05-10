# forecast_preprocessing.py

import pandas as pd
import numpy as np
from scipy.stats import zscore

# 1. Filter data by date range
def filter_by_date_range(df, start_date, end_date):
    """
    Filters the DataFrame to rows within the start and end date (inclusive).
    Assumes datetime index.
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    return df.loc[start_date:end_date]

# 2. Check for missing values
def check_missing_values(df):
    """
    Prints count of missing values per column.
    Returns True if any missing values exist.
    """
    missing = df.isna().sum()
    print("\n🧪 Missing Values:")
    print(missing[missing > 0])
    return missing.any()

# 3. Handle missing values
def drop_or_impute_missing(df, method='drop'):
    """
    Handles missing values via drop or mean imputation.
    """
    df = df.copy()
    if method == 'drop':
        df = df.dropna()
    elif method == 'mean':
        df = df.fillna(df.mean(numeric_only=True))
    else:
        raise ValueError("method must be 'drop' or 'mean'")
    return df

# 4. Detect extreme outliers
def detect_extreme_outliers(df, cols, z_thresh=4.0):
    """
    Detects extreme outliers using z-score threshold.
    Returns a boolean mask for rows considered outliers.
    """
    df = df.copy()
    z = df[cols].apply(zscore)
    outlier_mask = (z.abs() > z_thresh).any(axis=1)
    print(f"\n⚠️ Outliers Detected: {outlier_mask.sum()} rows exceed z-score threshold of {z_thresh}")
    return outlier_mask

# 5. Remove outliers
def remove_outliers(df, outlier_mask):
    """
    Removes rows flagged as outliers.
    """
    return df[~outlier_mask]

# 6. Log-transform skewed data
def log_transform(df, cols):
    """
    Applies log(1 + x) transformation to selected columns.
    """
    df = df.copy()
    for col in cols:
        df[col] = np.log1p(df[col])
    return df

# 7. Winsorize extreme values
def winsorize_data(df, cols, lower_quantile=0.01, upper_quantile=0.99):
    """
    Caps values at specified lower and upper quantiles.
    """
    df = df.copy()
    for col in cols:
        lower = df[col].quantile(lower_quantile)
        upper = df[col].quantile(upper_quantile)
        df[col] = df[col].clip(lower, upper)
    return df

# 8. Normalize or scale columns
def scale_features(df, cols, method='standard'):
    """
    Scales columns using either 'standard' (z-score) or 'minmax' normalization.
    """
    df = df.copy()
    if method == 'standard':
        df[cols] = df[cols].apply(zscore)
    elif method == 'minmax':
        for col in cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[col] = (df[col] - min_val) / (max_val - min_val)
    else:
        raise ValueError("method must be 'standard' or 'minmax'")
    return df

# 9. Create lagged features
def create_lagged_features(df, cols, lags=[1]):
    """
    Creates lagged versions of selected columns.
    """
    df = df.copy()
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df

