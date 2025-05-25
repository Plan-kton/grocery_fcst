# --- __init__.py for forecast_engine package ---

# OLS Engine
from .ols_engine import (
    fit_ols_matrix,
    predict_ols_matrix,
    fit_ols_statsmodels,
    predict_ols_statsmodels,
    fit_ols_sklearn,
    predict_ols_sklearn,
    evaluate_model,
    add_fitted_and_residuals
)

# Bayesian Engine
from .bayesian_engine import (
    fit_bayesian_regression,
    simulate_bayesian_forecasts,
    summarize_bayesian_distribution
)

# Residual Bootstrap Engine
from .bootstrap_engine import (
    simulate_bootstrap_forecasts,
    check_residual_stationarity,
    summarize_bootstrap_distribution
)

# Plotting
from .plotting import (
    plot_true_vs_predicted,
    plot_actual_vs_fitted_vs_forecast,
    plot_selected_forecasts,
    plot_bootstrap_forecast,
    plot_input_variables,
    plot_aggregate_forecast_distribution  # ✅ new
)

# Utilities
from .utils import (
    get_evaluation_metrics,
    summarize_forecast_table,
    summarize_forecast_table_with_colors,
    add_comb_and_flag,
    calculate_yoy,
    convert_weekly_to_monthly,
    convert_weekly_to_fiscal
)

# Preprocessing (if you created forecast_preprocessing.py)
from .forecast_preprocessing import (
    filter_by_date_range,
    check_missing_values,
    drop_or_impute_missing,
    detect_extreme_outliers,
    remove_outliers,
    log_transform,
    winsorize_data,
    scale_features,
    create_lagged_features,
    inject_forward_inputs
)
