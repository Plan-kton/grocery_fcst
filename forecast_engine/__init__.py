# --- __init__.py for forecast_engine package ---

# Expose OLS Engine
from .ols_engine import (
    fit_linear_regression,
    predict_linear_regression,
    evaluate_model,
    add_fitted_and_residuals
)

# Expose Bayesian Engine
from .bayesian_engine import (
    fit_bayesian_regression,
    simulate_bayesian_forecasts,
    summarize_bayesian_distribution
)

# Expose Residual Bootstrap Engine
from .bootstrap_engine import (
    simulate_bootstrap_forecasts,
    check_residual_stationarity,
    summarize_bootstrap_distribution
)

# Expose Plotting
from .plotting import (
    plot_true_vs_predicted,
    plot_actual_vs_fitted_vs_forecast,
    plot_all_forecasts,
    plot_bootstrap_forecast,
    plot_input_variables
)

# Expose Utilities
from .utils import (
    get_evaluation_metrics,
    summarize_forecast_table,
    summarize_forecast_table_with_colors
)
