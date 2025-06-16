# forecast_control.py
import pandas as pd

forecast_month = '_2025_06'

# Target and input features
dep = 'grocery_sales'
ind = ['cpi_fah', 'rdi_adj', 'home_price', 'covid1', 'covid2']
dep_lag = 'grocery_sales_lag1'

# Forecast training and testing windows
from pandas import to_datetime
start_date_training = to_datetime('2004-01-01')
end_date_training = to_datetime('2025-05-01')
start_date_forecast = end_date_training + pd.DateOffset(months=1)
end_date_forecast = start_date_forecast + pd.DateOffset(months=5)


# Color for rolling forecast chart
color_map = {
    'Actuals': 'black',
    '_2025_02': 'blue',
    '_2025_05': 'red',
    '_2025_06': 'green'
}

