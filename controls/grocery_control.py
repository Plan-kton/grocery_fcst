# forecast_control.py
import pandas as pd

# Target and input features
dep = 'grocery_sales'
ind = ['cpi_fah', 'rdi_adj', 'home_price', 'covid1', 'covid2']

# Forecast training and testing windows
from pandas import to_datetime

start_training_date = to_datetime('2004-01-01')
end_training_date = to_datetime('2025-03-01')
start_test_date = to_datetime('2025-04-01')
end_test_date = to_datetime('2026-09-01')
