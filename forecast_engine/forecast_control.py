# forecast_control.py

# Target and input features
dep = 'grocery_sales'
ind = ['cpi_fah', 'rdi_adj', 'home_price', 'covid1', 'covid2']

# Forecast training and testing windows
start_training_date = '2004-01-01'
end_training_date = '2023-12-31'

start_test_date = '2024-01-01'
end_test_date = '2025-03-31'