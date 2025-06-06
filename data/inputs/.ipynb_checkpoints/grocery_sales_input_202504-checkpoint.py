import pandas as pd

# Forward forecast months
frwd_months = [
    '2025-04-01',
    '2025-05-01',
    '2025-06-01',
    '2025-07-01',
    '2025-08-01',
    '2025-09-01'
]

# Define each variable separately
cpi_fah = {
    pd.Timestamp('2025-04-01'): 2.1,
    pd.Timestamp('2025-05-01'): 2.5,
    pd.Timestamp('2025-06-01'): 2.75,
    pd.Timestamp('2025-07-01'): 2.4,
    pd.Timestamp('2025-08-01'): 2.4,
    pd.Timestamp('2025-09-01'): 2.3
}

rdi_adj = {
    pd.Timestamp('2025-04-01'): 2.0,
    pd.Timestamp('2025-05-01'): 2.0,
    pd.Timestamp('2025-06-01'): 2.0,
    pd.Timestamp('2025-07-01'): 2.0,
    pd.Timestamp('2025-08-01'): 2.0,
    pd.Timestamp('2025-09-01'): 2.0
}

home_price = {
    pd.Timestamp('2025-04-01'): 3.5,
    pd.Timestamp('2025-05-01'): 3.0,
    pd.Timestamp('2025-06-01'): 2.5,
    pd.Timestamp('2025-07-01'): 2.5,
    pd.Timestamp('2025-08-01'): 2.0,
    pd.Timestamp('2025-09-01'): 2.0
}

covid1 = {
    pd.Timestamp('2025-04-01'): 0.0,
    pd.Timestamp('2025-05-01'): 0.0,
    pd.Timestamp('2025-06-01'): 0.0,
    pd.Timestamp('2025-07-01'): 0.0,
    pd.Timestamp('2025-08-01'): 0.0,
    pd.Timestamp('2025-09-01'): 0.0
}

covid2 = {
    pd.Timestamp('2025-04-01'): 0.0,
    pd.Timestamp('2025-05-01'): 0.0,
    pd.Timestamp('2025-06-01'): 0.0,
    pd.Timestamp('2025-07-01'): 0.0,
    pd.Timestamp('2025-08-01'): 0.0,
    pd.Timestamp('2025-09-01'): 0.0
}
