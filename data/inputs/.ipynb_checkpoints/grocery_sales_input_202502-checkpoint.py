# forecast_inputs_apr_to_sep.py
#ind_0 = 'grocery_sales_lag1' 
#ind_1 = 'cpi_fah'
#ind_2 = 'rdi_adj'
#ind_3 = 'home_price'
#ind_4 = 'covid1'
#ind_5 = 'covid2'

import pandas as pd

# Forward month strings
fwrd_1 = '2025-02-01'
fwrd_2 = '2025-03-01'
fwrd_3 = '2025-04-01'
fwrd_4 = '2025-05-01'
fwrd_5 = '2025-06-01'
fwrd_6 = '2025-07-01'

ind_label_1_dic = {
    pd.Timestamp(fwrd_1): 1.76,
    pd.Timestamp(fwrd_2): 1.8,
    pd.Timestamp(fwrd_3): 2.5,
    pd.Timestamp(fwrd_4): 2.85,
    pd.Timestamp(fwrd_5): 3.2,
    pd.Timestamp(fwrd_6): 2.4
}

ind_label_2_dic = {
    pd.Timestamp(fwrd_1): 2.0,
    pd.Timestamp(fwrd_2): 2.0,
    pd.Timestamp(fwrd_3): 2.0,
    pd.Timestamp(fwrd_4): 2.0,
    pd.Timestamp(fwrd_5): 2.0,
    pd.Timestamp(fwrd_6): 2.0
}

ind_label_3_dic = {
    pd.Timestamp(fwrd_1): 3.0,
    pd.Timestamp(fwrd_2): 2.5,
    pd.Timestamp(fwrd_3): 2.5,
    pd.Timestamp(fwrd_4): 2.0,
    pd.Timestamp(fwrd_5): 2.0,
    pd.Timestamp(fwrd_6): 2.0
}

ind_label_4_dic = {
    pd.Timestamp(fwrd_1): 0,
    pd.Timestamp(fwrd_2): 0,
    pd.Timestamp(fwrd_3): 0,
    pd.Timestamp(fwrd_4): 0,
    pd.Timestamp(fwrd_5): 0,
    pd.Timestamp(fwrd_6): 0
}

ind_label_5_dic = {
    pd.Timestamp(fwrd_1): 0,
    pd.Timestamp(fwrd_2): 0,
    pd.Timestamp(fwrd_3): 0,
    pd.Timestamp(fwrd_4): 0,
    pd.Timestamp(fwrd_5): 0,
    pd.Timestamp(fwrd_6): 0
}


