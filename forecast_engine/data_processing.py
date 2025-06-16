import os
import numpy as np
import pandas as pd
from controls.grocery_control import *

# --- Load and prepare your data ---

def ingest_data(econ_data, rdi_data):

    # Import dataset and set index
    df = pd.read_csv(econ_data)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df = df.set_index('date')

    # bring in rdi adjusted
    rdi_adj = pd.read_excel(rdi_data)
    rdi_adj = rdi_adj[['observation_date', 'RDI_adj']]
    rdi_adj = rdi_adj.rename(columns={'observation_date':'date', 'RDI_adj':'rdi_adj'})
    rdi_adj = rdi_adj.set_index('date')

    # where rdi_adj is NaN, use Real Disposable Income data
    df = df.join(rdi_adj, how='left')
    df['rdi_adj'] = df['rdi_adj'].fillna(df['Real Disposable Income'])

    # rename columns
    df = df.rename(columns={
        'CPI (Food at Home)' : 'cpi_fah',
        'Avg Home Price' : 'home_price',
        'Grocery Sales' : 'grocery_sales',
        'Real Disposable Income' : 'rdi' 
    })

    # Create COVID indicator variables - this astype(int) converts True and False into 0 or 1
    df['covid1'] = (df.index == '2020-03-01').astype(int)
    df['covid2'] = ((df.index > '2020-03-01') & (df.index <= '2020-10-01')).astype(int)
    
    return df

# --- Build Forward File ----------------------------------------------------

def forward_values(csv_file, df_train):

    # Step 1: Create a date range (monthly)
    forward_values = pd.date_range(start_date_forecast, end_date_forecast, freq='MS')  # MS = Month Start

    # Step 2: Convert to DataFrame and calculate LY months
    forward_values_df = pd.DataFrame({'month': forward_values})
    forward_values_df['month_ly'] = forward_values_df['month'] - pd.DateOffset(years=1)

    # Step 3: Filter on LY X'set
    df_ly = df_train
    forward_values_df = pd.merge(
        forward_values_df, df_ly, left_on='month_ly', right_index=True, how='left').drop(columns=['y_fitted', 'residuals', 'grocery_sales'])

    # Step 4: rename ly columns
    forward_values_df = forward_values_df.rename(columns={'cpi_fah':'cpi_fah_ly', 'rdi_adj' : 'rdi_adj_ly', 'home_price': 'home_price_ly'})                                            

    # Step 4: add on YOY values and calculate forward values
    forward_values = pd.read_csv(csv_file)
    forward_values['date'] = pd.to_datetime(forward_values['date'])
    forward_values_df = pd.merge(forward_values_df, forward_values, left_on='month', right_on='date', how='inner').drop(columns='date')

    # Step 4: rename yoy columns
    forward_values_df = forward_values_df.rename(columns={'cpi_fah':'cpi_fah_yoy', 'rdi_adj' : 'rdi_adj_yoy', 'home_price': 'home_price_yoy'}) 

    # Step 5: calculate new forward values
    forward_values_df['cpi_fah'] = forward_values_df['cpi_fah_ly'] * (1 + forward_values_df['cpi_fah_yoy']/100)
    forward_values_df['home_price'] = forward_values_df['home_price_ly'] * (1 + forward_values_df['home_price_yoy']/100)
    forward_values_df['rdi_adj'] = forward_values_df['rdi_adj_ly'] * (1 + forward_values_df['rdi_adj_yoy']/100)

    forward_values_df.set_index('month', inplace=True)
    forward_values_df = forward_values_df[ind]
    
    return forward_values_df
