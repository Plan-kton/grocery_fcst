# grocery_fcst
This file forecasts US Grocery Sales
The dependent variable is US Grocery Sales
The indepdent variables are CPI Food-at-home, Real Disposable Income, Average Home Prices, and Covid dummy vars

The data is sources from the Federal Reserve Banks FRED database
* US Grocery Sales: US Census Bureau, Advanced Retail Sales, Grocery
* US Grocery Sales Lag 1: This reduces autocorrelation and is just the US Grocery Sales lagged one period
* CPI Food-at-Home: U.S. Bureau of Labor Statistics
* Real Disposable Income: US Bureau of Economic Analysis
* Average Home Prices: S&P CoreLogic Case-Shiller U.S. National Home Price Index

* CPI Food-at-Home, which is the average price index for grocery sales, is the biggest driver of grocery sales. When it grows or declines, so does grocery sales.  
* Real Disposable Income is the second strongest driver.  When incomes are growing faster than inflation that drives grocery sales higher
* Average home prices represents the wealth effect.  When home prices are increasing, people are spend more on groceries

Validation is next.  Last year the model predicted 2% YoY sales increase for CY 2024 and the year ended at 1.8%.  Could have been luck and we will find out as I 
introduce holdout samples.
