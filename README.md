# Grocery Forecasting Engine

This repository contains a modular forecasting engine for U.S. grocery sales, using historical economic and consumer behavior indicators. The project is designed for flexibility, scalability, and transparency across different models and inputs.

---

## 📁 Project Structure

```
grocery_fcst/
├── forecast_engine/        # Reusable modeling, plotting, and utility functions
├── controls/               # Forecast-specific parameters (target, drivers, dates)
├── data/                   # Raw and processed data
│   └── inputs/             # Forward-looking input scenarios (ROC or absolute values)
├── output/                 # Forecast results, plots, and summary tables
├── notebooks/              # Jupyter notebooks to run forecasts
├── dashboard/              # Streamlit dashboard (optional)
├── README.md               # Project overview (this file)
└── requirements.txt        # (optional) Python dependencies
```

---

## 🚀 How to Run a Forecast

1. **Install dependencies (if not already installed)**

   ```bash
   pip install -r requirements.txt
   ```

2. **Load and run the notebook**

   ```bash
   cd notebooks/
   jupyter lab grocery_forecast_template.ipynb
   ```

3. **Customize your forecast**

   * Edit or create a new file in `controls/` to define:

     * Dependent variable
     * Independent variables
     * Date ranges
   * Swap forward-looking inputs in `data/inputs/`

---

## 🔧 Forecasting Methods Included

* OLS Regression (matrix, statsmodels, sklearn)
* Bayesian Linear Regression
* Residual Bootstrapping

---

## 📊 Visuals Included

* Actual vs Fitted vs Forecast (line chart)
* R² Fit (scatterplot)
* Input variable trends
* Distribution forecast with P5/P95 range

---

## 📦 Upcoming Enhancements

* YOY charting with confidence bands
* Automated Streamlit deployment
* CLI-based forecast runner

---

## 👤 Maintained by

**Plan-kton / Erick Karlson**

> Forecasting with purpose, structure, and clarity.
