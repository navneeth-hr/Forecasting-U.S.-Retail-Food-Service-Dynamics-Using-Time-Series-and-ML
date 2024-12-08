# U.S. Retail & Food Service Forecasting Dashboard

This repository contains a Streamlit-based interactive dashboard for forecasting U.S. Retail & Food Service data. The dashboard is designed to assist in trend analysis and predictive modeling using multiple advanced forecasting techniques. It is an end-to-end tool that supports data upload, hyperparameter tuning, and visualization of results.

---

## **Table of Contents**
1. [Overview](#overview)
2. [Features](#features)
3. [Models Implemented](#models-implemented)
4. [Installation](#installation)
5. [Usage](#usage)

---

## **Overview**

The **U.S. Retail & Food Service Forecasting Dashboard** provides users with the ability to:
- Analyze historical data.
- Generate forecasts using advanced machine learning and statistical models.
- Visualize predictions and compare model performance.

The dashboard integrates with external CSV files, such as future exogenous variable predictions, to enhance forecast accuracy.

---

## **Features**

- **File Uploader**: Upload datasets for custom analysis.
- **Forecast Configurations**: Set up short-term and long-term predictions.
- **Hyperparameter Tuning**: Adjust model-specific parameters for optimization.
- **Exogenous Variable Integration**: Use external predictors to improve forecasting accuracy.
- **Visualization**: Generate clear and interactive plots for historical data, predictions, and future trends.
- **Evaluation Metrics**: Assess model performance using metrics like RMSE, MAE, MAPE, and R².

*Figure 1 below demonstrates the dashboard's layout and modular design.*

---

## **Models Implemented**

The dashboard supports the following forecasting models:
1. **LSTM (Long Short-Term Memory)**:
   - Utilizes neural networks for sequential data forecasting.
   - Supports MinMax scaling and future prediction capabilities.
   - Implements `EarlyStopping` for optimized training.
2. **Prophet**: A Bayesian model designed for time series forecasting with seasonality and trend adjustments.
3. **Random Forest**: A tree-based ensemble machine learning algorithm.
4. **Holt's Winter**: A statistical method incorporating trend and seasonal components.
5. **SARIMAX (Seasonal AutoRegressive Integrated Moving Average with Exogenous variables)**: Combines ARIMA with seasonal and external predictors.

Each model script is modularized for easy updates and reuse:
- `lstm_model.py`
- `prophet_model.py`
- `random_forest_model.py`
- `holts_winter_model.py`
- `sarima_model.py`

---

## **Installation**

### Prerequisites
- Python 3.8 or higher
- Libraries: pandas, numpy, math, matplotlib, seaborn, sklearn, streamlit, keras, tensorflow, prophet, scipy, statsmodels

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Forecasting-U.S.-Retail-Food-Service-Dynamics-Using-Time-Series-and-ML.git

2. Navigate to the project directory:
   ```bash
   cd Final_code

3. Runn the Streamlit app
   ```bash
   streamlit run Dashboard.py

## **Usage**
- Upload Data: Navigate to the file uploader section to upload your dataset.
- Configure Forecast: Select a forecasting model, adjust hyperparameters, and define the forecast period.
- Visualize Results: View plots for actual vs. predicted data, future forecasts, and evaluation metrics.
- Compare Models: Analyze the metrics table to determine the best-performing model.
   
