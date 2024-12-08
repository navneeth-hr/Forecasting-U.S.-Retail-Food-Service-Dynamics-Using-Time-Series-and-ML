import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

def prepare_data(df, future_exog_df, target_column, exog_vars, test_ratio=0.2):
    """
    Prepare and split the data into train, test, and future datasets.
    """
    # Split historical data into training and test sets based on the specified ratio
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]

    # Extract target and exogenous variables for training and test sets
    y_train = train[target_column]
    y_test = test[target_column]
    X_train = train[exog_vars]
    X_test = test[exog_vars]

    # Set up future exogenous data, starting right after the last historical point
    future_start_date = test.index[-1] + pd.DateOffset(months=1)
    future_exog_df.index = pd.date_range(start=future_start_date, periods=len(future_exog_df), freq='M')
    X_future = future_exog_df[exog_vars]

    return y_train, y_test, X_train, X_test, X_future

def evaluate_forecast(actual, predicted):
    """
    Evaluate forecast accuracy using RMSE, MAE, MAPE, and R2 metrics.
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2
    }

def sarimax_forecast(y_train, X_train, X_test, X_future, order, seasonal_order, future_steps, selected_series, y_test=None):
    """
    Fit SARIMAX model, forecast test data, and forecast future data using provided exogenous variables.
    """
    # Fit the SARIMAX model on the training data
    model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    # Forecast on test set
    test_forecast = results.get_forecast(steps=len(X_test), exog=X_test)
    test_forecast_mean = test_forecast.predicted_mean

    # Plot training and test data along with forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Actual Train', color='blue')
    plt.plot(y_test.index, y_test, label='Actual Test', color='orange')
    plt.plot(test_forecast_mean.index, test_forecast_mean, label='Predicted Test', color='green')
    plt.xlabel('Month')
    plt.ylabel(selected_series)
    plt.title('Actual vs Test Predicted')
    plt.legend()
    plt.show()

    # Evaluate and display forecast accuracy if test data is provided
    metrics = evaluate_forecast(y_test, test_forecast_mean) if y_test is not None else {}

    # Forecast into the future using future exogenous variables
    future_forecast = results.get_forecast(steps=future_steps, exog=X_future)
    future_forecast_mean = future_forecast.predicted_mean
    future_forecast_ci = future_forecast.conf_int()

    # Plot future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(future_forecast_mean.index, future_forecast_mean, label='Future Forecast', linestyle='-', color='green')
    plt.fill_between(future_forecast_mean.index, future_forecast_ci.iloc[:, 0], future_forecast_ci.iloc[:, 1], color='pink', alpha=0.3, label='95% CI')
    plt.title(f'Future {selected_series} Forecast')
    plt.xlabel('Month')
    plt.ylabel(selected_series)
    plt.legend()
    plt.show()

    return future_forecast_mean, metrics

def run_sarima_model(df, selected_series, selected_regressors, future_exog_df, future_steps, order, seasonal_order):
    """
    Prepare data, run the SARIMAX forecast, and return the results.
    """
    y_train, y_test, X_train, X_test, X_future = prepare_data(df, future_exog_df, selected_series, selected_regressors)

    return sarimax_forecast(y_train, X_train, X_test, X_future, order, seasonal_order, future_steps, selected_series, y_test)

