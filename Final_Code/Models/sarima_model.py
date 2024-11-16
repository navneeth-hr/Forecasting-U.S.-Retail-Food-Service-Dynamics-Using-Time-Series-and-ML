import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def prepare_data(df, future_exog_df, target_column, exog_vars, test_ratio=0.2):
    """
    Combine historical data with future exogenous data and split into train/test sets.
    
    Parameters:
    df : pd.DataFrame
        Historical data including the target and exogenous variables.
    future_exog_df : pd.DataFrame
        Future exogenous variables for forecasting.
    target_column : str
        Name of the target column.
    exog_vars : list of str
        List of exogenous variable column names.
    test_ratio : float
        Ratio of data to use as the test set from the historical data.
    
    Returns:
    tuple
        y_train, y_test, X_train, X_test, X_future
    """
    # Split historical data into training and test sets based on test_ratio
    split_idx = int(len(df) * (1 - test_ratio))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    # Extract target and exogenous variables for training and test sets
    y_train = train[target_column]
    y_test = test[target_column]
    X_train = train[exog_vars]
    X_test = test[exog_vars]
    
    # Ensure `target_column` has NaNs in `future_exog_df` to avoid issues with the target in future data
    future_exog_df[target_column] = np.nan
    
    # Set the future index to start right after the last date in the test set
    future_start_date = test.index[-1] + pd.DateOffset(months=1)
    future_exog_df.index = pd.date_range(start=future_start_date, periods=len(future_exog_df), freq='M')
    
    # Extract future exogenous variables only (no target)
    X_future = future_exog_df[exog_vars]

    return y_train, y_test, X_train, X_test, X_future

def evaluate_forecast(actual, predicted):
    """
    Calculate RMSE, MAE, MAPE, and R2 metrics for evaluating the forecast.
    """
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    r2 = r2_score(actual, predicted)

    # rmse_val = round(rmse - (rmse * 0.95),2)
    # mae_val = round(mae - (mae * 0.95),2)

    rmse_val = rmse
    mae_val = mae

    return {
        'RMSE': rmse_val,
        'MAE': mae_val,
        'MAPE': mape,
        'R2': r2 * 100
    }

def sarimax_forecast(y_train, X_train, X_test, X_future, order, seasonal_order, future_steps, selected_series, y_test=None):
    """
    Fit SARIMAX model, forecast on test data, and then forecast future data using predicted exogenous variables.
    
    Parameters:
    y_train : pd.Series
        Training target data.
    X_train : pd.DataFrame
        Exogenous variables for training data.
    X_test : pd.DataFrame
        Exogenous variables for testing data.
    X_future : pd.DataFrame
        Exogenous variables for future forecasting.
    order : tuple
        SARIMA order parameters (p, d, q).
    seasonal_order : tuple
        Seasonal order parameters (P, D, Q, S).
    y_test : pd.Series, optional
        Actual test data for evaluation, if available.
    
    Returns:
    tuple
        Forecast mean, forecast confidence interval, and evaluation metrics.
    """
    # Fit SARIMAX model
    model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit(disp=False)

    test_steps = len(X_test)
    test_forecast = results.get_forecast(steps=test_steps, exog=X_test)
    test_forecast_mean = test_forecast.predicted_mean
    test_forecast_ci = test_forecast.conf_int()

    # Plot the historical data and test forecast
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Train')
    plt.plot(y_test.index, y_test, label='Test')
    plt.plot(test_forecast_mean.index, test_forecast_mean, linestyle='-', label='Test Forecast')
    plt.fill_between(test_forecast_ci.index, test_forecast_ci.iloc[:, 0], test_forecast_ci.iloc[:, 1], color='pink', alpha=0.25)
    plt.legend()
    plt.grid(True)
    plt.title('SARIMAX Forecast with Train and Test Data')
    st.pyplot(plt)

    # Evaluate test forecast if actual test values are provided
    metrics = {}
    if y_test is not None:
        metrics = evaluate_forecast(y_test, test_forecast_mean)
        st.write(f"RMSE: {metrics['RMSE']:.3f}")
        st.write(f"MAE: {metrics['MAE']:.3f}")
        st.write(f"MAPE: {metrics['MAPE']:.1f} %")
        st.write(f"R2: {metrics['R2']:.1f} %")
    
    rmse = metrics['RMSE']
    mae = metrics['MAE']
    r2 = metrics['R2']
    mape = metrics['MAPE']

    # Future forecasting using predicted exogenous variables
    future_steps = len(X_future) if future_steps in (None, '') else future_steps
    future_index = pd.date_range(start=X_test.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')
    future_forecast = results.get_forecast(steps=future_steps, exog=X_future)
    future_forecast_mean = future_forecast.predicted_mean
    future_forecast_ci = future_forecast.conf_int()

    future_forecast_df = pd.DataFrame({
        'Month': future_index,
        f'Predicted {selected_series}': future_forecast_mean.values
    })

    future_forecast_df['Month'] = future_forecast_df['Month'].dt.date
    future_forecast_df[f'Predicted {selected_series}'] = future_forecast_df[f'Predicted {selected_series}'].round(2)

    plt.figure(figsize=(12, 6))
    # plt.plot(y_train.index, y_train, label='Historical Data', color = 'blue')
    if y_test is not None:
        plt.plot(y_test.index, y_test, color = 'blue', label='Historical Data')
    # plt.plot(future_index, future_forecast_mean + rmse, linestyle='-', label='Future Forecast', color='green')
    plt.plot(future_index, future_forecast_mean, linestyle='-', label='Future Forecast', color='green')
    plt.fill_between(future_index, future_forecast_ci.iloc[:, 0], future_forecast_ci.iloc[:, 1], color='pink', alpha=0.3, label='95% Prediction Interval')
    plt.legend()
    plt.title('SARIMAX Forecast with Historical and Future Data')
    plt.xlabel('Time')
    plt.grid(True)
    plt.ylabel('Retail Sales')
    st.pyplot(plt)

    return future_forecast_df, rmse, mae, mape, r2

def run_sarima_model(df, selected_series, selected_regressors, future_exog_df, future_steps, order, seasonal_order):
    target_column = selected_series
    exog_vars = selected_regressors

    y_train, y_test, X_train, X_test, X_future = prepare_data(df, future_exog_df, target_column, exog_vars)
    
    st.subheader("SARIMAX Model Predictions vs Actual Data")

    # Run SARIMAX analysis
    return sarimax_forecast(y_train, X_train, X_test, X_future, order, seasonal_order, future_steps, target_column, y_test)