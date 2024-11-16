import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import warnings
import streamlit as st
warnings.filterwarnings("ignore")

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

def mle_hw_model(y_train, y_test, X_train, X_test, X_future, future_steps, selected_series):

    lr_model = LinearRegression().fit(X_train, y_train)
    train_pred = lr_model.predict(X_train)
    train_resid = y_train - train_pred

    # Apply Holt-Winters to the residuals
    model_hw_resid = ExponentialSmoothing(train_resid, seasonal_periods=12, trend='add', seasonal='add').fit()

    # For test data
    resid_forecast = model_hw_resid.forecast(len(y_test))
    final_forecast = lr_model.predict(X_test) + resid_forecast

    rmse = np.sqrt(mean_squared_error(y_test, final_forecast))
    mae = mean_absolute_error(y_test, final_forecast)
    r2 = r2_score(y_test, final_forecast)
    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    mape = np.mean(np.abs((y_test - final_forecast) / y_test)) * 100

    st.write(f'RMSE: {rmse:.3f}')
    st.write(f'MAE: {mae:.3f}')
    st.write(f'MAPE: {mape:.3f}%')
    st.write(f'R-Squared: {r2:.3%}')

    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Train Data')
    plt.plot(y_test.index, y_test, label='Test Data')
    plt.plot(y_test.index, final_forecast, label='MLE + HW Forecast', linestyle='-')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(selected_series)
    plt.grid(True)
    st.pyplot(plt)
    
    # For Future steps
    resid_forecast_future = model_hw_resid.forecast(future_steps)
    trend_forecast = lr_model.predict(X_future)
    future_forecast = trend_forecast + resid_forecast_future

    # Future forecasting using predicted exogenous variables
    future_steps = len(X_future) if future_steps in (None, '') else future_steps
    future_index = pd.date_range(start=X_test.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Train Data')
    plt.plot(y_test.index, y_test, label='Test Data')
    plt.plot(future_index, future_forecast + rmse, label='Future Forecast', linestyle='-', color = 'Green')
    plt.legend()
    plt.grid(True)
    plt.title('Future Retail Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel(selected_series)
    st.pyplot(plt)

    future_forecast_df = pd.DataFrame({
        'Month': future_index,
        f'Predicted {selected_series}': future_forecast.values
    })


    future_forecast_df['Month'] = future_forecast_df['Month'].dt.date
    future_forecast_df[f'Predicted {selected_series}'] = future_forecast_df[f'Predicted {selected_series}'].round(2)

    return future_forecast_df, rmse, mae, mape, r2


def run_hw_model(df, selected_series, selected_regressors, future_exog_df, future_steps):
    target_column = selected_series
    exog_vars = selected_regressors
    y_train, y_test, X_train, X_test, X_future = prepare_data(df, future_exog_df, target_column, exog_vars)

    st.subheader("Holt-Winters Model Predictions vs Actual Data")
    return mle_hw_model(y_train, y_test, X_train, X_test, X_future, future_steps, selected_series)
