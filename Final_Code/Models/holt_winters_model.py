import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import streamlit as st
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

def prepare_data(df, future_exog_df, target_column, exog_vars, lags, test_ratio=0.2):
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
    # Create lagged features for exogenous variables
    for lag in [lags]:
        for var in exog_vars:
            df[f'{var}_lag_{lag}'] = df[var].shift(lag)

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
    model_hw_resid = ExponentialSmoothing(
                        train_resid, 
                        seasonal_periods=12, 
                        trend='add', 
                        seasonal='add',
                        initialization_method="estimated",
                        damped_trend = True
                        ).fit(smoothing_level=0.23, smoothing_slope=0.78, smoothing_seasonal=0.01)

    # For test data
    resid_forecast = model_hw_resid.forecast(len(y_test))
    final_forecast = lr_model.predict(X_test) + resid_forecast

    rmse = np.sqrt(mean_squared_error(y_test, final_forecast))
    mae = mean_absolute_error(y_test, final_forecast)
    r2 = r2_score(y_test, final_forecast)
    mape = np.mean(np.abs((y_test - final_forecast) / y_test)) * 100

    print(f'RMSE: {rmse:.2f}')
    print(f'MAE: {mae:.2f}')
    print(f'MAPE: {mape:.2f}%')
    print(f'R-Squared: {r2:.2%}')

    confidence_level = 0.95
    z_score = stats.norm.ppf((1 + confidence_level) / 2)

    std_error = rmse /2

    ci_range_test = z_score * std_error

    lower_ci_test = final_forecast - ci_range_test
    upper_ci_test = final_forecast + ci_range_test

    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, label='Historical Data', color='blue')
    plt.plot(y_test.index, y_test, color='blue')
    plt.plot(y_test.index, final_forecast, label='MLE + HW Forecast', linestyle='-', color='orange')
    plt.fill_between(y_test.index, lower_ci_test, upper_ci_test, color='orange', alpha=0.2, label=f'{confidence_level*100}% CI')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel(selected_series)
    plt.grid(True)
    plt.title('Test Period Forecast with Confidence Interval')
    st.pyplot(plt)

    # For Future steps
    resid_forecast_future = model_hw_resid.forecast(future_steps)
    trend_forecast = lr_model.predict(X_future)
    future_forecast = trend_forecast + resid_forecast_future

    # Future forecasting using predicted exogenous variables
    future_steps = len(X_future) if future_steps in (None, '') else future_steps
    future_index = pd.date_range(start=X_test.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')

    ci_range_future = z_score * std_error

    lower_ci_future = future_forecast + (ci_range_future*1.25)
    upper_ci_future = future_forecast + (ci_range_future*0.75)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(y_train.index, y_train, color = "blue")
    plt.plot(y_test.index, y_test, label='Historical Data', color = "blue")
    plt.plot(future_index, future_forecast + rmse, label='Future Forecast', linestyle='-', color='green')
    plt.fill_between(future_index, lower_ci_future, upper_ci_future, color='green', alpha=0.2, label=f'{confidence_level*100}% CI')
    plt.legend()
    plt.grid(True)
    plt.title(f'Future {selected_series.split("(")[0]} Forecast with Confidence Interval')
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

def run_hw_model(df, selected_series, selected_regressors, future_exog_df, future_steps, lag_feature):
    target_column = selected_series
    exog_vars = selected_regressors
    y_train, y_test, X_train, X_test, X_future = prepare_data(df, future_exog_df, target_column, exog_vars, lag_feature)

    st.subheader("Holt-Winters Model Predictions vs Actual Data")
    return mle_hw_model(y_train, y_test, X_train, X_test, X_future, future_steps, selected_series)