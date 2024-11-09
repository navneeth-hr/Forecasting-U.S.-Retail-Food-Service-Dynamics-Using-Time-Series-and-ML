import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import warnings
import streamlit as st
warnings.filterwarnings("ignore")



def prepare_data(df, target_column, exog_vars, split_ratio, time_column='Month'):
    """
    Prepare and split the data into train and test sets, handling exogenous variables.
    """
    # Calculate split index
    split_idx = int(len(df) * split_ratio)
    
    # Splitting the data
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    # Setting the index
    train.set_index(time_column, inplace=True)
    test.set_index(time_column, inplace=True)
    
    # Extracting target and exogenous variables
    y_train = train[target_column]
    y_test = test[target_column]
    X_train = train[exog_vars]
    X_test = test[exog_vars]
    
    return y_train, y_test, X_train, X_test


def hw_model(y_train, y_test, X_train, X_test, future_steps):
    lr_model = LinearRegression().fit(X_train, y_train)

    train_pred = lr_model.predict(X_train)
    train_resid = y_train - train_pred

    # Apply Holt-Winters to the residuals
    model_hw_resid = ExponentialSmoothing(
        train_resid,
        seasonal_periods=12,
        trend='add',
        seasonal='add'
    ).fit()

    # Forecast the residuals
    resid_forecast = model_hw_resid.forecast(len(y_test))

    # Final forecast is the sum of long-term forecast and residual forecast
    final_forecast = lr_model.predict(X_test) + resid_forecast

    # Plot and evaluate
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, final_forecast, label='Hybrid ECM Forecast', linestyle='--')
    plt.legend()
    # plt.show()
    st.pyplot(plt)

    mae = mean_absolute_error(y_test, final_forecast)
    rmse = np.sqrt(mean_squared_error(y_test, final_forecast))
    r2 = r2_score(y_test, final_forecast)

    st.write(f'MAE: {mae:.3f}')
    st.write(f'RMSE: {rmse:.3f}')
    st.write(f'R2: {r2:.3%}')

    mape = np.mean(np.abs((y_test - final_forecast) / y_test)) * 100
    # print(f'MAPE: {mape:.3f}%')

    n = len(y_test)
    p = X_test.shape[1]
    adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    st.write(f'Adjusted R2: {adjusted_r2:.3%}')


def run_hw_model(df, selected_series, future_steps):
    target_column = selected_series
    exog_vars = ['Monthly Real GDP Index', 'UNRATE', 'CPI Value']

    y_train, y_test, X_train, X_test = prepare_data(df, target_column, exog_vars, split_ratio = 0.8)
    st.subheader("Holt-Winters Model Predictions vs Actual Data")

    hw_model(y_train, y_test, X_train, X_test, future_steps)