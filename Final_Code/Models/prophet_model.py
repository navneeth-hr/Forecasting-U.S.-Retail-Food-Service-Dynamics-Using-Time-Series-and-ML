import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
from Utilities.utils import display_error_metrics

def run_prophet_model(data, target_col, regressors, periods, changepoint_prior_scale, seasonality_prior_scale,seasonality_mode,
                    yearly_seasonality, weekly_seasonality, daily_seasonality, freq, train_size,):

    df_prophet = data[[target_col] + regressors].reset_index()
    df_prophet.rename(columns={'Month': 'ds', target_col: 'y'}, inplace=True)  # Prophet requires 'ds' (date) and 'y' (target)

    split_idx = int(len(df_prophet) * train_size)
    train_df = df_prophet[:split_idx]
    test_df = df_prophet[split_idx:]

    model = Prophet(seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale)

    for regressor in regressors:
        model.add_regressor(regressor)

    model.fit(train_df)

    future = model.make_future_dataframe(periods=len(test_df), freq=freq)

    future = pd.merge(future, df_prophet[['ds'] + regressors], how='left', on='ds')
    forecast = model.predict(future)

    forecast_test = forecast[-len(test_df):]['yhat'].values
    y_true = test_df['y'].values
    y_pred = forecast_test

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) * 100
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    plt.figure(figsize=(10, 6))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual Sales', color='blue')
    plt.plot(test_df['ds'], forecast_test, label='Predicted Sales (Test)', color='orange')
    plt.axvline(df_prophet['ds'][split_idx], color='red', linestyle='--', label='Train/Test Split')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales (Millions)')
    plt.title(f'Prophet Predictions {target_col}')
    plt.legend()
    st.pyplot(plt)

    st.write(f"**RMSE:** {rmse}")
    st.write(f"**MAE:** {mae}")
    st.write(f"**MAPE:** {mape:.2f} %")
    st.write(f"**R-Squared:** {r2:.2f} %")
