import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
from Utilities.utils import display_error_metrics

def run_prophet_model(data, target_col, regressors, periods, changepoint_prior_scale, seasonality_prior_scale,
                      seasonality_mode, yearly_seasonality, weekly_seasonality, daily_seasonality, freq, train_size,
                      future_exog_df, future_steps):

    # Prepare data for Prophet
    df_prophet = data[[target_col] + regressors].reset_index()
    df_prophet.rename(columns={'Month': 'ds', target_col: 'y'}, inplace=True)  # Prophet requires 'ds' (date) and 'y' (target)

    # Split data into training and testing sets
    split_idx = int(len(df_prophet) * train_size)
    train_df = df_prophet[:split_idx]
    test_df = df_prophet[split_idx:]

    # Initialize the Prophet model
    model = Prophet(seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale)

    # Add regressors to the model
    for regressor in regressors:
        model.add_regressor(regressor)

    # Fit the model on training data
    model.fit(train_df)

    # Prepare future dataframe for predictions (test period + future steps)
    future = model.make_future_dataframe(periods=len(test_df) + future_steps, freq=freq)

    # Merge future exogenous variables
    future = pd.merge(future, df_prophet[['ds'] + regressors], how='left', on='ds')

    # Add future exogenous variables from future_exog_df
    future_exog_df = future_exog_df.reset_index().rename(columns={'Month': 'ds'})
    future = pd.concat([future, future_exog_df], axis=0).drop_duplicates(subset='ds').sort_values(by='ds').reset_index(drop=True)

    # Fill missing values for regressors (if any)
    for regressor in regressors:
        future[regressor] = future[regressor].fillna(method='ffill').fillna(method='bfill')

    # Generate predictions
    forecast = model.predict(future)

    # Extract predictions for test and future periods
    forecast_test = forecast[-(len(test_df) + future_steps):-future_steps]['yhat'].values
    forecast_future = forecast[-future_steps:]['yhat'].values
    y_true = test_df['y'].values
    y_pred = forecast_test

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) * 100
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    # Plot actual vs. predicted results for the test period
    plt.figure(figsize=(12, 6))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual Sales', color='blue')
    plt.plot(test_df['ds'], forecast_test, label='Predicted Sales (Test)', color='orange')
    plt.axvline(df_prophet['ds'][split_idx], color='red', linestyle='--', label='Train/Test Split')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales (Millions)')
    plt.title(f'Prophet Predictions {target_col} (Test Period)')
    plt.legend()
    st.pyplot(plt)

    # Display evaluation metrics
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {mape:.2f} %")
    st.write(f"**R-Squared:** {r2:.2f} %")

    # Future Predictions Plot
    future_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq=freq)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[target_col], label='Historical Sales', color='blue')
    plt.plot(future_index, forecast_future, label='Future Predictions', linestyle='--', color='orange')
    plt.xlabel('Date')
    plt.ylabel('Retail Sales (Millions)')
    plt.title(f'Future {target_col} Sales Predictions')
    plt.legend()
    st.pyplot(plt)

    # Future predictions DataFrame
    future_predictions_df = pd.DataFrame({
        'Month': future_index,
        f'Predicted {target_col}': forecast_future
    })

    # Show the future predictions table
    st.write("### Future Predictions Table")
    st.dataframe(future_predictions_df)

    return future_predictions_df
