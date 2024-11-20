import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import streamlit as st
import matplotlib.pyplot as plt
from Utilities.utils import display_error_metrics
from prophet.diagnostics import cross_validation, performance_metrics

def run_prophet_model(data, target_col, regressors, periods, changepoint_prior_scale, seasonality_prior_scale,
                      seasonality_mode, yearly_seasonality, weekly_seasonality, daily_seasonality, freq, train_size,
                      future_exog_df, future_steps):

    # Prepare data for Prophet
    df_prophet = data[[target_col] + regressors].reset_index()
    df_prophet.rename(columns={'Month': 'ds', target_col: 'y'}, inplace=True)

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

    # Cross-validation using Prophet's built-in function
    cv_results = cross_validation(model, initial=f'{540} days', 
                                  period=f'{30} days', horizon=f'{future_steps * 30} days')

    # Performance metrics from cross-validation
    cv_metrics = performance_metrics(cv_results)

    st.subheader("Cross-Validation Metrics")
    st.write(f"**RMSE (Cross-Validation):** {cv_metrics['rmse'].mean():.2f}")
    st.write(f"**MAE (Cross-Validation):** {cv_metrics['mae'].mean():.2f}")
    st.write(f"**MAPE (Cross-Validation):** {(cv_metrics['mape'].mean()) * 100:.2f} %")

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

    # Extract predictions and 95% CI for test and future periods
    forecast_test = forecast[-(len(test_df) + future_steps):-future_steps][['yhat', 'yhat_lower', 'yhat_upper']]
    forecast_future = forecast[-future_steps:][['yhat', 'yhat_lower', 'yhat_upper']]
    y_true = test_df['y'].values
    y_pred = forecast_test['yhat'].values

    # Calculate evaluation metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) * 100
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    st.subheader("Model Training & Test Prediction")

    # Plot actual vs. predicted results for the test period
    plt.figure(figsize=(12, 6))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual Train', color='blue')
    plt.plot(test_df['ds'], forecast_test['yhat'], label='Predicted Test', color='green')
    plt.axvline(df_prophet['ds'][split_idx], color='red', linestyle='--', label='Train/Test Split')
    plt.xlabel('Month')
    plt.ylabel(f'{target_col}')
    plt.title(f'Actual vs Test Predicted')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Test Evaluation Metrics")

    # Display evaluation metrics
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {mape:.2f} %")
    st.write(f"**R-Squared:** {r2:.2f} %")

    st.subheader("Future Predictions")

    # Future Predictions Plot with 95% CI
    future_index = pd.date_range(start=data.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq=freq)
    plt.figure(figsize=(12, 6))
    plt.plot(data.index, data[target_col], label='Historical Sales', color='blue')
    plt.plot(future_index, forecast_future['yhat'], label='Future Predictions', linestyle='--', color='green')
    plt.fill_between(future_index, forecast_future['yhat_lower'], forecast_future['yhat_upper'], color='pink', alpha=0.25, label='95% CI')
    plt.xlabel('Month')
    plt.ylabel(f'{target_col}')
    plt.title(f'Future {target_col}')
    plt.legend()
    st.pyplot(plt)

    # Future predictions DataFrame
    future_predictions_df = pd.DataFrame({
        'Month': future_index,
        f'Predicted {target_col}': forecast_future['yhat'].values,
        'Lower Bound': forecast_future['yhat_lower'].values,
        'Upper Bound': forecast_future['yhat_upper'].values
    })
    future_predictions_df['Month'] = pd.to_datetime(future_predictions_df['Month']).dt.date

    return future_predictions_df, rmse, mae, mape, r2
