import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
import streamlit as st
from prophet.diagnostics import cross_validation, performance_metrics

def run_prophet_model(data, target_col, regressors, periods, changepoint_prior_scale, seasonality_prior_scale,
                      seasonality_mode, yearly_seasonality, weekly_seasonality, daily_seasonality, freq, train_size,
                      future_exog_df, future_steps):
    """
    Train a Prophet model, evaluate it, and make future predictions.
    """
    # Prepare data for Prophet
    df_prophet = data[[target_col] + regressors].reset_index()
    df_prophet.rename(columns={'index': 'ds', target_col: 'y'}, inplace=True)

    # Split data into training and testing sets
    split_idx = int(len(df_prophet) * train_size)
    train_df = df_prophet[:split_idx]
    test_df = df_prophet[split_idx:]

    # Initialize the Prophet model with specified settings
    model = Prophet(seasonality_mode=seasonality_mode,
                    yearly_seasonality=yearly_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    daily_seasonality=daily_seasonality,
                    changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale)

    # Add specified regressors to the model
    for regressor in regressors:
        model.add_regressor(regressor)

    # Fit the model on the training data
    model.fit(train_df)

    # Perform cross-validation and calculate performance metrics
    cv_results = cross_validation(model, initial=f'{540} days', period=f'{30} days', horizon=f'{future_steps * 30} days')
    cv_metrics = performance_metrics(cv_results)

    # Display cross-validation results
    st.subheader("Cross-Validation Metrics")
    st.write(f"**RMSE (Cross-Validation):** {cv_metrics['rmse'].mean():.2f}")
    st.write(f"**MAE (Cross-Validation):** {cv_metrics['mae'].mean():.2f}")
    st.write(f"**MAPE (Cross-Validation):** {cv_metrics['mape'].mean() * 100:.2f} %")

    # Prepare future dataframe for forecasting
    future = model.make_future_dataframe(periods=len(test_df) + future_steps, freq=freq)

    # Add future exogenous variables, ensuring they align with the forecast dates
    future = future.merge(future_exog_df.reset_index().rename(columns={'index': 'ds'}), on='ds', how='left')
    for regressor in regressors:
        future[regressor] = future[regressor].fillna(method='ffill').fillna(method='bfill')

    # Generate forecasts
    forecast = model.predict(future)

    # Extract forecasts for the test period and future period
    forecast_test = forecast.iloc[-(len(test_df) + future_steps):-future_steps]
    forecast_future = forecast.iloc[-future_steps:]
    y_true = test_df['y'].values
    y_pred = forecast_test['yhat'].values

    # Calculate and display evaluation metrics for the test set
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    r2 = r2_score(y_true, y_pred) * 100

    # Plot actual vs. predicted results for the test period
    plt.figure(figsize=(12, 6))
    plt.plot(df_prophet['ds'], df_prophet['y'], label='Actual Data', color='blue')
    plt.plot(test_df['ds'], forecast_test['yhat'], label='Predicted Test', color='green')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.show()

    # Display test evaluation metrics
    st.subheader("Test Evaluation Metrics")
    st.write(f"**RMSE:** {rmse:.2f}")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**MAPE:** {mape:.2f} %")
    st.write(f"**R2 Score:** {r2:.2f} %")

    # Plot future predictions with confidence intervals
    plt.figure(figsize=(12, 6))
    plt.plot(future['ds'], forecast_future['yhat'], label='Future Predictions', linestyle='--', color='green')
    plt.fill_between(future['ds'], forecast_future['yhat_lower'], forecast_future['yhat_upper'], color='pink', alpha=0.3, label='95% CI')
    plt.xlabel('Date')
    plt.ylabel(target_col)
    plt.title('Future Predictions')
    plt.legend()
    plt.show()

    # Prepare a DataFrame to return future predictions and confidence intervals
    future_predictions_df = pd.DataFrame({
        'Date': future['ds'][-future_steps:],
        'Predicted': forecast_future['yhat'].values,
        'Lower CI': forecast_future['yhat_lower'].values,
        'Upper CI': forecast_future['yhat_upper'].values
    })

    return future_predictions_df, rmse, mae, mape, r2

