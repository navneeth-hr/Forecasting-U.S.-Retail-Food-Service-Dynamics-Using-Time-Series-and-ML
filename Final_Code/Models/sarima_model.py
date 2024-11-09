import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st

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

def plot_exogenous_relationships(df, target, exog_vars):
    """
    Generate scatter plots to visualize the relationship between the target and exogenous variables.
    """
    fig, axes = plt.subplots(nrows=len(exog_vars), ncols=1, figsize=(12, len(exog_vars) * 4))
    for i, var in enumerate(exog_vars):
        axes[i].scatter(df[var], df[target], alpha=0.5)
        axes[i].set_title(f'Relationship between {target} and {var}')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel(target)
        # Correlation
        corr = df[var].corr(df[target])
        axes[i].annotate(f'Corr: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12,
                         horizontalalignment='left', verticalalignment='top', backgroundcolor='white')
    plt.tight_layout()
    plt.show()

def sarimax_analysis(train, test, train_exog, test_exog, order, seasonal_order):
    """
    Perform SARIMAX analysis, forecasting, and evaluation with exogenous variables.
    """
    # Fit SARIMAX model
    model = SARIMAX(train, exog = train_exog, order = order, 
                    seasonal_order = seasonal_order,
                    enforce_stationarity = False, 
                    enforce_invertibility = False)
    
    results = model.fit(disp=False)

    # Forecast
    forecast_steps = len(test)
    forecast = results.get_forecast(steps=forecast_steps, exog=test_exog)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int(alpha=0.01)
    
    # Ensure forecast index aligns with test data index
    forecast_mean.index = test.index
    forecast_ci.index = test.index

    plt.figure(figsize=(12, 6))
    plt.plot(train.index, train, label='Train')
    plt.plot(test.index, test, label='Test')
    plt.plot(forecast_mean.index, forecast_mean, label='Forecast', linestyle='--')
    plt.fill_between(forecast_ci.index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.25, label="C.I")
    plt.title('SARIMAX Forecast vs Actuals')
    plt.xticks(rotation=45)
    plt.xlabel('Date')  # Label for the x-axis
    plt.ylabel('Retail Sales (In Millions of Dollars)')  # Label for the y-axis
    plt.legend()
    plt.grid(True)  # Optionally add a grid for better readability
    # plt.show()  # Uncomment this line if you're testing this code outside Streamlit
    st.pyplot(plt)


    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(test, forecast_mean))
    mae = mean_absolute_error(test, forecast_mean)
    r2 = r2_score(test, forecast_mean)
    mape = np.mean(np.abs((test - forecast_mean) / test)) * 100

    return {
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2 * 100,
        'MAPE': mape,
        'model_summary': results.summary().as_text()
    }

def run_sarima_model(df, selected_series, future_steps):
    target_column = selected_series
    exog_vars = ['Monthly Real GDP Index', 'UNRATE', 'CPI Value']

    y_train, y_test, X_train, X_test = prepare_data(df, target_column, exog_vars, split_ratio = 0.8)
    
    order = (0, 1, 0)
    seasonal_order = (0, 1, 0, 12)

    st.subheader("SARIMAX Model Predictions vs Actual Data")
    results = sarimax_analysis(y_train, y_test, X_train, X_test, order, seasonal_order)
    
    st.write(f"RMSE: {results['RMSE']:.3f}")
    st.write(f"MAE: {results['MAE']:.3f}")
    st.write(f"R2: {results['R2']:.1f} %")
    st.write(f"MAPE: {results['MAPE']:.1f} %")
    