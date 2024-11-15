import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import summary_table
import matplotlib.pyplot as plt

def prepare_data(df, target_column, exog_vars, split_ratio):
    """
    Prepare and split the data into train and test sets, handling exogenous variables.
    """
    # 1. Data Scaling
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

    # 2. Handle Outliers
    def remove_outliers(df, columns, threshold=3):
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df = df[z_scores < threshold]
        return df

    df_cleaned = remove_outliers(df_scaled, df_scaled.columns)

    # Calculate split index
    split_idx = int(len(df) * split_ratio)
    
    # Splitting the data
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    # Extracting target and exogenous variables
    y_train = train[target_column]
    y_test = test[target_column]
    X_train = train[exog_vars]
    X_test = test[exog_vars]
    
    return y_train, y_test, X_train, X_test

def plot_exogenous_time_series(df, target, exog_vars):
    """
    Generate time series plots for the target and exogenous variables.
    """

    fig, axes = plt.subplots(nrows=len(exog_vars)+1, ncols=1, figsize=(12, (len(exog_vars)+1) * 4))
    
    # Plot target variable
    axes[0].plot(df.index, df[target], label=target)
    axes[0].set_title(f'Time Series of {target}')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(target)
    axes[0].legend()

    # Plot exogenous variables
    for i, var in enumerate(exog_vars, start=1):
        axes[i].plot(df.index, df[var], label=var)
        axes[i].set_title(f'Time Series of {var}')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel(var)
        axes[i].legend()

        # Correlation
        corr = df[var].corr(df[target])
        axes[i].annotate(f'Corr with {target}: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', 
                         fontsize=12, horizontalalignment='left', verticalalignment='top', 
                         backgroundcolor='white')

    plt.tight_layout()
    plt.show()

def forecast_test_data(train, test, train_exog, test_exog, order, seasonal_order):
    """
    Test Data forecast and model evaluation metrics
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
    plt.xlabel('Date')
    plt.ylabel('Retail Sales (In Millions of Dollars)')
    plt.legend()
    plt.grid(True)  
    st.pyplot(plt)


    # Evaluation Metrics
    rmse = np.sqrt(mean_squared_error(test, forecast_mean))
    mae = mean_absolute_error(test, forecast_mean)
    r2 = r2_score(test, forecast_mean)
    mape = np.mean(np.abs((test - forecast_mean) / test)) * 100

    # rmse_val = round(rmse - (rmse * 0.95),2)
    # mae_val = round(mae - (mae * 0.95),2)

    rmse_val = rmse
    mae_val = mae

    return results, {
        'RMSE': rmse_val,
        'MAE': mae_val,
        'R2': r2 * 100,
        'MAPE': mape,
        'model_summary': results.summary().as_text()
    }

def forecast_future(train, test, train_exog, test_exog, order, seasonal_order, future_steps):
    """
    Forecast future values using a trained SARIMA model and future exogenous variables.

    Returns:
    pandas.Series: Forecasted values.
    """
    # Combine train and test data for full model fitting
    full_data = pd.concat([train, test])
    full_exog = pd.concat([train_exog, test_exog])

    # Fit SARIMAX model on full data
    model = SARIMAX(full_data, exog=full_exog, order=order, 
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    
    results = model.fit(disp=False) 

    # In-sample forecast (for the entire dataset)
    in_sample_forecast = results.get_prediction(start=full_data.index[0], end=full_data.index[-1], exog=full_exog)
    in_sample_forecast_mean = in_sample_forecast.predicted_mean

def sarimax_analysis(train, test, train_exog, test_exog, order, seasonal_order, future_steps):
    """
    Perform SARIMAX analysis, forecasting, and evaluation with exogenous variables.
    """
    # Sarima model on train and test data
    model_results, test_data_results = forecast_test_data(train, test, train_exog, test_exog, order, seasonal_order)
    return test_data_results

def future_forecast(model_results, steps, future_exog):
    """
    Generate future forecasts using the fitted SARIMAX model.
    """
    future_forecast = model_results.get_forecast(steps=steps, exog=future_exog)
    forecast_mean = future_forecast.predicted_mean
    forecast_ci = future_forecast.conf_int(alpha=0.05)
    
    return forecast_mean, forecast_ci

def run_sarima_model(df, selected_series, selected_regressors, future_steps):
    target_column = selected_series
    exog_vars = selected_regressors

    y_train, y_test, X_train, X_test = prepare_data(df, target_column, exog_vars, split_ratio = 0.8)
    
    # Define model non-seasonal order and seasonal order
    order = (0, 1, 0)
    seasonal_order = (0, 1, 0, 12)

    st.subheader("SARIMAX Model Predictions vs Actual Data")

    # Run SARIMAX analysis
    results = sarimax_analysis(y_train, y_test, X_train, X_test, order, seasonal_order, future_steps)
    
    st.write(f"RMSE: {results['RMSE']:.3f}")
    st.write(f"MAE: {results['MAE']:.3f}")
    st.write(f"R2: {results['R2']:.1f} %")
    st.write(f"MAPE: {results['MAPE']:.1f} %")