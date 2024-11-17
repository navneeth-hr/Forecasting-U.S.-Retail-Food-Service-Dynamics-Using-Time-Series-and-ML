import streamlit as st
import pandas as pd
from Utilities.utils import plot_data, display_error_metrics, generate_insights
from Models.lstm_model import run_lstm_model
from Models.prophet_model import run_prophet_model
from Models.random_forest_model import run_rf_model
from Models.xgboost_model import run_xgboost_model
from Models.holt_winters_model import run_hw_model
from Models.sarima_model import run_sarima_model

# Title and Sidebar
st.title("US Retail & Food Service Forecasting Dashboard")
st.sidebar.title("Model Configuration")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df['Month'] = pd.to_datetime(df['Month'], format='mixed')
    df.set_index('Month', inplace=True)
    regressors = ['Monthly Real GDP Index', 'UNRATE(%)', 'CPI Value']

    st.write("### Dataset Shape")
    st.write(df.shape)
    st.write("### Dataset Statistics")
    st.write(df.describe())

    url = "https://raw.githubusercontent.com/navneeth-hr/Forecasting-U.S.-Retail-Food-Service-Dynamics-Using-Time-Series-and-ML/refs/heads/main/Final_Code/Datasets/future_exo_vars_predictions_using_LSTM.csv"
    future_exog_df = pd.read_csv(url)
    future_exog_df['Month'] = pd.to_datetime(future_exog_df['Month'], format='mixed')
    future_exog_df.set_index('Month', inplace=True)

    category_type = st.sidebar.selectbox(
        "Select Category Type", 
        ["Retail sales($MM)", "Food services and drinking places($MM)"]
    )

    if category_type == "Retail sales($MM)":
        selected_series = st.sidebar.selectbox(
            "Select Retail Sub-Category", 
            [
                "Retail sales($MM)", 
                "Motor vehicle and parts dealers($MM)",
                "Furniture and home furnishings stores($MM)",
                "Electronics and appliance stores($MM)",
                "Building mat. and garden equip. and supplies dealers($MM)",
                "Food and beverage stores($MM)",
                "Health and personal care stores($MM)",
                "Gasoline stations($MM)",
                "Clothing and clothing access. Stores($MM)",
                "Sporting goods, hobby, musical instrument, and book stores($MM)",
                "General merchandise stores($MM)",
                "Miscellaneous store retailers($MM)",
                "Nonstore retailers($MM)"
            ]
        )
    else:
        selected_series = "Food services and drinking places($MM)"
    

    # Forecast Section
    st.sidebar.header("Forecast Configuration")
    
    # Use key parameter to update session state
    forecast_type = st.sidebar.radio("Select Forecast Type", ["Short-term", "Long-term"], key="forecast_type")


    if forecast_type == "Short-term":
        future_steps = st.sidebar.slider("Future Steps (months)", min_value=1, max_value=8, value=3, key="future_steps")
        model_type = st.sidebar.selectbox("Select Model", ["Holt Winters", "SARIMAX"], key="model_type")
        st.sidebar.info("Holts Winter and SARIMAX are recommended for short-term forecasting up to 8 months.")
    else:
        future_steps = st.sidebar.slider("Future Steps (months)", min_value=8, max_value=24, value=12, key="future_steps")
        model_type = st.sidebar.selectbox("Select Model", ["Prophet", "LSTM", "Random Forest", "XGBoost"], key="model_type")
        st.sidebar.info("Prophet and LSTM are recommended for complex long-term forecasting up to 2 years and Random forest to determine the non-linear relationship")
    
    future_exog_df = future_exog_df.iloc[: future_steps]

    # Hyperparameter Tuning
    if model_type == "LSTM":
        if selected_series == "Retail sales($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.01, max_value=1.0, value=0.01)
        elif selected_series == "Food services and drinking places($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=95)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=24)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.01, max_value=1.0, value=0.05)
        elif selected_series == "Motor vehicle and parts dealers($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Furniture and home furnishings stores($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=48)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=85)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.01, max_value=1.0, value=0.01)
        elif selected_series == "Electronics and appliance stores($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Building mat. and garden equip. and supplies dealers($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Food and beverage stores($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Health and personal care stores($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Gasoline stations($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Clothing and clothing access. Stores($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Sporting goods, hobby, musical instrument, and book stores($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "General merchandise stores($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Miscellaneous store retailers($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Nonstore retailers($MM)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        
    elif model_type == "Prophet":
        if selected_series == "Retail sales($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=10.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=24)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"], index=0)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Food services and drinking places($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.000, max_value=1.00, value=0.001)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=10.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"], index=1)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Motor vehicle and parts dealers($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.9)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=6.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"],index=0)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Furniture and home furnishings stores($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.07)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=10.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=24)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"],index=1)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Electronics and appliance stores($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.01)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=10.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=24)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"],index=1)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Building mat. and garden equip. and supplies dealers($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Food and beverage stores($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Health and personal care stores($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Gasoline stations($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Clothing and clothing access. Stores($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Sporting goods, hobby, musical instrument, and book stores($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "General merchandise stores($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Miscellaneous store retailers($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Nonstore retailers($MM)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)

    elif model_type == "Random Forest":
        if selected_series == "Retail sales($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Food services and drinking places($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Motor vehicle and parts dealers($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Furniture and home furnishings stores($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Electronics and appliance stores($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Building mat. and garden equip. and supplies dealers($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Food and beverage stores($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Health and personal care stores($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Gasoline stations($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Clothing and clothing access. Stores($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Sporting goods, hobby, musical instrument, and book stores($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "General merchandise stores($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Miscellaneous store retailers($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Nonstore retailers($MM)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)

    elif model_type == "XGBoost":
        learning_rate = st.sidebar.slider("Learning Rate (XGBoost)", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
        n_estimators = st.sidebar.slider("Number of Estimators (XGBoost)", min_value=50, max_value=500, value=100, step=10)
        max_depth = st.sidebar.slider("Max Depth (XGBoost)", min_value=3, max_value=10, value=6)
        min_child_weight = st.sidebar.slider("Min Child Weight (XGBoost)", min_value=1, max_value=10, value=1)

    elif model_type == "XGBoost":
        if selected_series == "Retail sales($MM)":
            learning_rate = st.sidebar.slider("Learning Rate (XGBoost)", min_value=0.01, max_value=0.3, value=0.01)
            n_estimators = st.sidebar.slider("Number of Estimators (XGBoost)", min_value=50, max_value=500, value=100)
            max_depth = st.sidebar.slider("Max Depth (XGBoost)", min_value=3, max_value=10, value=6)
            min_child_weight = st.sidebar.slider("Min Child Weight (XGBoost)", min_value=1, max_value=10, value=1)
        elif selected_series == "Food services and drinking places($MM)":
            learning_rate = st.sidebar.slider("Learning Rate (XGBoost)", min_value=0.01, max_value=0.3, value=0.03)
            n_estimators = st.sidebar.slider("Number of Estimators (XGBoost)", min_value=50, max_value=500, value=100)
            max_depth = st.sidebar.slider("Max Depth (XGBoost)", min_value=3, max_value=10, value=6)
            min_child_weight = st.sidebar.slider("Min Child Weight (XGBoost)", min_value=1, max_value=10, value=1)
        elif selected_series == "Furniture and home furnishings stores($MM)":
            learning_rate = st.sidebar.slider("Learning Rate (XGBoost)", min_value=0.01, max_value=0.3, value=0.05)
            n_estimators = st.sidebar.slider("Number of Estimators (XGBoost)", min_value=50, max_value=500, value=100)
            max_depth = st.sidebar.slider("Max Depth (XGBoost)", min_value=3, max_value=10, value=6)
            min_child_weight = st.sidebar.slider("Min Child Weight (XGBoost)", min_value=1, max_value=10, value=1)

    elif model_type == "SARIMAX":
        if selected_series == "Retail sales($MM)":

            st.sidebar.header("Order Configurations for SARIMAX")
            p = st.sidebar.slider("p (AR)", min_value=0, max_value=7, value=1, step=1)
            q = st.sidebar.slider("Q (MA)", min_value=0, max_value=7, value=1, step=1)
            d = st.sidebar.slider("d (differencing)", min_value=0, max_value=3, value=1, step=1)
            order = (p, d, q)

            # seasonal_order = (1, 1, 1, 12)
            st.sidebar.subheader("Seasonal Orders")
            P = st.sidebar.slider("P (S-AR)", min_value=1, max_value=7, value=1, step=1)
            Q = st.sidebar.slider("Q (S-MA)", min_value=0, max_value=7, value=1, step=1)
            D = st.sidebar.slider("D (Seasonal differencing)", min_value=0, max_value=3, value=1, step=1)
            m = st.sidebar.slider("m (Seasonal Period)", min_value=2, max_value=24, value=12, step=1)
            seasonal_order = (P, D, Q, m)

        elif selected_series == "Furniture and home furnishings stores($MM)":
            # order = (0, 0, 1)
            # seasonal_order = (0, 1, 1, 12)

            st.sidebar.header("Order Configurations for SARIMAX")
            p = st.sidebar.slider("p (AR)", min_value=0, max_value=7, value=0, step=1)
            q = st.sidebar.slider("Q (MA)", min_value=0, max_value=7, value=2, step=1)
            d = st.sidebar.slider("d (differencing)", min_value=0, max_value=2, value=1, step=1)
            order = (p, d, q)

            # seasonal_order = (0, 0, 1, 12)
            st.sidebar.subheader("Seasonal Orders")
            P = st.sidebar.slider("P (S-AR)", min_value=0, max_value=7, value=0, step=1)
            Q = st.sidebar.slider("Q (S-MA)", min_value=0, max_value=7, value=1, step=1)
            D = st.sidebar.slider("D (Seasonal differencing)", min_value=0, max_value=2, value=0, step=1)
            m = st.sidebar.slider("m (Seasonal Period)", min_value=2, max_value=24, value=12, step=1)
            seasonal_order = (P, D, Q, m)

        elif selected_series == "Food services and drinking places($MM)":
            # order = (1, 1, 1)
            # seasonal_order = (1, 1, 1, 12)

            st.sidebar.header("Order Configurations for SARIMAX")
            p = st.sidebar.slider("p (AR)", min_value=0, max_value=7, value=1, step=1)
            q = st.sidebar.slider("Q (MA)", min_value=0, max_value=7, value=1, step=1)
            d = st.sidebar.slider("d (differencing)", min_value=0, max_value=3, value=1, step=1)
            order = (p, d, q)

            # seasonal_order = (1, 1, 1, 12)
            st.sidebar.subheader("Seasonal Orders")
            P = st.sidebar.slider("P (S-AR)", min_value=1, max_value=7, value=1, step=1)
            Q = st.sidebar.slider("Q (S-MA)", min_value=0, max_value=7, value=1, step=1)
            D = st.sidebar.slider("D (Seasonal differencing)", min_value=0, max_value=3, value=1, step=1)
            m = st.sidebar.slider("m (Seasonal Period)", min_value=2, max_value=24, value=12, step=1)
            seasonal_order = (P, D, Q, m)

    elif model_type == "Holt Winters":
        if selected_series == "Retail sales($MM)":
            st.sidebar.header("Hyper-Parameter Tuning for MLE + HW")
            lag_feature = st.sidebar.slider("Number Of Lagged Features", min_value = 0, max_value = 5, value = 1, step = 1)

        elif selected_series == "Furniture and home furnishings stores($MM)":
            st.sidebar.header("Hyper-Parameter Tuning for MLE + HW")
            lag_feature = st.sidebar.slider("Number Of Lagged Features", min_value = 0, max_value = 5, value = 1, step = 1)

        elif selected_series == "Food services and drinking places($MM)":
            st.sidebar.header("Hyper-Parameter Tuning for MLE + HW")
            lag_feature = st.sidebar.slider("Number Of Lagged Features", min_value = 0, max_value = 5, value = 1, step = 1)

    # Display selected data series
    if selected_series in df.columns:
        data = df[[selected_series]]
        st.subheader(f"Displaying {selected_series} Data")
        plot_data(data, selected_series)
    else:
        st.error(f"Selected series '{selected_series}' not found in the dataset.")

    # Exogenous Variables selection
    st.sidebar.header("External Factors")
    include_gdp = st.sidebar.checkbox("Monthly Real GDP", value=True)
    include_unrate = st.sidebar.checkbox("Unemployment Rate(%)", value=True)
    include_cpi = st.sidebar.checkbox("Consumer Price Index", value=True)

    # Collect the selected variables into a list
    selected_regressors = []
    if include_gdp:
        selected_regressors.append('Monthly Real GDP Index')
    if include_unrate:
        selected_regressors.append('UNRATE(%)')
    if include_cpi:
        selected_regressors.append('CPI Value')

    # Submit to run the model
    if st.sidebar.button("Submit"):

        if model_type == "LSTM":
            #future_predictions_df, rmse, mae, mape, r2 = run_lstm_model(df, selected_series, selected_regressors, sequence_length, epochs, batch_size, units, dropout_rate, future_exog_df, future_steps)
            future_predictions_df, rmse, mae, mape, r2 = run_lstm_model(df, selected_series, sequence_length, epochs, batch_size, units, dropout_rate, future_steps)

        elif model_type == "Prophet":
            future_predictions_df, rmse, mae, mape, r2 = run_prophet_model(df, selected_series, selected_regressors, periods, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode,
                            yearly_option, weekly_option, daily_option, 'MS', 0.8, future_exog_df, future_steps)

        elif model_type == "Random Forest":
            future_predictions_df, rmse, mae, mape, r2 = run_rf_model(df, selected_series, selected_regressors, n_estimators, future_exog_df, future_steps)

        elif model_type == "XGBoost":
            future_predictions_df, rmse, mae, mape, r2 = run_xgboost_model(df, selected_series, learning_rate, n_estimators, max_depth, min_child_weight, future_exog_df, future_steps)

        elif model_type == "Holt Winters":
            future_predictions_df, rmse, mae, mape, r2 = run_hw_model(df, selected_series, selected_regressors, future_exog_df, future_steps, lag_feature)

        elif model_type == "SARIMAX":
            future_predictions_df, rmse, mae, mape, r2 = run_sarima_model(df, selected_series, selected_regressors, future_exog_df, future_steps, order, seasonal_order)

        insights = generate_insights(selected_series, rmse, mae, mape, r2, future_predictions_df, forecast_type)
        
        # Display insights
        st.write(insights)