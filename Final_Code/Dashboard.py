import streamlit as st
import pandas as pd
from Utilities.utils import plot_data, display_error_metrics
from Models.lstm_model import run_lstm_model
from Models.prophet_model import run_prophet_model
from Models.random_forest_model import run_rf_model
from Models.holts_winter_model import run_hw_model
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
    
    category_type = st.sidebar.selectbox(
        "Select Category Type", 
        ["Retail sales, Adjusted(Millions of Dollars)", "Food services and drinking places, Adjusted(Millions of Dollars)"]
    )

    if category_type == "Retail sales, Adjusted(Millions of Dollars)":
        selected_series = st.sidebar.selectbox(
            "Select Retail Sub-Category", 
            [
                "Retail sales, Adjusted(Millions of Dollars)", 
                "Motor vehicle and parts dealers, Adjusted(Millions of Dollars)",
                "Furniture and home furnishings stores, Adjusted(Millions of Dollars)",
                "Electronics and appliance stores, Adjusted(Millions of Dollars)",
                "Building mat. and garden equip. and supplies dealers, Adjusted(Millions of Dollars)",
                "Food and beverage stores, Adjusted(Millions of Dollars)",
                "Health and personal care stores, Adjusted(Millions of Dollars)",
                "Gasoline stations, Adjusted(Millions of Dollars)",
                "Clothing and clothing access. Stores, Adjusted(Millions of Dollars)",
                "Sporting goods, hobby, musical instrument, and book stores, Adjusted(Millions of Dollars)",
                "General merchandise stores, Adjusted(Millions of Dollars)",
                "Miscellaneous store retailers, Adjusted(Millions of Dollars)",
                "Nonstore retailers, Adjusted(Millions of Dollars)"
            ]
        )
    else:
        selected_series = "Food services and drinking places, Adjusted(Millions of Dollars)"
    

    # Initialize session state for storing selections
    if 'forecast_type' not in st.session_state:
        st.session_state.forecast_type = "Short-term"
    if 'forecast_period' not in st.session_state:
        st.session_state.forecast_period = 3
    if 'model_type' not in st.session_state:
        st.session_state.model_type = "Holts Winter"


    # Forecast Section
    st.sidebar.header("Forecast Configuration")
    
    # Use key parameter to update session state
    forecast_type = st.sidebar.radio("Select Forecast Type", ["Short-term", "Long-term"], key="forecast_type")

    if forecast_type == "Short-term":
        future_steps = st.sidebar.slider("Future Steps (months)", min_value=1, max_value=12, value=st.session_state.forecast_period, key="forecast_period")
        model_type = st.sidebar.selectbox("Select Model", ["Holts Winter", "SARIMAX"], key="model_type")
        st.sidebar.info("Holts Winter and SARIMAX are recommended for short-term forecasting up to 12 months.")
    else:
        future_steps = st.sidebar.slider("Future Steps (months)", min_value=12, max_value=48, value=max(st.session_state.forecast_period, 12), key="forecast_period")
        model_type = st.sidebar.selectbox("Select Model", ["Prophet", "LSTM", "Random Forest"], key="model_type")
        st.sidebar.info("Prophet and LSTM are recommended for complex long-term forecasting up to 4 years and Random forest to determine the non-linear relationship")

    # Hyperparameter Tuning
    if model_type == "LSTM":
        if selected_series == "Retail sales, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=150)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Food services and drinking places, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=95)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=32)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.2)
        elif selected_series == "Motor vehicle and parts dealers, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Furniture and home furnishings stores, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Electronics and appliance stores, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Building mat. and garden equip. and supplies dealers, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Food and beverage stores, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Health and personal care stores, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Gasoline stations, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Clothing and clothing access. Stores, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Sporting goods, hobby, musical instrument, and book stores, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "General merchandise stores, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Miscellaneous store retailers, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        elif selected_series == "Nonstore retailers, Adjusted(Millions of Dollars)":
            sequence_length = st.sidebar.slider("Sequence Length (LSTM)", min_value=12, max_value=150, value=100)
            epochs = st.sidebar.slider("Epochs (LSTM)", min_value=5, max_value=50, value=20)
            batch_size = st.sidebar.slider("Batch Size (LSTM)", min_value=1, max_value=64, value=2)
            units = st.sidebar.slider("Units (LSTM)", min_value=50, max_value=300, value=100)
            dropout_rate = st.sidebar.slider("Dropout rate (LSTM)", min_value=0.1, max_value=1.0, value=0.1)
        
    elif model_type == "Prophet":
        if selected_series == "Retail sales, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=10.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=24)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"], index=0)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Food services and drinking places, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.000, max_value=1.00, value=0.001)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=10.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"], index=1)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Motor vehicle and parts dealers, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.9)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=6.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"],index=0)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Furniture and home furnishings stores, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.2)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=7.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=18)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"],index=0)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Electronics and appliance stores, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.01)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=10.0)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=24)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"],index=1)
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Building mat. and garden equip. and supplies dealers, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Food and beverage stores, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Health and personal care stores, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Gasoline stations, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Clothing and clothing access. Stores, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Sporting goods, hobby, musical instrument, and book stores, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "General merchandise stores, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Miscellaneous store retailers, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)
        elif selected_series == "Nonstore retailers, Adjusted(Millions of Dollars)":
            changepoint_prior_scale = st.sidebar.slider("Changepoint Prior Scale (Prophet)", min_value=0.01, max_value=1.0, value=0.05)
            seasonality_prior_scale=st.sidebar.slider("Seasonality Prior Scale (Prophet)", min_value=0.01, max_value=15.0, value=0.05)
            periods = st.sidebar.slider("Periods (Prophet)", min_value=12, max_value=36, value=12)
            seasonality_mode = st.sidebar.selectbox("Seasonality Mode (Prophet)", ["additive", "multiplicative"])
            yearly_option = st.sidebar.radio("Yearly Seasonality:", [True, False], horizontal=True, index=0)
            weekly_option = st.sidebar.radio("Weekly Seasonality:", [True, False], horizontal=True, index=0)
            daily_option = st.sidebar.radio("Daily Seasonality:", [True, False], horizontal=True, index=1)

    elif model_type == "Random Forest":
        if selected_series == "Retail sales, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Food services and drinking places, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Motor vehicle and parts dealers, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Furniture and home furnishings stores, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Electronics and appliance stores, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Building mat. and garden equip. and supplies dealers, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Food and beverage stores, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Health and personal care stores, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Gasoline stations, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Clothing and clothing access. Stores, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Sporting goods, hobby, musical instrument, and book stores, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "General merchandise stores, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Miscellaneous store retailers, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)
        elif selected_series == "Nonstore retailers, Adjusted(Millions of Dollars)":
            n_estimators = st.sidebar.slider("Number of Estimators (Random Forest)", min_value=50, max_value=500, value=100, step=10)

    # Display selected data series
    if selected_series in df.columns:
        data = df[[selected_series]]
        st.write(f"Displaying {selected_series} Data")
        plot_data(data, selected_series)
    else:
        st.error(f"Selected series '{selected_series}' not found in the dataset.")

    # Submit to run the model
    if st.sidebar.button("Submit"):

        if model_type == "LSTM":
            run_lstm_model(df, selected_series, sequence_length, epochs, batch_size, units, dropout_rate, future_steps)

        elif model_type == "Prophet":
            run_prophet_model(df, selected_series, regressors, periods, changepoint_prior_scale, seasonality_prior_scale, seasonality_mode,
                            yearly_option, weekly_option, daily_option, 'MS', 0.8)

        elif model_type == "Random Forest":
            run_rf_model(df, selected_series, n_estimators, future_steps)

        elif model_type == "Holts Winter":
            run_hw_model(df, selected_series, future_steps)

        elif model_type == "SARIMAX":
            run_sarima_model(df, selected_series, future_steps)