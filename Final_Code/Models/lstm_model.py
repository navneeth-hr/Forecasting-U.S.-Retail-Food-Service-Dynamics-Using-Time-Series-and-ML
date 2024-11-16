import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import streamlit as st
import pandas as pd

np.random.seed(6450)

def run_lstm_model(df, selected_series, selected_regressors, sequence_length, epochs, batch_size, units, dropout_rate, future_exog_df, future_steps):
    # Initialize scalers
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the historical data
    data = df[[selected_series] + selected_regressors]
    scaled_target = target_scaler.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
    scaled_features = feature_scaler.fit_transform(data.iloc[:, 1:].values)
    scaled_data = np.concatenate((scaled_target, scaled_features), axis=1)

    # Scale future exogenous variables
    future_scaled_features = feature_scaler.transform(future_exog_df[selected_regressors])
    future_scaled_data = np.concatenate((np.zeros((len(future_exog_df), 1)), future_scaled_features), axis=1)

    # Concatenate historical and future data
    full_scaled_data = np.concatenate((scaled_data, future_scaled_data), axis=0)
    full_df = pd.concat([df, future_exog_df], axis=0)

    # Define function to create sequences
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # Create sequences using the full scaled data
    X, y = create_sequences(full_scaled_data, sequence_length)

    # Split data into training (80%) and testing (20%)
    split_index = int(len(df) * 0.8) - sequence_length
    X_train, X_test = X[:split_index], X[split_index:len(df)-sequence_length]
    y_train, y_test = y[:split_index], y[split_index:len(df)-sequence_length]

    # Prepare future data for forecasting
    X_future = X[len(df)-sequence_length:]

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))
    X_future = X_future.reshape((X_future.shape[0], X_future.shape[1], X_future.shape[2]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[early_stop], verbose=2)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    future_predict = model.predict(X_future)

    # Inverse transform predictions and actual values
    train_predict = target_scaler.inverse_transform(train_predict)
    y_train_inv = target_scaler.inverse_transform(y_train.reshape(-1, 1))
    test_predict = target_scaler.inverse_transform(test_predict)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))
    future_predict = target_scaler.inverse_transform(future_predict[:future_steps])
    future_predict = future_predict.flatten()

    # Plot evaluation results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[sequence_length:len(y_train_inv) + sequence_length], y_train_inv, label='Actual Train')
    plt.plot(df.index[len(y_train_inv) + sequence_length:], y_test_inv, label='Actual Test')
    plt.plot(df.index[len(y_train_inv) + sequence_length:], test_predict, label='Predicted Test')
    plt.xlabel('Month')
    plt.ylabel(f'{selected_series}')
    plt.title(f'Actual vs Predicted {selected_series} Sales (Evaluation)')
    plt.legend()
    st.pyplot(plt)

    # Calculate evaluation metrics
    rmse = math.sqrt(mean_squared_error(y_test_inv, test_predict[:, 0]))
    mae = mean_absolute_error(y_test_inv, test_predict[:, 0])
    mape = mean_absolute_percentage_error(y_test_inv, test_predict[:, 0]) * 100
    r2 = r2_score(y_test_inv, test_predict[:, 0]) * 100

    st.write(f"*RMSE:* {rmse:.2f}")
    st.write(f"*MAE:* {mae:.2f}")
    st.write(f"*MAPE:* {mape:.2f}%")
    st.write(f"*R-Squared:* {r2:.2f}%")

    # Plot future predictions
    future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[selected_series], label='Historical Sales')
    plt.plot(future_index, future_predict, label='Future Predictions', linestyle='--', color='orange')
    plt.xlabel('Month')
    plt.ylabel(f'{selected_series}')
    plt.title(f'Future {selected_series} Sales Predictions')
    plt.legend()
    st.pyplot(plt)

    future_predictions_df = pd.DataFrame({
            'Month': future_index,
            f'Predicted {selected_series}': future_predict
        })

    # Show the table with future months and predictions
    st.write("### Future Predictions Table")
    st.dataframe(future_predictions_df)
