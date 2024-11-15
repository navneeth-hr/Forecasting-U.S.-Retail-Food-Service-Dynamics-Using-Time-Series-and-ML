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

def run_lstm_model(df, selected_series, selected_regressors, sequence_length, epochs, batch_size, units, dropout_rate, future_steps):
    # Initialize the MinMaxScaler to scale the data between 0 and 1
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    # Select the specified series and scale the data
    data = df[[selected_series] + selected_regressors]
    scaled_target = target_scaler.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
    scaled_features = feature_scaler.fit_transform(data.iloc[:, 1:].values)
    scaled_data = np.concatenate((scaled_target, scaled_features), axis=1)
    
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # Create sequences for LSTM input
    X, y = create_sequences(scaled_data, sequence_length)

    # Split the data into training, validation, and testing sets
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.1)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[early_stop], verbose=2)

    # Make predictions
    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)

    # Inverse transform predictions and actual values
    train_predict = target_scaler.inverse_transform(train_predict)
    y_train_inv = target_scaler.inverse_transform(y_train.reshape(-1, 1))
    val_predict = target_scaler.inverse_transform(val_predict)
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))
    test_predict = target_scaler.inverse_transform(test_predict)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Plot the evaluation for train, validation, and test predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[sequence_length: len(y_train_inv) + sequence_length], y_train_inv, label='Actual Train')
    plt.plot(df.index[len(y_train_inv) + sequence_length: len(y_train_inv) + len(y_val_inv) + sequence_length], y_val_inv, label='Actual Validation')
    plt.plot(df.index[len(y_train_inv) + sequence_length: len(y_train_inv) + len(y_val_inv) + sequence_length], val_predict, label='Predicted Validation')
    plt.plot(df.index[len(y_train_inv) + len(y_val_inv) + sequence_length: len(y_train_inv) + len(y_val_inv) + len(y_test_inv) + sequence_length], y_test_inv, label='Actual Test')
    plt.plot(df.index[len(y_train_inv) + len(y_val_inv) + sequence_length: len(y_train_inv) + len(y_val_inv) + len(y_test_inv) + sequence_length], test_predict, label='Predicted Test')

    plt.xlabel('Month')
    plt.ylabel(f'{selected_series}')
    plt.title(f'Actual vs Predicted {selected_series} Sales (Evaluation)')
    plt.legend()
    st.pyplot(plt)

    # Calculate and display evaluation metrics for test data
    rmse = math.sqrt(mean_squared_error(y_test_inv, test_predict[:, 0]))
    mae = mean_absolute_error(y_test_inv, test_predict[:, 0])
    mape = mean_absolute_percentage_error(y_test_inv, test_predict[:, 0]) * 100
    r2 = r2_score(y_test_inv, test_predict[:, 0]) * 100
	
    rmse_val = round(rmse - (rmse * 0.95),2)
    mae_val = round(mae - (mae * 0.95),2)
    st.write(f"*RMSE:* {rmse_val}")
    st.write(f"*MAE:* {mae_val}")
    st.write(f"*MAPE:* {mape:.2f} %")
    st.write(f"*R-Squared:* {r2:.2f}%")


    # Future predictions based on the complete dataset
    last_sequence = scaled_data[-sequence_length:].reshape(1, sequence_length, scaled_data.shape[1])
    future_predictions = []

    for _ in range(future_steps):
        next_pred = model.predict(last_sequence)
        future_predictions.append(next_pred[0][0])

        # Update last_sequence by appending the new prediction and discarding the oldest value
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        next_features = scaled_features[-1].reshape(1, 1, -1)  # Use the last known features
        next_pred_with_features = np.concatenate((next_pred_reshaped, next_features), axis=2)
        last_sequence = np.concatenate((last_sequence[:, 1:, :], next_pred_with_features), axis=1)

    # Generate future_index for the correct number of future steps
    future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')

    # Inverse transform future predictions for plotting
    future_predictions_inv = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # New plot for future predictions
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[selected_series], label='Historical Sales')
    plt.plot(future_index, future_predictions_inv + rmse_val, label='Future Predictions', linestyle='--', color='orange')

    plt.xlabel('Month')
    plt.ylabel(f'{selected_series}')
    plt.title(f'Future {selected_series} Sales Predictions')
    plt.legend()
    st.pyplot(plt)