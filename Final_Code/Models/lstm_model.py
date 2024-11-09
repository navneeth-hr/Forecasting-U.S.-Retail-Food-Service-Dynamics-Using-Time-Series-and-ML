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


def run_lstm_model(df, selected_series, sequence_length, epochs, batch_size, units, dropout_rate, future_steps):
    """
    Run an LSTM (Long Short-Term Memory) model on the selected time series data.

    This function preprocesses the data, creates and trains an LSTM model,
    makes predictions, and visualizes the results.

    Parameters:
    df (pandas.DataFrame): The input dataframe containing the time series data.
    selected_series (str): The name of the column in df to be used for prediction.
    sequence_length (int): The number of time steps to use as input for each prediction.
    epochs (int): The number of epochs to train the model.
    batch_size (int): The batch size for training the model.
    units (int): The number of LSTM units in each layer of the model.

    Returns:
    None. The function prints metrics and displays plots as side effects.
    """

    # Initialize the MinMaxScaler to scale the data between 0 and 1
    target_scaler = MinMaxScaler(feature_range=(0, 1))
    feature_scaler = MinMaxScaler(feature_range=(0, 1))

    # Select the specified series and scale the data
    data = df[[selected_series] +  ['Monthly Real GDP Index', 'UNRATE(%)', 'CPI Value']]
    scaled_target = target_scaler.fit_transform(data.iloc[:, 0].values.reshape(-1, 1))
    scaled_features = feature_scaler.fit_transform(data.iloc[:, 1:].values)
    scaled_data = np.concatenate((scaled_target, scaled_features), axis=1)
    
    def create_sequences(data, sequence_length):
        """
        Create input sequences and corresponding labels for the LSTM model.

        Parameters:
        data (numpy.array): The scaled input data.
        sequence_length (int): The number of time steps to use as input for each prediction.

        Returns:
        X (numpy.array): Input sequences.
        y (numpy.array): Corresponding labels.
        """
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # Create sequences for LSTM input
    X, y = create_sequences(scaled_data, sequence_length)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Build the LSTM model
    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=2)

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform predictions and actual values
    train_predict = target_scaler.inverse_transform(train_predict)
    y_train_inv = target_scaler.inverse_transform(y_train.reshape(-1,1))
    test_predict = target_scaler.inverse_transform(test_predict)
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1,1))

    # Ensure sequence_length does not exceed the length of y_test_inv
    sequence_length = min(sequence_length, len(y_test_inv))

    # Initialize last_sequence with actual test values and features
    last_sequence = scaled_data[-sequence_length:].copy()
    last_sequence[:, 0] = y_test_inv[-sequence_length:].flatten()  # Replace target column with actual test values
    last_sequence = last_sequence.reshape(1, sequence_length, scaled_data.shape[1])

    future_predictions = []

    for _ in range(future_steps):
        # Predict the next step
        next_pred = model.predict(last_sequence)
        future_predictions.append(next_pred[0][0])

        # Prepare the next input sequence by appending the predicted value
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        next_features = last_sequence[:, -1, 1:].reshape(1, 1, -1)  # Keep the features from the last timestep
        next_pred_with_features = np.concatenate((next_pred_reshaped, next_features), axis=2)
                
        # Update last_sequence to prepare for the next prediction
        last_sequence = np.append(last_sequence[:, 1:, :], next_pred_with_features, axis=1)

    # Inverse transform the future predictions
    future_predictions_inv = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

    # Plotting results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[sequence_length: len(y_train_inv) + sequence_length], y_train_inv, label='Actual Train')
    plt.plot(df.index[len(y_train_inv) + sequence_length: len(y_train_inv) + len(y_test_inv) + sequence_length], y_test_inv, label='Actual Test')
    plt.plot(df.index[len(y_train_inv) + sequence_length: len(y_train_inv) + len(y_test_inv) + sequence_length], test_predict, label='Predicted Test')

    future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='ME')
    plt.plot(future_index, future_predictions_inv, label='Future Predictions', linestyle='--', color='red')

    plt.xlabel('Month')
    plt.ylabel(f'{selected_series}')
    plt.title(f'Actual vs Predicted {selected_series} Sales')
    plt.legend()
    st.pyplot(plt)

    # Calculate and display evaluation metrics
    rmse = math.sqrt(mean_squared_error(y_test_inv, test_predict[:, 0]))
    mae = mean_absolute_error(y_test_inv, test_predict[:, 0])
    mape = mean_absolute_percentage_error(y_test_inv, test_predict[:, 0]) * 100
    r2 = r2_score(y_test_inv, test_predict[:, 0]) * 100
    st.write(f"**RMSE:** {rmse}")
    st.write(f"**MAE:** {mae}")
    st.write(f"**MAPE:** {mape:.2f} %")
    st.write(f"**R-Squared:** {r2:.2f} %")