import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from keras.regularizers import l2
from keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import streamlit as st
import pandas as pd

np.random.seed(6450)

def run_lstm_model(df, selected_series, sequence_length, epochs, batch_size, units, dropout_rate, future_steps):
    # Initialize the scaler
    target_scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the historical data
    scaled_target = target_scaler.fit_transform(df[[selected_series]].values)

    # Define function to create sequences
    def create_sequences(data, sequence_length):
        X, y = [], []
        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    # Create sequences using the scaled data
    X, y = create_sequences(scaled_target, sequence_length)

    # Split data into training (80%) and testing (20%)
    split_index = int(len(df) * 0.8) - sequence_length
    X_train_full, X_test = X[:split_index], X[split_index:len(df)-sequence_length]
    y_train_full, y_test = y[:split_index], y[split_index:len(df)-sequence_length]

    # Sliding window validation
    validation_size = int(len(X_train_full) * 0.2)
    X_train, X_val = X_train_full[:-validation_size], X_train_full[-validation_size:]
    y_train, y_val = y_train_full[:-validation_size], y_train_full[-validation_size:]

    # Reshape input to be [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2]))
    X_val = X_val.reshape((X_val.shape[0], X_val.shape[1], X_val.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2]))

    # Prepare future data for iterative forecasting
    X_future = X_test[-1:]

    model = Sequential()
    model.add(LSTM(units=units, return_sequences=True, 
                   input_shape=(X_train.shape[1], X_train.shape[2]), 
                   kernel_regularizer=l2(0.001)))  # L2 regularization
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=units, return_sequences=False, kernel_regularizer=l2(0.001)))  # L2 regularization
    model.add(Dense(units=1))
    model.compile(optimizer=Adam(learning_rate=0.0005), loss='mean_squared_error')  # Reduced learning rate

    # Train the model with sliding window validation
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)  # Reduced patience
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val), 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        callbacks=[early_stop], 
                        verbose=2)

    # Plot train/validation loss across epochs
    st.subheader("Train vs Validation Loss")
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    st.pyplot(plt)

    # Make predictions
    train_predict = model.predict(X_train)
    val_predict = model.predict(X_val)
    test_predict = model.predict(X_test)

    # Inverse transform predictions and actual values
    train_predict = target_scaler.inverse_transform(train_predict)
    val_predict = target_scaler.inverse_transform(val_predict)
    test_predict = target_scaler.inverse_transform(test_predict)
    y_train_inv = target_scaler.inverse_transform(y_train.reshape(-1, 1))
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))
    y_test_inv = target_scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calculate residuals and standard deviation for confidence intervals
    residuals = y_test_inv.flatten() - test_predict.flatten()
    residual_std = np.std(residuals)

    # Calculate evaluation metrics for test data
    rmse = math.sqrt(mean_squared_error(y_test_inv, test_predict[:, 0]))
    mae = mean_absolute_error(y_test_inv, test_predict[:, 0])
    mape = mean_absolute_percentage_error(y_test_inv, test_predict[:, 0]) * 100
    r2 = r2_score(y_test_inv, test_predict[:, 0]) * 100

    # Residual plot: residuals vs. predicted values
    st.subheader("Residual Plot: Predicted vs Residuals")

    plt.figure(figsize=(10, 6))
    plt.scatter(test_predict, residuals, color='blue', alpha=0.6, label='Residuals')
    plt.axhline(y=0, color='red', linestyle='--', label='Zero Residual Line')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predicted Values')
    plt.legend()
    st.pyplot(plt)

    # Generate future predictions iteratively
    future_predictions = []
    current_input = X_future

    for step in range(future_steps):
        next_pred = model.predict(current_input)
        future_predictions.append(next_pred[0, 0])

        # Reshape the prediction to match input dimensions
        next_pred = next_pred.reshape((1, 1, 1))
        current_input = np.append(current_input[:, 1:, :], next_pred, axis=1)

    # Inverse transform the future predictions
    future_predict = target_scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()

    # Calculate 95% prediction intervals with scaling for future uncertainty
    confidence_interval = 1.96 * residual_std
    interval_scaling = np.sqrt(np.arange(1, future_steps + 1))
    lower_bound = future_predict - (confidence_interval * interval_scaling)
    upper_bound = future_predict + (confidence_interval * interval_scaling)

    st.subheader("Model Training & Test Prediction")

    # Plot evaluation results
    plt.figure(figsize=(12, 6))
    plt.plot(df.index[sequence_length:len(y_train_inv) + sequence_length], y_train_inv, label='Actual Train', color='blue')
    plt.plot(df.index[len(y_train_inv) + sequence_length:len(y_train_inv) + len(y_val_inv) + sequence_length], y_val_inv, label='Actual Validation', color='purple')
    plt.plot(df.index[len(y_train_inv) + sequence_length:len(y_train_inv) + len(y_val_inv) + sequence_length], val_predict, label='Predicted Validation', color='pink')
    plt.plot(df.index[len(y_train_inv) + len(y_val_inv) + sequence_length:], y_test_inv, label='Actual Test', color='orange')
    plt.plot(df.index[len(y_train_inv) + len(y_val_inv) + sequence_length:], test_predict, label='Predicted Test', color='green')
    plt.xlabel('Month')
    plt.ylabel(f'{selected_series}')
    plt.title('Actual vs Predicted')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Test Evaluation Metrics")

    st.write(f"*RMSE:* {rmse:.2f}")
    st.write(f"*MAE:* {mae:.2f}")
    st.write(f"*MAPE:* {mape:.2f}%")
    st.write(f"*R-Squared:* {r2:.2f}%")

    # Plot future predictions with prediction intervals
    future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')

    st.subheader("Future Predictions")

    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df[selected_series], label='Historical Sales', color='blue')
    plt.plot(future_index, future_predict, label='Future Predictions', linestyle='--', color='green')
    plt.fill_between(future_index, lower_bound, upper_bound, color='pink', alpha=0.25, label='95% CI')
    plt.xlabel('Month')
    plt.ylabel(f'{selected_series}')
    plt.title(f'Future {selected_series}')
    plt.legend()
    st.pyplot(plt)

    # Create DataFrame for future predictions with intervals
    future_predictions_df = pd.DataFrame({
        'Month': future_index,
        f'Predicted {selected_series}': future_predict,
        'Lower Bound': lower_bound,
        'Upper Bound': upper_bound
    })
    future_predictions_df['Month'] = pd.to_datetime(future_predictions_df['Month']).dt.date

    return future_predictions_df, rmse, mae, mape, r2
