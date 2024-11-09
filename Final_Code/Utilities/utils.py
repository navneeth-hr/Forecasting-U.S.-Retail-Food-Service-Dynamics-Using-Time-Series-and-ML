import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import math


def plot_data(data, title):
    st.subheader("Historical Sales Data")
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data.values, label=title)
    plt.xlabel('Month', fontsize=12)
    plt.ylabel(title, fontsize=12)
    plt.title(f'Historical {title}', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend()
    st.pyplot(plt)


def display_error_metrics(actual, predicted):
    rmse = math.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    r2 = r2_score(actual, predicted) * 100
    return rmse, mae, mape, r2


#def plot_lstm_predictions(df, y_train_inv, y_test_inv, test_predict, sequence_length, selected_series):
#    st.subheader("LSTM Model Predictions vs Actual Data")
#    
#    plt.figure(figsize=(12, 6))
#    plt.plot(df.index[sequence_length: len(y_train_inv[0]) + sequence_length], y_train_inv[0], label='Actual Train Data')
#    plt.plot(df.index[len(y_train_inv[0]) + sequence_length: len(y_train_inv[0]) + len(y_test_inv[0]) + sequence_length], y_test_inv[0], label='Actual Test Data')
#    plt.plot(df.index[len(y_train_inv[0]) + sequence_length: len(y_train_inv[0]) + len(y_test_inv[0]) + sequence_length], test_predict, label='Predicted Test Data')
#    plt.xlabel('Month')
#    plt.ylabel(selected_series)
#    plt.title(f'Actual vs Predicted {selected_series} Sales')
#    plt.legend()
#    st.pyplot(plt)
#
#
#def plot_rf_predictions(actual, predicted):
#    st.subheader("Random Forest Model Predictions vs Actual Data")
#    
#    plt.figure(figsize=(10, 6))
#    plt.plot(actual.values, label="Actual Sales", color='blue')
#    plt.plot(predicted, label="Predicted Sales", color='orange')
#    plt.xlabel('Month')
#    plt.ylabel("Sales")
#    plt.title("Random Forest Predictions vs Actual Sales")
#    plt.legend()
#    st.pyplot(plt)