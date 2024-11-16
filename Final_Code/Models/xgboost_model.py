import numpy as np
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

np.random.seed(6450)

def run_xgboost_model(df, selected_series, learning_rate, n_estimators, max_depth, min_child_weight, future_exog_df, future_steps):
    def make_stationary(data, target_col):
        data['diff'] = data[target_col].diff()
        data.dropna(inplace=True)
        return data

    def revert_differencing(data, predictions, target_col):
        last_actual_value = data[target_col].iloc[-len(predictions) - 1]
        original_predictions = [last_actual_value + predictions[0]]
        for i in range(1, len(predictions)):
            original_predictions.append(original_predictions[-1] + predictions[i])
        return original_predictions

    def xgboost_with_feature_engineering(df, selected_series, learning_rate, n_estimators, max_depth, min_child_weight, future_exog_df, future_steps):
        df = make_stationary(df, selected_series)

        # Add lag and rolling features
        for i in range(1, 3):
            df[f'lag_{i}'] = df['diff'].shift(i)
        df['rolling_3'] = df['diff'].rolling(window=3).mean()
        df['rolling_6'] = df['diff'].rolling(window=6).mean()

        df.dropna(inplace=True)

        # Define features and target
        features = [f'lag_{i}' for i in range(1, 3)] + ['rolling_3', 'rolling_6', 'Monthly Real GDP Index', 'UNRATE(%)', 'CPI Value']
        X = df[features]
        y = df['diff']

        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define the XGBoost model with hyperparameters
        xgb_model = XGBRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            random_state=42
        )

        # Fit the model
        xgb_model.fit(X_train, y_train)

        # Predictions for train and test sets
        train_predictions = xgb_model.predict(X_train)
        test_predictions = xgb_model.predict(X_test)

        # Revert differencing
        train_original = revert_differencing(df, train_predictions, selected_series)
        test_original = revert_differencing(df, test_predictions, selected_series)

        # Plot for train and test predictions
        y_train_actual = df[selected_series].iloc[:len(train_original)]
        y_test_actual = df[selected_series].iloc[-len(test_original):]

        plt.figure(figsize=(10, 6))
        plt.plot(y_train_actual.index, y_train_actual.values, label="True Train Sales", color='blue')
        plt.plot(y_test_actual.index, y_test_actual.values, label="True Test Sales", color='red')
        plt.plot(y_test_actual.index, test_original, label="Predicted Test Sales", color='brown')
        plt.title("XGBoost Predictions vs Actual (Original Data)")
        plt.legend()
        st.pyplot(plt)

        # Evaluation metrics for test data
        rmse = math.sqrt(mean_squared_error(y_test_actual, test_original))
        mae = mean_absolute_error(y_test_actual, test_original)
        mape = mean_absolute_percentage_error(y_test_actual, test_original) * 100
        r2 = r2_score(y_test_actual, test_original) * 100

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2f} %")
        st.write(f"**R-Squared:** {r2:.2f} %")

        # Feature importance plot
        feature_importances = xgb_model.feature_importances_

        plt.figure(figsize=(10, 6))
        sns.barplot(x=features, y=feature_importances)
        plt.title('Feature Importances in XGBoost Model')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Prepare data for future predictions
        last_sequence = X.iloc[-1:].copy()
        future_predictions = []

        for i in range(future_steps):
            # Update exogenous features with future data
            for regressor in features:
                if regressor in future_exog_df.columns:
                    last_sequence[regressor] = future_exog_df.iloc[i][regressor]

            # Predict the next value
            next_pred = xgb_model.predict(last_sequence)
            future_predictions.append(next_pred[0])

            # Update last_sequence with the new prediction and drop the oldest lag
            last_sequence = last_sequence.shift(-1, axis=1)
            last_sequence.iloc[:, 0] = next_pred  # Update with prediction in lag_1
            last_sequence['rolling_3'] = pd.concat([pd.Series([next_pred[0]]), last_sequence['rolling_3']]).rolling(3).mean().iloc[-1]
            last_sequence['rolling_6'] = pd.concat([pd.Series([next_pred[0]]), last_sequence['rolling_6']]).rolling(6).mean().iloc[-1]

        # Revert differencing for future predictions
        future_predictions_inv = revert_differencing(df, future_predictions, selected_series)
        future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')

        # Plot future predictions
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[selected_series], label='Historical Sales')
        plt.plot(future_index, future_predictions_inv, label='Future Predictions', linestyle='--', color='green')
        plt.xlabel('Month')
        plt.ylabel(f'{selected_series}')
        plt.title(f'Future {selected_series} Sales Predictions')
        plt.legend()
        st.pyplot(plt)

        # Create a DataFrame for future months and predicted values
        future_predictions_df = pd.DataFrame({
            'Month': future_index,
            f'Predicted {selected_series}': future_predictions_inv
        })
        future_predictions_df['Month'] = pd.to_datetime(future_predictions_df['Month']).dt.date

        return future_predictions_df, rmse, mae, mape, r2

    # Ensure the function returns the result
    return xgboost_with_feature_engineering(df, selected_series, learning_rate, n_estimators, max_depth, min_child_weight, future_exog_df, future_steps)
