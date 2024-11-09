import numpy as np
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import GridSearchCV
import seaborn as sns


def run_rf_model(df, selected_series, n_estimators, future_steps):
    def make_stationary(data, target_col):
        data['diff'] = data[target_col].diff()
        data.dropna(inplace=True)
        return data

    def revert_differencing(data, predictions, target_col):
        last_actual_value = data[target_col].iloc[-len(predictions)-1]
        original_predictions = [last_actual_value + predictions[0]]
        for i in range(1, len(predictions)):
            original_predictions.append(original_predictions[-1] + predictions[i])
        return original_predictions

    def random_forest_with_feature_engineering(df, selected_series, n_estimators):
        df = make_stationary(df, selected_series)

        for i in range(1, 3):
            df[f'lag_{i}'] = df['diff'].shift(i)

        df['rolling_3'] = df['diff'].rolling(window=3).mean()
        df['rolling_6'] = df['diff'].rolling(window=6).mean()

        df.dropna(inplace=True)

        X = df[[f'lag_{i}' for i in range(1, 3)] + ['rolling_3', 'rolling_6', 'Monthly Real GDP Index', 'UNRATE(%)', 'CPI Value']]
        y = df['diff']

        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        param_grid = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf_model = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_

        differenced_predictions = best_model.predict(X_test)

        original_predictions = revert_differencing(df, differenced_predictions, selected_series)

        y_test_actual = df[selected_series].iloc[-len(differenced_predictions):]
        rmse = math.sqrt(mean_squared_error(y_test_actual, original_predictions))
        mae = mean_absolute_error(y_test_actual, original_predictions)
        mape = mean_absolute_percentage_error(y_test_actual, original_predictions) * 100
        r2 = r2_score(y_test_actual, original_predictions) * 100

        plt.figure(figsize=(10, 6))
        plt.plot(y_test_actual.values, label="True Sales", color='blue')
        plt.plot(original_predictions, label="Predicted Sales", color='orange')
        plt.title("Random Forest Predictions vs Actual (Original Data)")
        plt.legend()
        st.pyplot(plt)

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2f} %")
        st.write(f"**R-Squared:** {r2:.2f} %")

        feature_importances = best_model.feature_importances_

        plt.figure(figsize=(10, 6))
        sns.barplot(x=X.columns, y=feature_importances)
        plt.title('Feature Importances in Random Forest Model')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        st.pyplot(plt)

    random_forest_with_feature_engineering(df, selected_series, n_estimators)