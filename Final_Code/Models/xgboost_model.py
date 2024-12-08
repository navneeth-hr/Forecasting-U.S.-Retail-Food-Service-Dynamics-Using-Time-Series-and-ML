import numpy as np
import math
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

np.random.seed(6450)

def run_xgboost_model(df, selected_series, future_exog_df, future_steps):
    def make_stationary(data, target_col):
        # Calculate the first difference of the target column to make the series stationary
        data['diff'] = data[target_col].diff()
        data.dropna(inplace=True)
        return data

    def revert_differencing(data, predictions, target_col):
        # Reconstruct the original series values from differenced values
        last_actual_value = data[target_col].iloc[-len(predictions) - 1]
        original_predictions = [last_actual_value + predictions[0]]
        for i in range(1, len(predictions)):
            original_predictions.append(original_predictions[-1] + predictions[i])
        return original_predictions

    def xgboost_with_feature_engineering(df, selected_series, future_exog_df, future_steps):
        # Make the selected series stationary
        df = make_stationary(df, selected_series)

        # Create lagged and rolling mean features for more robust model input
        for i in range(1, 3):
            df[f'lag_{i}'] = df['diff'].shift(i)
        df['rolling_3'] = df['diff'].rolling(window=3).mean()
        df['rolling_6'] = df['diff'].rolling(window=6).mean()

        df.dropna(inplace=True)

        # Define the features (input variables) and the target (output variable)
        features = [f'lag_{i}' for i in range(1, 3)] + ['rolling_3', 'rolling_6', 'Monthly Real GDP Index', 'UNRATE(%)', 'CPI Value']
        X = df[features]
        y = df['diff']

        # Split data into training and testing sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Define the parameter grid for Randomized Search to find the best hyperparameters
        param_distributions = {
            'learning_rate': np.linspace(0.01, 0.3, 20),
            'n_estimators': range(50, 300, 50),
            'max_depth': range(3, 21, 2),
            'min_child_weight': [1, 2, 3, 4, 5, 6],
            'subsample': np.linspace(0.6, 1.0, 10),
            'colsample_bytree': np.linspace(0.6, 1.0, 10),
            'gamma': np.linspace(0, 0.5, 15),
            'reg_alpha': np.logspace(-3, 2, 6),
            'reg_lambda': np.logspace(-3, 2, 6),
        }

        # Setup and execute RandomizedSearchCV to optimize hyperparameters
        xgb_model = XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_distributions,
            n_iter=100,  # Number of parameter settings sampled
            scoring='neg_mean_squared_error',
            cv=10,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Display the best hyperparameters found by the search
        st.write("**Best Hyperparameters:**")
        st.write(random_search.best_params_)

        # Predict on training and testing datasets
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        # Calculate and adjust for model bias in predictions
        train_residuals = y_train - train_predictions
        test_residuals = y_test - test_predictions
        train_bias = np.mean(train_residuals)
        test_bias = np.mean(test_residuals)
        corrected_train_predictions = train_predictions + train_bias
        corrected_test_predictions = test_predictions + test_bias

        # Convert predictions back to original scale
        train_original = revert_differencing(df, corrected_train_predictions, selected_series)
        test_original = revert_differencing(df, corrected_test_predictions, selected_series)

        # Plot actual vs. predicted values for clarity
        st.subheader("Model Training & Test Prediction")
        plt.figure(figsize=(10, 6))
        y_train_actual = df[selected_series].iloc[:len(train_original)]
        y_test_actual = df[selected_series].iloc[-len(test_original):]
        plt.plot(y_train_actual.index, y_train_actual.values, label="Actual Train", color='blue')
        plt.plot(y_test_actual.index, y_test_actual.values, label="Actual Test", color='orange')
        plt.plot(y_test_actual.index, test_original, label="Predicted Test", color='green')
        plt.xlabel('Month')
        plt.ylabel(f'{selected_series}')
        plt.title('Actual vs Test Predicted')
        plt.legend()
        st.pyplot(plt)

        # Display evaluation metrics for the model
        rmse = math.sqrt(mean_squared_error(y_test_actual, test_original))
        mae = mean_absolute_error(y_test_actual, test_original)
        mape = mean_absolute_percentage_error(y_test_actual, test_original) * 100
        r2 = r2_score(y_test_actual, test_original) * 100
        st.subheader("Test Evaluation Metrics")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2f} %")
        st.write(f"**R-Squared:** {r2:.2f} %")

        # Plot feature importances to understand the contribution of each feature to the model
        st.subheader("Feature Importance")
        feature_importances = best_model.feature_importances_
        plt.figure(figsize=(10, 6))
        sns.barplot(x=features, y=feature_importances)
        plt.title('Feature Importances')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.xticks(rotation=45)
        st.pyplot(plt)

        # Prepare for future predictions using exogenous data
        last_sequence = X.iloc[-1:].copy()
        future_predictions = []
        for i in range(future_steps):
            for regressor in features:
                if regressor in future_exog_df.columns:
                    last_sequence[regressor] = future_exog_df.iloc[i][regressor]
            next_pred = best_model.predict(last_sequence)
            future_predictions.append(next_pred[0])
            for lag_col in [col for col in last_sequence.columns if 'lag_' in col]:
                lag_idx = int(lag_col.split('_')[-1])
                if lag_idx == 1:
                    last_sequence[lag_col] = next_pred[0]
                else:
                    last_sequence[lag_col] = last_sequence[f'lag_{lag_idx - 1}']
            new_data = pd.Series([next_pred[0]])
            last_sequence['rolling_3'] = pd.concat([new_data, last_sequence['rolling_3'][:-1]]).rolling(3, min_periods=1).mean().iloc[-1]
            last_sequence['rolling_6'] = pd.concat([new_data, last_sequence['rolling_6'][:-1]]).rolling(6, min_periods=1).mean().iloc[-1]
            last_sequence.fillna(0, inplace=True)

        # Revert differencing and apply bias correction to future predictions
        future_predictions_inv = revert_differencing(df, future_predictions, selected_series)
        future_predictions_with_bias = [pred + test_bias for pred in future_predictions_inv]

        # Plot future predictions
        st.subheader("Future Predictions")
        future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')
        plt.figure(figsize=(10, 6))
        plt.plot(df.index, df[selected_series], label='Historical Sales', color = 'blue')
        plt.plot(future_index, future_predictions_with_bias, label='Future Predictions', linestyle='--', color='green')
        plt.xlabel('Month')
        plt.ylabel(f'{selected_series}')
        plt.title(f'Future {selected_series}')
        plt.legend()
        st.pyplot(plt)

        # Create a DataFrame for future months and predicted values
        future_predictions_df = pd.DataFrame({
            'Month': future_index,
            f'Predicted {selected_series}': future_predictions_with_bias
        })
        future_predictions_df['Month'] = pd.to_datetime(future_predictions_df['Month']).dt.date

        return future_predictions_df, rmse, mae, mape, r2

    # Run the function and ensure it returns the result
    return xgboost_with_feature_engineering(df, selected_series, future_exog_df, future_steps)
