import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
import seaborn as sns
import streamlit as st

np.random.seed(6450)

def run_rf_model(df, selected_series, selected_regressors, future_exog_df, future_steps):
    def make_stationary(data, target_col):
        """
        Make a time series stationary by differencing.
        """
        data['diff'] = data[target_col].diff()
        data.dropna(inplace=True)
        return data

    def revert_differencing(data, predictions, target_col):
        """
        Reconstruct the original data from differenced values.
        """
        last_actual_value = data[target_col].iloc[-len(predictions) - 1]
        original_predictions = [last_actual_value + predictions[0]]
        for i in range(1, len(predictions)):
            original_predictions.append(original_predictions[-1] + predictions[i])
        return original_predictions

    def random_forest_with_feature_engineering(df, selected_series, future_exog_df, future_steps):
        """
        Train a Random Forest model with lag and rolling features, perform hyperparameter tuning, and predict future values.
        """
        df = make_stationary(df, selected_series)

        # Add lag and rolling features
        for i in range(1, 3):
            df[f'lag_{i}'] = df['diff'].shift(i)
        df['rolling_3'] = df['diff'].rolling(window=3).mean()
        df['rolling_6'] = df['diff'].rolling(window=6).mean()

        df.dropna(inplace=True)

        # Define features and target
        features = [f'lag_{i}' for i in range(1, 3)] + ['rolling_3', 'rolling_6'] + selected_regressors
        X = df[features]
        y = df['diff']

        # Split data into train and test sets
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Set up hyperparameter grid and random search
        param_distributions = {
            'n_estimators': range(50, 400, 50),
            'max_depth': range(10, 51, 10),
            'min_samples_split': np.arange(2, 21, 2),
            'min_samples_leaf': np.arange(1, 11, 1),
            'max_features': [0.2, 0.5, 0.7, 'sqrt', 'log2'],
            'bootstrap': [True, False],
            'criterion': ['squared_error', 'absolute_error', 'poisson'],
            'max_samples': np.linspace(0.5, 1.0, 10).tolist(),
            'min_impurity_decrease': np.logspace(-3, -1, 5)
        }

        # Execute random search
        rf_model = RandomForestRegressor(random_state=42)
        random_search = RandomizedSearchCV(
            estimator=rf_model,
            param_distributions=param_distributions,
            n_iter=50,
            scoring='neg_mean_squared_error',
            cv=10,
            random_state=42,
            n_jobs=-1,
            return_train_score=True
        )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Display best hyperparameters
        st.write("**Best Hyperparameters:**")
        st.write(random_search.best_params_)

        # Predict on train and test data
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)
        corrected_train_predictions = train_predictions + np.mean(y_train - train_predictions)
        corrected_test_predictions = test_predictions + np.mean(y_test - test_predictions)

        # Revert differencing and visualize results
        train_original = revert_differencing(df, corrected_train_predictions, selected_series)
        test_original = revert_differencing(df, corrected_test_predictions, selected_series)

        plt.figure(figsize=(10, 6))
        plt.plot(df[selected_series].index, df[selected_series], label="Actual Data", color='blue')
        plt.plot(df.index[len(df) - len(test_original):], test_original, label="Predicted Test", color='green')
        plt.title("Random Forest Predictions vs Actual")
        plt.xlabel('Date')
        plt.ylabel(selected_series)
        plt.legend()
        plt.show()

        # Calculate and display evaluation metrics
        rmse = np.sqrt(mean_squared_error(df[selected_series][len(df) - len(test_original):], test_original))
        mae = mean_absolute_error(df[selected_series][len(df) - len(test_original):], test_original)
        mape = mean_absolute_percentage_error(df[selected_series][len(df) - len(test_original):], test_original) * 100
        r2 = r2_score(df[selected_series][len(df) - len(test_original):], test_original) * 100

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2f} %")
        st.write(f"**R2:** {r2:.2f} %")

        # Prepare and plot future predictions
        future_predictions = []
        last_sequence = X.iloc[-1:].copy()

        for i in range(future_steps):
            for regressor in selected_regressors:
                last_sequence[regressor] = future_exog_df.iloc[i][regressor]
            next_pred = best_model.predict(last_sequence)
            future_predictions.append(next_pred[0])
            # Update features for next step
            last_sequence['rolling_3'] = np.mean(future_predictions[-3:])
            last_sequence['rolling_6'] = np.mean(future_predictions[-6:])

        future_predictions_inv = revert_differencing(df, future_predictions, selected_series)
        future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='M')

        plt.figure(figsize=(10, 6))
        plt.plot(df[selected_series].index, df[selected_series], label='Historical Data', color='blue')
        plt.plot(future_index, future_predictions_inv, label='Future Predictions', linestyle='--', color='green')
        plt.xlabel('Date')
        plt.ylabel(selected_series)
        plt.title(f'Future Predictions for {selected_series}')
        plt.legend()
        plt.show()

        # Return data frame with future predictions and metrics
        future_predictions_df = pd.DataFrame({
            'Date': future_index,
            f'Predicted {selected_series}': future_predictions_inv
        })
        future_predictions_df['Date'] = future_predictions_df['Date'].dt.date

        return future_predictions_df, rmse, mae, mape, r2

    return random_forest_with_feature_engineering(df, selected_series, future_exog_df, future_steps)
