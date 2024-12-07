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
        data['diff'] = data[target_col].diff()
        data.dropna(inplace=True)
        return data

    def revert_differencing(data, predictions, target_col):
        last_actual_value = data[target_col].iloc[-len(predictions) - 1]
        original_predictions = [last_actual_value + predictions[0]]
        for i in range(1, len(predictions)):
            original_predictions.append(original_predictions[-1] + predictions[i])
        return original_predictions

    def xgboost_with_feature_engineering(df, selected_series, future_exog_df, future_steps):
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

        # Grid search for hyperparameter tuning
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

        xgb_model = XGBRegressor(random_state=42)
        random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=100,  # Number of random combinations to try
        scoring='neg_mean_squared_error',
        cv=10,
        random_state=42,
        n_jobs=-1,
        return_train_score=True
    )
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_

        # Display cross-validation results
        # st.subheader("Cross-Validation Results")
        # cv_results = pd.DataFrame(random_search.cv_results_)
        # cv_summary = cv_results[['param_learning_rate', 'param_n_estimators', 'param_max_depth', 
        #                           'param_min_child_weight', 'mean_test_score', 'std_test_score']]
        # cv_summary['mean_test_score'] = -cv_summary['mean_test_score']  # Convert to positive MSE
        # cv_summary.rename(columns={
        #     'param_learning_rate': 'Learning Rate',
        #     'param_n_estimators': 'n_estimators',
        #     'param_max_depth': 'max_depth',
        #     'param_min_child_weight': 'min_child_weight',
        #     'mean_test_score': 'Mean Test MSE',
        #     'std_test_score': 'Std Dev Test MSE'
        # }, inplace=True)
        # st.dataframe(cv_summary)

        # Display best hyperparameters
        st.write("**Best Hyperparameters:**")
        st.write(random_search.best_params_)

        # Predictions for train and test sets
        train_predictions = best_model.predict(X_train)
        test_predictions = best_model.predict(X_test)

        # Calculate residuals
        train_residuals = y_train - train_predictions
        test_residuals = y_test - test_predictions

        # Calculate Mean Bias (bias)
        train_bias = np.mean(train_residuals)
        test_bias = np.mean(test_residuals)

        # Correct the forecast by adding the bias
        corrected_train_predictions = train_predictions + train_bias
        corrected_test_predictions = test_predictions + test_bias

        # Revert differencing
        train_original = revert_differencing(df, corrected_train_predictions, selected_series)
        test_original = revert_differencing(df, corrected_test_predictions, selected_series)

        # Plot for train and test predictions
        y_train_actual = df[selected_series].iloc[:len(train_original)]
        y_test_actual = df[selected_series].iloc[-len(test_original):]

        st.subheader("Model Training & Test Prediction")

        plt.figure(figsize=(10, 6))
        plt.plot(y_train_actual.index, y_train_actual.values, label="Actual Train", color='blue')
        plt.plot(y_test_actual.index, y_test_actual.values, label="Actual Test", color='orange')
        plt.plot(y_test_actual.index, test_original, label="Predicted Test", color='green')
        plt.xlabel('Month')
        plt.ylabel(f'{selected_series}')
        plt.title('Actual vs Test Predicted')
        plt.legend()
        st.pyplot(plt)

        # Evaluation metrics for test data
        rmse = math.sqrt(mean_squared_error(y_test_actual, test_original))
        mae = mean_absolute_error(y_test_actual, test_original)
        mape = mean_absolute_percentage_error(y_test_actual, test_original) * 100
        r2 = r2_score(y_test_actual, test_original) * 100

        st.subheader("Test Evaluation Metrics")

        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**MAE:** {mae:.2f}")
        st.write(f"**MAPE:** {mape:.2f} %")
        st.write(f"**R-Squared:** {r2:.2f} %")

        st.subheader("Feature Importance")

        # Feature importance plot
        feature_importances = best_model.feature_importances_

        plt.figure(figsize=(10, 6))
        sns.barplot(x=features, y=feature_importances)
        plt.title('Feature Importances')
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
            next_pred = best_model.predict(last_sequence)
            future_predictions.append(next_pred[0])

            # Shift and update lag features
            for lag_col in [col for col in last_sequence.columns if 'lag_' in col]:
                lag_idx = int(lag_col.split('_')[-1])  # Get lag number
                if lag_idx == 1:
                    last_sequence[lag_col] = next_pred[0]
                else:
                    last_sequence[lag_col] = last_sequence[f'lag_{lag_idx - 1}']

            # Update rolling features
            new_data = pd.Series([next_pred[0]])  # New prediction as a series
            last_sequence['rolling_3'] = (
                pd.concat([new_data, last_sequence['rolling_3'][:-1]]).rolling(3, min_periods=1).mean().iloc[-1]
            )
            last_sequence['rolling_6'] = (
                pd.concat([new_data, last_sequence['rolling_6'][:-1]]).rolling(6, min_periods=1).mean().iloc[-1]
            )

            # Fill any NaN values with 0 or another appropriate value
            last_sequence.fillna(0, inplace=True)
            
        # Revert differencing for future predictions
        future_predictions_inv = revert_differencing(df, future_predictions, selected_series)

        # Apply the bias correction to the future predictions
        future_predictions_with_bias = [pred + test_bias for pred in future_predictions_inv]

        future_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(), periods=future_steps, freq='MS')
        
        st.subheader("Future Predictions")
        
        # Plot future predictions
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

    # Ensure the function returns the result
    return xgboost_with_feature_engineering(df, selected_series, future_exog_df, future_steps)
