import matplotlib.pyplot as plt
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
import math


def plot_data(data, title):
    st.subheader("Historical Sales Data")
    plt.figure(figsize=(12, 6))
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

def generate_insights(category_name, rmse, mae, mape, r2_score, predictions_df, forecast_type):
    # 1. General Interpretation of Metrics
    metrics_summary = (
        f"### Insights for {category_name}\n"
        f"**Model Performance:**\n"
        f"- **RMSE:** {rmse:.2f} (Average deviation of predictions from actual values)\n"
        f"- **MAE:** {mae:.2f} (Average absolute error in predictions)\n"
        f"- **MAPE:** {mape:.2f}% (Percentage error, indicates forecasting accuracy)\n"
        f"- **R² Score:** {r2_score:.2f}% (Explains {r2_score:.2f}% of the variance in sales data)\n\n"
    )

    # 2. Business Interpretation
    if mape < 10:
        accuracy_comment = "The model shows high accuracy with a MAPE below 10%, making it reliable for forecasting."
    elif 10 <= mape < 20:
        accuracy_comment = "The model has moderate accuracy. Predictions can be used for strategic planning but should be monitored closely."
    else:
        accuracy_comment = "The model has lower accuracy with a high MAPE. It may require additional tuning or feature enhancements."

    # Analyze trend
    trend_growth = predictions_df[f'Predicted {category_name}'].pct_change().mean()
    if trend_growth > 0:
        trend_comment = "Predicted sales values indicate a consistent growth trend, suggesting strong consumer demand in this category."
        trend_type = "growth"
    elif trend_growth < 0:
        trend_comment = "The forecast shows a decline, indicating potential challenges in consumer demand."
        trend_type = "decline"
    else:
        trend_comment = "The forecast suggests stagnation, with little to no change in sales trends."
        trend_type = "stagnation"

    # 3. Future Sales Analysis with dynamic forecast length
    next_months = predictions_df

    # Set the number of months to forecast based on the selected forecast type
    if forecast_type == "Short-term":
        forecast_length = len(predictions_df)
    elif forecast_type == "Long-term":
        forecast_length = len(predictions_df)
    else:
        forecast_length = 0  # If no valid forecast type is chosen

    # Generate future comment dynamically based on forecast length
    future_comment = f"**{forecast_type} Outlook:**\n"
    for i in range(forecast_length):
        future_comment += f"- **{next_months.iloc[i, 0]}:** Predicted Sales = ${next_months.iloc[i, 1]:,.2f}\n"

    future_comment += "\n"  # Add a newline at the end for better formatting


    # 4. Recommendations based on trend
    if trend_type == "growth":
        recommendation = (
            "### Recommendations:\n"
            "- Retailers should prepare for increased demand by optimizing inventory levels and enhancing marketing strategies, especially for the upcoming season.\n"
            "- Policymakers may consider this trend as an indicator of economic recovery and increased consumer confidence.\n"
            "- Investors could view this as an opportunity for strategic investments in retail stocks or related sectors.\n"
        )
    elif trend_type == "decline":
        recommendation = (
            "### Recommendations:\n"
            "- Retailers should be cautious and may need to adjust inventory levels and reduce operational costs to mitigate potential losses.\n"
            "- Policymakers might need to explore economic stimulus measures or investigate factors contributing to reduced consumer spending.\n"
            "- Investors should consider a conservative approach, possibly diversifying investments to hedge against potential downturns in this sector.\n"
        )
    else:  # stagnation
        recommendation = (
            "### Recommendations:\n"
            "- Retailers may consider maintaining current inventory levels but should remain flexible to adapt to sudden changes in demand.\n"
            "- Policymakers might view this trend as a signal of stable consumer behavior, with no immediate signs of economic growth or decline.\n"
            "- Investors should monitor the sector closely for any signs of emerging trends or shifts in consumer behavior before making significant investments.\n"
        )

    # Combine all sections
    insights = metrics_summary + accuracy_comment + "\n\n" + trend_comment + "\n\n" + future_comment + recommendation
    return insights
