
from datetime import datetime
import pandas as pd


# ============================================================
# 1. IMPORT ALL PROJECT MODULES
# ============================================================

from predict_traffic_volume import predict_traffic

from bottleneck_prediction import (
    get_available_features,
    analyze_bottleneck
)

from predict_congestion_duration import (
    predict_congestion_duration
)

from predict_next_5_days import (
    predict_future_traffic
)


# ============================================================
# 2. DISPLAY AVAILABLE BOTTLENECK FEATURES
# ============================================================

def display_available_bottleneck_features():
    """
    Returns the list of features available
    for historical bottleneck analysis.
    """

    return get_available_features()


# ============================================================
# 3. HELPER FUNCTION FOR DATAFRAME DISPLAY
# ============================================================

def display_dataframe(data):

    if isinstance(data, pd.DataFrame):
        print(data.to_string(index=False))

    elif data is not None:
        print(data)

    else:
        print("No data available.")


# ============================================================
# 4. MAIN TRAFFIC INTELLIGENCE SYSTEM
# ============================================================

def run_complete_traffic_analysis(

    temperature,
    rainfall,
    snowfall,
    cloud_cover,
    timestamp_input,
    weather,
    weather_desc,
    holiday,
    bottleneck_feature,
    future_days=5

):

    print("\n" + "=" * 75)

    print(
        "          AI-BASED TRAFFIC INTELLIGENCE SYSTEM"
    )

    print("=" * 75)

    print("\nPrediction Starting Time:", timestamp_input)

    print("\nInitializing all prediction modules...")

    # ========================================================
    # MODULE 1: CURRENT TRAFFIC VOLUME PREDICTION
    # ========================================================

    print("\n" + "=" * 75)

    print(
        "1. CURRENT TRAFFIC VOLUME PREDICTION"
    )

    print("=" * 75)

    current_traffic_result = predict_traffic(

        temperature=temperature,

        rainfall=rainfall,

        snowfall=snowfall,

        cloud_cover=cloud_cover,

        timestamp_input=timestamp_input,

        weather=weather,

        weather_desc=weather_desc,

        holiday=holiday

    )

    if not isinstance(current_traffic_result, dict):

        raise TypeError(
            "Current traffic prediction must return a dictionary."
        )

    print("\nCurrent Traffic Prediction Result:")

    print(current_traffic_result)

    # ========================================================
    # MODULE 2: TRAFFIC BOTTLENECK ANALYSIS
    # ========================================================

    print("\n" + "=" * 75)

    print(
        "2. TRAFFIC BOTTLENECK ANALYSIS"
    )

    print("=" * 75)

    current_features = {

        "Temperature": temperature,

        "Rainfall_last_hour": rainfall,

        "Snowfall_last_hour": snowfall,

        "Cloud_Cover": cloud_cover,

        "Weather": weather,

        "Weather_Desc": weather_desc,

        "IsHoliday": holiday

    }

    bottleneck_result = analyze_bottleneck(

        selected_feature=bottleneck_feature,

        timestamp_input=timestamp_input,

        current_features=current_features

    )

    if not isinstance(bottleneck_result, dict):

        raise TypeError(
            "Bottleneck analysis must return a dictionary."
        )

    print("\nBottleneck Result Keys:")

    print(list(bottleneck_result.keys()))

    # Extract bottleneck information

    bottleneck_insight = bottleneck_result.get(
        "bottleneck_insight",
        {}
    )

    traffic_summary = bottleneck_result.get(
        "traffic_summary"
    )

    cluster_information = bottleneck_result.get(
        "cluster_information"
    )

    print("\nSelected Bottleneck Feature:")

    print(
        bottleneck_insight.get(
            "selected_feature",
            bottleneck_feature
        )
    )

    print("\nHistorical Traffic Summary:")

    display_dataframe(traffic_summary)

    print("\nBottleneck Insight:")

    print(
        "Bottleneck Detected:",
        bottleneck_insight.get(
            "bottleneck_detected",
            "N/A"
        )
    )

    print(
        "Feature Value:",
        bottleneck_insight.get(
            "feature_value",
            "N/A"
        )
    )

    print(
        "Average Traffic Volume:",
        bottleneck_insight.get(
            "average_traffic_volume",
            "N/A"
        )
    )

    print(
        "Maximum Traffic Volume:",
        bottleneck_insight.get(
            "maximum_traffic_volume",
            "N/A"
        )
    )

    print(
        "Message:",
        bottleneck_insight.get(
            "message",
            "Bottleneck analysis completed."
        )
    )

    print("\nDBSCAN Cluster Information:")

    display_dataframe(cluster_information)

    # ========================================================
    # MODULE 3: CONGESTION DURATION PREDICTION
    # ========================================================

    print("\n" + "=" * 75)

    print(
        "3. CONGESTION DURATION PREDICTION"
    )

    print("=" * 75)

    congestion_result = predict_congestion_duration(

        timestamp_input=timestamp_input

    )

    if not isinstance(congestion_result, dict):

        raise TypeError(
            "Congestion prediction must return a dictionary."
        )

    print("\nSelected Timestamp:")

    print(
        congestion_result.get(
            "selected_timestamp",
            timestamp_input
        )
    )

    print("\nPredicted Traffic Volume:")

    print(
        congestion_result.get(
            "predicted_traffic_volume",
            "N/A"
        )
    )

    print("\nCongestion Level:")

    congestion_level = congestion_result.get(
        "congestion_level",
        "Unknown"
    )

    print(str(congestion_level).upper())

    print("\nExpected Congestion Duration:")

    duration_hours = congestion_result.get(
        "duration_hours",
        0
    )

    print(duration_hours, "Hour(s)")

    print("\nExpected Congestion Until:")

    print(
        congestion_result.get(
            "congestion_end_time",
            "N/A"
        )
    )

    print("\nNext Traffic Condition:")

    print(
        str(
            congestion_result.get(
                "next_congestion_level",
                "Unknown"
            )
        ).upper()
    )

    print("\nCongestion Forecast:")

    display_dataframe(
        congestion_result.get("forecast")
    )

    # ========================================================
    # MODULE 4: FUTURE TRAFFIC FORECAST
    # ========================================================

    print("\n" + "=" * 75)

    print(
        f"4. NEXT {future_days} DAYS TRAFFIC FORECAST"
    )

    print("=" * 75)

    if not isinstance(future_days, int) or future_days < 1:

        raise ValueError(
            "Future forecast days must be a positive integer."
        )

    forecast_result = predict_future_traffic(

        timestamp_input=timestamp_input,

        future_days=future_days

    )

    if not isinstance(forecast_result, dict):

        raise TypeError(
            "Future traffic prediction must return a dictionary."
        )

    print("\nForecast Starting Timestamp:")

    print(
        forecast_result.get(
            "selected_timestamp",
            timestamp_input
        )
    )

    print("\nTotal Predictions:")

    print(
        forecast_result.get(
            "total_predictions",
            "N/A"
        )
    )

    print("\nHistorical Records Used:")

    print(
        forecast_result.get(
            "historical_records_used",
            "N/A"
        )
    )

    print("\nTraffic Forecast:")

    forecast_dataframe = forecast_result.get(
        "forecast"
    )

    display_dataframe(forecast_dataframe)

    # ========================================================
    # 5. FINAL TRAFFIC INTELLIGENCE SUMMARY
    # ========================================================

    print("\n" + "=" * 75)

    print(
        "          FINAL TRAFFIC INTELLIGENCE SUMMARY"
    )

    print("=" * 75)

    # Current traffic

    print("\n1. CURRENT TRAFFIC VOLUME")

    print(
        current_traffic_result
    )

    # Bottleneck analysis

    print("\n2. BOTTLENECK ANALYSIS")

    print(
        bottleneck_insight.get(
            "message",
            "Bottleneck analysis completed."
        )
    )

    # Congestion duration

    print("\n3. CONGESTION DURATION")

    print(
        "Traffic Level:",
        str(
            congestion_result.get(
                "congestion_level",
                "Unknown"
            )
        ).upper()
    )

    print(
        "Duration:",
        congestion_result.get(
            "duration_hours",
            "N/A"
        ),
        "hours"
    )

    print(
        "Expected Until:",
        congestion_result.get(
            "congestion_end_time",
            "N/A"
        )
    )

    # Future forecast

    print("\n4. FUTURE TRAFFIC FORECAST")

    print(
        "Forecast Period:",
        future_days,
        "Days"
    )

    if isinstance(forecast_dataframe, pd.DataFrame):

        print(
            "Total Hourly Predictions:",
            len(forecast_dataframe)
        )

    else:

        print(
            "Total Hourly Predictions:",
            "N/A"
        )

    print("\n" + "=" * 75)

    print(
        "       TRAFFIC INTELLIGENCE ANALYSIS COMPLETED"
    )

    print("=" * 75)

    # ========================================================
    # 6. RETURN ALL RESULTS TO STREAMLIT
    # ========================================================

    final_results = {

        "timestamp": timestamp_input,

        "current_traffic": current_traffic_result,

        "bottleneck_analysis": bottleneck_result,

        "congestion_duration": congestion_result,

        "future_traffic_forecast": forecast_result

    }

    return final_results


# ============================================================
# 7. MAIN PROGRAM EXECUTION
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # CURRENT DATE AND TIME
    # --------------------------------------------------------

    timestamp_input = datetime.now().strftime(
        "%d/%m/%y %H:%M"
    )

    # --------------------------------------------------------
    # WEATHER INPUTS
    # --------------------------------------------------------

    temperature = 23

    rainfall = 0

    snowfall = 0

    cloud_cover = 50

    weather = "Cloudy skies"

    weather_desc = "Clear skies"

    holiday = "no"

    # --------------------------------------------------------
    # BOTTLENECK FEATURE SELECTION
    # --------------------------------------------------------

    features = get_available_features()

    if not features:

        raise ValueError(
            "No bottleneck analysis features are available."
        )

    bottleneck_feature = features[0]

    # --------------------------------------------------------
    # FUTURE FORECAST PERIOD
    # --------------------------------------------------------

    future_days = 5

    # --------------------------------------------------------
    # RUN COMPLETE TRAFFIC ANALYSIS
    # --------------------------------------------------------

    final_result = run_complete_traffic_analysis(

        temperature=temperature,

        rainfall=rainfall,

        snowfall=snowfall,

        cloud_cover=cloud_cover,

        timestamp_input=timestamp_input,

        weather=weather,

        weather_desc=weather_desc,

        holiday=holiday,

        bottleneck_feature=bottleneck_feature,

        future_days=future_days

    )

    # --------------------------------------------------------
    # DISPLAY FINAL RESULT KEYS
    # --------------------------------------------------------

    print("\nFinal Result Dictionary Keys:")

    print(
        list(final_result.keys())
    )
