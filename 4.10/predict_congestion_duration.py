
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ============================================================
# 1. LOAD TRAINED GRU MODEL
# ============================================================

MODEL_DIR = "saved_gru_congestion_model"

print("Loading GRU Congestion Duration Model...")

model = tf.keras.models.load_model(
    f"{MODEL_DIR}/gru_congestion_duration.keras"
)

feature_scaler = joblib.load(
    f"{MODEL_DIR}/feature_scaler.pkl"
)

target_scaler = joblib.load(
    f"{MODEL_DIR}/target_scaler.pkl"
)

feature_columns = joblib.load(
    f"{MODEL_DIR}/feature_columns.pkl"
)

N_PAST = joblib.load(
    f"{MODEL_DIR}/n_past.pkl"
)

N_FUTURE = joblib.load(
    f"{MODEL_DIR}/n_future.pkl"
)

traffic_thresholds = joblib.load(
    f"{MODEL_DIR}/traffic_thresholds.pkl"
)

history_df = pd.read_pickle(
    f"{MODEL_DIR}/processed_history.pkl"
)


# ============================================================
# 2. LOAD TRAFFIC THRESHOLDS
# ============================================================

low_threshold = traffic_thresholds[
    "low_threshold"
]

moderate_threshold = traffic_thresholds[
    "moderate_threshold"
]


# ============================================================
# 3. PREPARE HISTORICAL DATA
# ============================================================

history_df["TimeStamp"] = pd.to_datetime(
    history_df["TimeStamp"]
)

history_df = history_df.sort_values(
    "TimeStamp"
).reset_index(
    drop=True
)

print("GRU Congestion Duration Model loaded successfully!")


# ============================================================
# 4. CLASSIFY TRAFFIC
# ============================================================

def classify_traffic(volume):
    """
    Classify traffic volume into:
    Low, Moderate, or High.
    """

    if volume <= low_threshold:

        return "Low"

    elif volume <= moderate_threshold:

        return "Moderate"

    else:

        return "High"


# ============================================================
# 5. PREDICT CONGESTION DURATION
# ============================================================

def predict_congestion_duration(timestamp_input):
    """
    Predict traffic volume and congestion duration.

    Input format:
    DD/MM/YY HH:MM

    Example:
    13/11/26 17:23

    Returns:
    - Selected timestamp
    - Predicted traffic volume
    - Congestion level
    - Duration in hours
    - Congestion end time
    - Next congestion level
    - Future traffic forecast
    """

    # --------------------------------------------------------
    # STEP 1: VALIDATE TIMESTAMP
    # --------------------------------------------------------

    selected_timestamp = pd.to_datetime(
        timestamp_input,
        format="%d/%m/%y %H:%M",
        errors="coerce"
    )

    if pd.isna(selected_timestamp):

        raise ValueError(
            "Invalid date format. "
            "Use DD/MM/YY HH:MM. "
            "Example: 13/11/26 17:23"
        )

    # --------------------------------------------------------
    # STEP 2: SELECT HISTORICAL DATA
    # --------------------------------------------------------

    historical_data = history_df[
        history_df["TimeStamp"] < selected_timestamp
    ].copy()

    if len(historical_data) < N_PAST:

        raise ValueError(
            f"Not enough historical data. "
            f"The model requires {N_PAST} "
            f"previous hourly records before "
            f"{selected_timestamp}."
        )

    historical_data = historical_data.tail(
        N_PAST
    ).copy()

    # --------------------------------------------------------
    # STEP 3: SELECT MODEL FEATURES
    # --------------------------------------------------------

    X_input_df = historical_data[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # STEP 4: SCALE INPUT DATA
    # --------------------------------------------------------

    X_input_scaled = feature_scaler.transform(
        X_input_df
    )

    # --------------------------------------------------------
    # STEP 5: RESHAPE INPUT FOR GRU MODEL
    # --------------------------------------------------------

    X_input = X_input_scaled.reshape(
        1,
        N_PAST,
        len(feature_columns)
    )

    # --------------------------------------------------------
    # STEP 6: PREDICT FUTURE TRAFFIC VOLUME
    # --------------------------------------------------------

    predicted_scaled = model.predict(
        X_input,
        verbose=0
    )

    # Convert predictions back to original scale

    predicted_volume = target_scaler.inverse_transform(
        predicted_scaled.reshape(
            -1,
            1
        )
    ).flatten()

    # Prevent negative traffic predictions

    predicted_volume = np.maximum(
        predicted_volume,
        0
    )

    # --------------------------------------------------------
    # STEP 7: GENERATE FUTURE TIMESTAMPS
    # --------------------------------------------------------

    future_timestamps = pd.date_range(
        start=selected_timestamp,
        periods=len(predicted_volume),
        freq="h"
    )

    # --------------------------------------------------------
    # STEP 8: CLASSIFY FUTURE TRAFFIC
    # --------------------------------------------------------

    traffic_levels = [

        classify_traffic(volume)

        for volume in predicted_volume

    ]

    # --------------------------------------------------------
    # STEP 9: CREATE FORECAST DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame({

        "TimeStamp":
            future_timestamps,

        "Predicted_Traffic_Volume":
            predicted_volume,

        "Congestion_Level":
            traffic_levels

    })

    # --------------------------------------------------------
    # STEP 10: GET CURRENT TRAFFIC CONDITION
    # --------------------------------------------------------

    current_volume = float(
        predicted_volume[0]
    )

    current_level = traffic_levels[0]

    # --------------------------------------------------------
    # STEP 11: CALCULATE CONGESTION DURATION
    # --------------------------------------------------------

    duration_hours = 0

    for level in traffic_levels:

        if level == current_level:

            duration_hours += 1

        else:

            break

    # --------------------------------------------------------
    # STEP 12: CALCULATE CONGESTION END TIME
    # --------------------------------------------------------

    congestion_end_time = (

        selected_timestamp +

        pd.Timedelta(
            hours=duration_hours
        )

    )

    # --------------------------------------------------------
    # STEP 13: PREDICT NEXT TRAFFIC CONDITION
    # --------------------------------------------------------

    if duration_hours < len(traffic_levels):

        next_level = traffic_levels[
            duration_hours
        ]

    else:

        next_level = (
            "Not available within forecast window"
        )

    # --------------------------------------------------------
    # STEP 14: PREPARE DISPLAY FORECAST
    # --------------------------------------------------------

    display_forecast = results_df.copy()

    display_forecast[
        "Predicted_Traffic_Volume"
    ] = display_forecast[
        "Predicted_Traffic_Volume"
    ].round(2)

    # --------------------------------------------------------
    # STEP 15: RETURN RESULTS
    # --------------------------------------------------------

    result = {

        "selected_timestamp":
            selected_timestamp.strftime(
                "%d/%m/%Y %H:%M"
            ),

        "predicted_traffic_volume":
            round(
                current_volume,
                2
            ),

        "congestion_level":
            current_level,

        "duration_hours":
            duration_hours,

        "congestion_end_time":
            congestion_end_time.strftime(
                "%d/%m/%Y %H:%M"
            ),

        "next_congestion_level":
            next_level,

        "forecast":
            display_forecast,

        "thresholds": {

            "low_threshold":
                float(low_threshold),

            "moderate_threshold":
                float(moderate_threshold)

        }

    }

    return result


# ============================================================
# 6. GET MODEL INFORMATION
# ============================================================

def get_congestion_model_info():
    """
    Returns GRU model configuration information.
    """

    return {

        "model_type":
            "GRU",

        "historical_hours_required":
            int(N_PAST),

        "future_hours_predicted":
            int(N_FUTURE),

        "low_threshold":
            float(low_threshold),

        "moderate_threshold":
            float(moderate_threshold),

        "feature_columns":
            list(feature_columns)

    }


# ============================================================
# 7. TEST THE MODEL
# ============================================================

if __name__ == "__main__":

    from datetime import datetime

    # Use current date and time

    timestamp = datetime.now().strftime(
        "%d/%m/%y %H:%M"
    )

    result = predict_congestion_duration(
        timestamp
    )

    print("\n")
    print("=" * 70)

    print(
        "CONGESTION DURATION PREDICTION RESULT"
    )

    print("=" * 70)

    print(
        "\nSelected Date & Time:",
        result["selected_timestamp"]
    )

    print(
        "\nPredicted Traffic Volume:",
        result["predicted_traffic_volume"]
    )

    print(
        "\nCurrent Congestion Level:",
        result["congestion_level"].upper()
    )

    print(
        "\nExpected Duration:",
        result["duration_hours"],
        "Hour(s)"
    )

    print(
        "\nExpected Until:",
        result["congestion_end_time"]
    )

    print(
        "\nExpected Next Condition:",
        result["next_congestion_level"].upper()
    )

    print("\n")
    print("=" * 70)

    print(
        "NEXT CONGESTION FORECAST"
    )

    print("=" * 70)

    print(
        result["forecast"].to_string(
            index=False
        )
    )

    print("=" * 70)

    print(
        "\nTotal Forecast Records:",
        len(result["forecast"])
    )
