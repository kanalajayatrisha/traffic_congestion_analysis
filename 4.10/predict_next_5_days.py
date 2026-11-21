
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf


# ============================================================
# 1. LOAD TRAINED LSTM FORECASTING MODEL
# ============================================================

MODEL_DIR = "saved_forecasting_model"

print("Loading forecasting model...")

model = tf.keras.models.load_model(
    f"{MODEL_DIR}/traffic_lstm.keras"
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

print("Forecasting model loaded successfully!")


# ============================================================
# 4. CLASSIFY TRAFFIC
# ============================================================

def classify_traffic(predicted_volume):
    """
    Classify traffic volume into:
    Low, Moderate, or High.
    """

    if predicted_volume <= low_threshold:

        return "Low"

    elif predicted_volume <= moderate_threshold:

        return "Moderate"

    else:

        return "High"


# ============================================================
# 5. CREATE FUTURE FEATURES
# ============================================================

def create_future_features(timestamp, previous_row):
    """
    Create feature values for a future timestamp.

    Time-based features are updated.

    Other features are carried forward from
    the previous available record.
    """

    row = previous_row.copy()

    row["TimeStamp"] = timestamp

    # Time features

    row["Hour"] = timestamp.hour

    row["Day"] = timestamp.day

    row["Month"] = timestamp.month

    row["Year"] = timestamp.year

    row["Weekday"] = timestamp.weekday()

    row["IsWeekend"] = int(
        timestamp.weekday() >= 5
    )

    # Cyclical hour features

    row["Hour_sin"] = np.sin(
        2 * np.pi * row["Hour"] / 24
    )

    row["Hour_cos"] = np.cos(
        2 * np.pi * row["Hour"] / 24
    )

    # Cyclical day features

    row["Day_sin"] = np.sin(
        2 * np.pi * row["Day"] / 31
    )

    row["Day_cos"] = np.cos(
        2 * np.pi * row["Day"] / 31
    )

    # Cyclical month features

    row["Month_sin"] = np.sin(
        2 * np.pi * row["Month"] / 12
    )

    row["Month_cos"] = np.cos(
        2 * np.pi * row["Month"] / 12
    )

    # Time period features

    row["IsMorning"] = int(
        6 <= row["Hour"] < 12
    )

    row["IsAfternoon"] = int(
        12 <= row["Hour"] < 18
    )

    row["IsEvening"] = int(
        18 <= row["Hour"] < 24
    )

    return row


# ============================================================
# 6. PREDICT NEXT N DAYS TRAFFIC
# ============================================================

def predict_future_traffic(timestamp_input, future_days=5):
    """
    Predict traffic volume hourly for the next N days.

    Example:
    future_days = 5

    Total predictions = 5 * 24 = 120

    Timestamp format:
    DD/MM/YY HH:MM
    """

    future_days = int(future_days)

    if future_days <= 0:

        raise ValueError(
            "future_days must be greater than 0."
        )

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
            "Invalid timestamp format. "
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
            f"Not enough historical data before "
            f"{selected_timestamp}. "
            f"Required: {N_PAST} previous records."
        )

    historical_data = historical_data.tail(
        N_PAST
    ).copy()

    # --------------------------------------------------------
    # STEP 3: PREPARE INPUT SEQUENCE
    # --------------------------------------------------------

    sequence_df = historical_data[
        feature_columns
    ].copy()

    future_predictions = []

    # Five days = 120 hours

    total_hours = future_days * 24

    # --------------------------------------------------------
    # STEP 4: GENERATE HOURLY PREDICTIONS
    # --------------------------------------------------------

    for hour_step in range(total_hours):

        # Scale input features

        scaled_sequence = feature_scaler.transform(
            sequence_df
        )

        # Reshape for LSTM input

        X_input = scaled_sequence.reshape(
            1,
            N_PAST,
            len(feature_columns)
        )

        # Predict traffic volume

        predicted_scaled = model.predict(
            X_input,
            verbose=0
        )

        # Convert to original traffic volume

        predicted_volume = (
            target_scaler.inverse_transform(
                predicted_scaled.reshape(-1, 1)
            )[0][0]
        )

        predicted_volume = max(
            0,
            float(predicted_volume)
        )

        # Generate future timestamp

        future_timestamp = (
            selected_timestamp +
            pd.Timedelta(hours=hour_step)
        )

        # Classify traffic

        traffic_level = classify_traffic(
            predicted_volume
        )

        # Store prediction

        future_predictions.append({

            "TimeStamp":
                future_timestamp,

            "Predicted_Traffic_Volume":
                round(predicted_volume, 2),

            "Traffic_Level":
                traffic_level

        })

        # ----------------------------------------------------
        # STEP 5: UPDATE FUTURE INPUT FEATURES
        # ----------------------------------------------------

        previous_row = historical_data.iloc[
            -1
        ].copy()

        future_row = create_future_features(
            future_timestamp,
            previous_row
        )

        future_features = pd.DataFrame(
            [future_row]
        )[feature_columns]

        # Add future row to sequence

        sequence_df = pd.concat(
            [
                sequence_df,
                future_features
            ],
            ignore_index=True
        )

        # Keep only required historical sequence

        sequence_df = sequence_df.tail(
            N_PAST
        ).copy()

        historical_data = pd.concat(
            [
                historical_data,
                pd.DataFrame([future_row])
            ],
            ignore_index=True
        )

    # --------------------------------------------------------
    # STEP 6: CREATE FORECAST DATAFRAME
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        future_predictions
    )

    # --------------------------------------------------------
    # STEP 7: PREPARE FINAL RESULT
    # --------------------------------------------------------

    result = {

        "selected_timestamp":
            selected_timestamp.strftime(
                "%d/%m/%Y %H:%M"
            ),

        "future_days":
            future_days,

        "total_predictions":
            total_hours,

        "forecast":
            results_df,

        "low_threshold":
            float(low_threshold),

        "moderate_threshold":
            float(moderate_threshold),

        "historical_records_used":
            int(N_PAST)

    }

    return result


# ============================================================
# 7. GET FORECASTING MODEL INFORMATION
# ============================================================

def get_forecasting_model_info():
    """
    Returns forecasting model information.
    """

    return {

        "model_type":
            "LSTM",

        "historical_records_required":
            int(N_PAST),

        "low_threshold":
            float(low_threshold),

        "moderate_threshold":
            float(moderate_threshold),

        "feature_columns":
            list(feature_columns)

    }


# ============================================================
# 8. TEST THE FORECASTING MODEL
# ============================================================

if __name__ == "__main__":

    from datetime import datetime

    # Current date and time

    timestamp = datetime.now().strftime(
        "%d/%m/%y %H:%M"
    )

    future_days = 5

    result = predict_future_traffic(
        timestamp_input=timestamp,
        future_days=future_days
    )

    print("\n")
    print("=" * 70)

    print(
        f"NEXT {result['future_days']} DAYS "
        "TRAFFIC FORECAST"
    )

    print("=" * 70)

    print(
        "\nStarting Timestamp:",
        result["selected_timestamp"]
    )

    print(
        "\nHistorical Records Used:",
        result["historical_records_used"]
    )

    print(
        "\nTotal Predictions:",
        result["total_predictions"]
    )

    print(
        "\nLow Threshold:",
        result["low_threshold"]
    )

    print(
        "\nModerate Threshold:",
        result["moderate_threshold"]
    )

    print("\nFIRST 10 PREDICTIONS:")

    print(
        result["forecast"].head(10).to_string(
            index=False
        )
    )

    print("\nLAST 10 PREDICTIONS:")

    print(
        result["forecast"].tail(10).to_string(
            index=False
        )
    )

    print("\nTotal Rows Generated:")

    print(
        len(result["forecast"])
    )

    print("=" * 70)
