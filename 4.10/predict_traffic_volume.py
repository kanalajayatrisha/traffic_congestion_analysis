
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

print("Loading Traffic Prediction Model...")

# ==========================================================
# LOAD MODEL AND PREPROCESSING OBJECTS
# ==========================================================

model = tf.keras.models.load_model(
    "saved_models/traffic_lstm_model.keras"
)

scaler_features = joblib.load(
    "saved_models/feature_scaler.pkl"
)

scaler_target = joblib.load(
    "saved_models/target_scaler.pkl"
)

label_encoder_weather = joblib.load(
    "saved_models/weather_encoder.pkl"
)

label_encoder_weather_desc = joblib.load(
    "saved_models/weather_desc_encoder.pkl"
)

feature_columns = joblib.load(
    "saved_models/feature_columns.pkl"
)

n_past = joblib.load(
    "saved_models/n_past.pkl"
)

# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

train_df = pd.read_csv("Dataset/Train.csv")

print("Model and preprocessing objects loaded successfully!")


# ==========================================================
# DISPLAY AVAILABLE WEATHER CATEGORIES
# ==========================================================

def get_weather_categories():

    print("\nAvailable Weather Categories:")
    print(list(label_encoder_weather.classes_))

    print("\nAvailable Weather Description Categories:")
    print(list(label_encoder_weather_desc.classes_))


# ==========================================================
# SAFE LABEL ENCODING
# ==========================================================

def safe_transform(encoder, value):

    value = str(value).strip()

    # Match without case sensitivity, but encode
    # using the exact category stored in the encoder.
    matching_category = None

    for category in encoder.classes_:

        if str(category).strip().casefold() == value.casefold():

            matching_category = category
            break

    if matching_category is None:

        available = ", ".join(
            str(category)
            for category in encoder.classes_
        )

        raise ValueError(
            f"Unknown category: '{value}'.\n"
            f"Available categories: {available}"
        )

    return encoder.transform([matching_category])[0]


# ==========================================================
# PREPARE HISTORICAL DATA
# ==========================================================

def prepare_historical_data():

    df = train_df.copy()

    # Convert timestamp
    df["TimeStamp"] = pd.to_datetime(
        df["TimeStamp"],
        format="%d/%m/%y %H:%M"
    )

    # Time features
    df["Hour"] = df["TimeStamp"].dt.hour
    df["Day"] = df["TimeStamp"].dt.day
    df["Month"] = df["TimeStamp"].dt.month
    df["Year"] = df["TimeStamp"].dt.year
    df["Weekday"] = df["TimeStamp"].dt.weekday

    # Weekend
    df["IsWeekend"] = (
        df["Weekday"] >= 5
    ).astype(int)

    # Holiday
    df["IsHoliday"] = (
        df["Holiday"].notnull()
    ).astype(int)

    # Weather encoding
    df["Weather_encoded"] = df["Weather"].apply(
        lambda x: safe_transform(
            label_encoder_weather,
            x
        )
    )

    df["Weather_Desc_encoded"] = df["Weather_Desc"].apply(
        lambda x: safe_transform(
            label_encoder_weather_desc,
            x
        )
    )

    # Drop unused columns
    df.drop(
        [
            "Date",
            "Holiday",
            "Weather",
            "Weather_Desc"
        ],
        axis=1,
        inplace=True,
        errors="ignore"
    )

    # Fill missing values
    df = df.ffill()

    # Cyclical features
    df["Hour_sin"] = np.sin(
        2 * np.pi * df["Hour"] / 24
    )

    df["Hour_cos"] = np.cos(
        2 * np.pi * df["Hour"] / 24
    )

    df["Day_sin"] = np.sin(
        2 * np.pi * df["Day"] / 31
    )

    df["Day_cos"] = np.cos(
        2 * np.pi * df["Day"] / 31
    )

    df["Month_sin"] = np.sin(
        2 * np.pi * df["Month"] / 12
    )

    df["Month_cos"] = np.cos(
        2 * np.pi * df["Month"] / 12
    )

    # Time of day
    df["IsMorning"] = (
        (df["Hour"] >= 6) &
        (df["Hour"] < 12)
    ).astype(int)

    df["IsAfternoon"] = (
        (df["Hour"] >= 12) &
        (df["Hour"] < 18)
    ).astype(int)

    df["IsEvening"] = (
        (df["Hour"] >= 18) &
        (df["Hour"] < 24)
    ).astype(int)

    # Remove target and timestamp
    historical_features = df.drop(
        [
            "Traffic_Vol",
            "TimeStamp"
        ],
        axis=1,
        errors="ignore"
    )

    # Maintain exact training feature order
    historical_features = historical_features[
        feature_columns
    ]

    # Take last n_past records
    historical_features = historical_features.tail(
        n_past
    )

    return historical_features


# Prepare historical data once
historical_features = prepare_historical_data()

print(
    f"Historical data prepared successfully "
    f"with {n_past} records."
)


# ==========================================================
# TRAFFIC PREDICTION FUNCTION
# ==========================================================

def predict_traffic(
    temperature,
    rainfall,
    snowfall,
    cloud_cover,
    timestamp_input,
    weather,
    weather_desc,
    holiday
):

    timestamp = pd.to_datetime(
        timestamp_input,
        format="%d/%m/%y %H:%M"
    )

    hour = timestamp.hour
    day = timestamp.day
    month = timestamp.month
    year = timestamp.year
    weekday = timestamp.weekday()

    is_weekend = int(weekday >= 5)

    is_holiday = int(
        str(holiday).lower() in [
            "yes", "y", "true", "1"
        ]
    )

    # Cyclical features
    hour_sin = np.sin(
        2 * np.pi * hour / 24
    )

    hour_cos = np.cos(
        2 * np.pi * hour / 24
    )

    day_sin = np.sin(
        2 * np.pi * day / 31
    )

    day_cos = np.cos(
        2 * np.pi * day / 31
    )

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    # Time of day
    is_morning = int(6 <= hour < 12)
    is_afternoon = int(12 <= hour < 18)
    is_evening = int(18 <= hour < 24)

    # Encode weather
    weather_encoded = safe_transform(
        label_encoder_weather,
        weather
    )

    weather_desc_encoded = safe_transform(
        label_encoder_weather_desc,
        weather_desc
    )

    # Create input data
    user_data = pd.DataFrame(
        [{
            "Temperature": float(temperature),
            "Rainfall_last_hour": float(rainfall),
            "Snowfall_last_hour": float(snowfall),
            "Cloud_Cover": float(cloud_cover),

            "Hour": hour,
            "Day": day,
            "Month": month,
            "Year": year,
            "Weekday": weekday,

            "IsWeekend": is_weekend,
            "IsHoliday": is_holiday,

            "Weather_encoded": weather_encoded,
            "Weather_Desc_encoded": weather_desc_encoded,

            "Hour_sin": hour_sin,
            "Hour_cos": hour_cos,

            "Day_sin": day_sin,
            "Day_cos": day_cos,

            "Month_sin": month_sin,
            "Month_cos": month_cos,

            "IsMorning": is_morning,
            "IsAfternoon": is_afternoon,
            "IsEvening": is_evening
        }]
    )

    # Exact feature order
    user_data = user_data[feature_columns]

    # Combine historical and current input
    combined_data = pd.concat(
        [
            historical_features,
            user_data
        ],
        ignore_index=True
    )

    combined_data = combined_data.tail(n_past)

    # Scaling
    scaled_input = scaler_features.transform(
        combined_data
    )

    X_input = scaled_input.reshape(
        1,
        n_past,
        len(feature_columns)
    )

    # Prediction
    prediction_scaled = model.predict(
        X_input,
        verbose=0
    )

    prediction_transformed = (
        scaler_target.inverse_transform(
            prediction_scaled
        )
    )

    prediction = np.power(
        prediction_transformed,
        1 / 1.1405
    )

    predicted_volume = float(prediction[0][0])

    predicted_volume = max(
        0,
        predicted_volume
    )

    return {
        "predicted_traffic_volume": round(
            predicted_volume, 2
        ),

        "timestamp": timestamp.strftime(
            "%d/%m/%Y %H:%M"
        ),

        "hour": hour,
        "weather": weather,
        "weather_description": weather_desc,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday
    }


# ==========================================================
# STANDALONE TEST
# Runs only when this file is executed directly
# ==========================================================

if __name__ == "__main__":

    get_weather_categories()

    # Replace these with exact categories
    # printed by get_weather_categories().

    result = predict_traffic(
        temperature=23,
        rainfall=0.0,
        snowfall=0.0,
        cloud_cover=50,
        timestamp_input="06/09/26 18:00",
        weather="Cloudy skies",
        weather_desc="Full cloud cover",
        holiday="yes"
    )

    print("\n" + "=" * 50)
    print("TRAFFIC PREDICTION RESULT")
    print("=" * 50)

    print(
        "Predicted Traffic Volume:",
        result["predicted_traffic_volume"]
    )

    print("Timestamp:", result["timestamp"])
    print("Weather:", result["weather"])
    print("Weekend:", result["is_weekend"])
    print("Holiday:", result["is_holiday"])

    print("=" * 50)
