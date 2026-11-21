
import pandas as pd
import joblib


# ============================================================
# 1. LOAD SAVED BOTTLENECK MODELS AND DATA
# ============================================================

MODEL_DIR = "saved_bottleneck_models"

print("Loading Bottleneck Detection Models...")

dbscan_models = joblib.load(
    f"{MODEL_DIR}/dbscan_models.pkl"
)

feature_encodings = joblib.load(
    f"{MODEL_DIR}/feature_encodings.pkl"
)

cluster_information = joblib.load(
    f"{MODEL_DIR}/cluster_information.pkl"
)

trend_summaries = joblib.load(
    f"{MODEL_DIR}/trend_summaries.pkl"
)

traffic_thresholds = joblib.load(
    f"{MODEL_DIR}/traffic_thresholds.pkl"
)

analysis_features = joblib.load(
    f"{MODEL_DIR}/analysis_features.pkl"
)

q33 = traffic_thresholds["q33"]

q66 = traffic_thresholds["q66"]

print("Bottleneck Detection Models loaded successfully!")


# ============================================================
# 2. GET AVAILABLE FEATURES
# ============================================================

def get_available_features():
    """
    Returns the available features for bottleneck analysis.
    """

    return list(analysis_features)


# ============================================================
# 3. ANALYZE BOTTLENECK
# ============================================================

def analyze_bottleneck(
    selected_feature,
    timestamp_input=None,
    current_features=None
):
    """
    Analyze historical traffic patterns and identify
    bottleneck conditions.

    Parameters
    ----------
    selected_feature : str
        Feature selected for analysis.

    timestamp_input : str, optional
        Current timestamp in DD/MM/YY HH:MM format.

    current_features : dict, optional
        Additional current input features.

    Returns
    -------
    dict
        Historical traffic summary, traffic categories,
        cluster information and bottleneck insight.
    """

    # --------------------------------------------------------
    # STEP 1: VALIDATE FEATURE
    # --------------------------------------------------------

    if selected_feature not in analysis_features:

        raise ValueError(
            f"Invalid feature: {selected_feature}. "
            f"Available features: {analysis_features}"
        )

    # --------------------------------------------------------
    # STEP 2: GET HISTORICAL TREND SUMMARY
    # --------------------------------------------------------

    summary = trend_summaries[
        selected_feature
    ].copy()

    if summary.empty:

        raise ValueError(
            f"No historical trend data found "
            f"for {selected_feature}."
        )

    summary = summary.sort_values(
        "Average_Traffic_Volume",
        ascending=False
    )

    # --------------------------------------------------------
    # STEP 3: SEPARATE TRAFFIC CATEGORIES
    # --------------------------------------------------------

    high_traffic = summary[
        summary["Traffic_Level"] == "High Traffic"
    ].copy()

    moderate_traffic = summary[
        summary["Traffic_Level"] == "Moderate Traffic"
    ].copy()

    low_traffic = summary[
        summary["Traffic_Level"] == "Low Traffic"
    ].copy()

    # --------------------------------------------------------
    # STEP 4: GET CLUSTER INFORMATION
    # --------------------------------------------------------

    cluster_info = cluster_information[
        selected_feature
    ].copy()

    # --------------------------------------------------------
    # STEP 5: HISTORICAL BOTTLENECK ANALYSIS
    # --------------------------------------------------------

    if len(high_traffic) > 0:

        top_bottleneck = high_traffic.iloc[0]

        bottleneck_insight = {

            "bottleneck_detected": True,

            "feature": selected_feature,

            "feature_value":
                top_bottleneck[selected_feature],

            "average_traffic_volume":
                round(
                    float(
                        top_bottleneck[
                            "Average_Traffic_Volume"
                        ]
                    ),
                    2
                ),

            "maximum_traffic_volume":
                round(
                    float(
                        top_bottleneck[
                            "Maximum_Traffic_Volume"
                        ]
                    ),
                    2
                ),

            "traffic_category":
                "High Traffic / Bottleneck",

            "message":
                "High traffic pattern detected."

        }

    else:

        bottleneck_insight = {

            "bottleneck_detected": False,

            "feature": selected_feature,

            "feature_value": None,

            "average_traffic_volume": 0,

            "maximum_traffic_volume": 0,

            "traffic_category": "No High Traffic Pattern",

            "message":
                "No significant historical bottleneck pattern detected."

        }

    # --------------------------------------------------------
    # STEP 6: CHECK CURRENT TIMESTAMP
    # --------------------------------------------------------

    current_pattern = None

    if timestamp_input is not None:

        selected_timestamp = pd.to_datetime(
            timestamp_input,
            format="%d/%m/%y %H:%M",
            errors="coerce"
        )

        if pd.isna(selected_timestamp):

            raise ValueError(
                "Invalid timestamp format. "
                "Use DD/MM/YY HH:MM."
            )

        # Generate time-based feature values

        time_features = {

            "Hour": selected_timestamp.hour,

            "Day": selected_timestamp.day,

            "Month": selected_timestamp.month,

            "Year": selected_timestamp.year,

            "Weekday": selected_timestamp.weekday(),

            "IsWeekend": int(
                selected_timestamp.weekday() >= 5
            )

        }

        # Add other current features if supplied

        if current_features is not None:

            time_features.update(
                current_features
            )

        if selected_feature == "IsHoliday":

            value = time_features.get(
                "IsHoliday"
            )

            if isinstance(value, str):

                time_features["IsHoliday"] = (
                    1 if value.lower() in ["yes", "true", "1"]
                    else 0
                )

        # Get the current selected feature value

        if selected_feature in time_features:

            current_value = time_features[
                selected_feature
            ]

            # Find matching historical feature value

            matching_rows = summary[
                summary[selected_feature] == current_value
            ].copy()

            if len(matching_rows) > 0:

                matched_row = matching_rows.iloc[0]

                current_level = matched_row[
                    "Traffic_Level"
                ]

                is_bottleneck = (
                    current_level == "High Traffic"
                )

                current_pattern = {

                    "feature": selected_feature,

                    "feature_value": current_value,

                    "traffic_level": current_level,

                    "average_traffic_volume":
                        round(
                            float(
                                matched_row[
                                    "Average_Traffic_Volume"
                                ]
                            ),
                            2
                        ),

                    "maximum_traffic_volume":
                        round(
                            float(
                                matched_row[
                                    "Maximum_Traffic_Volume"
                                ]
                            ),
                            2
                        ),

                    "bottleneck_detected":
                        is_bottleneck,

                    "message":
                        (
                            "High traffic / bottleneck pattern found "
                            "for the current feature value."
                            if is_bottleneck
                            else
                            "No high traffic pattern found "
                            "for the current feature value."
                        )

                }

            else:

                current_pattern = {

                    "feature": selected_feature,

                    "feature_value": current_value,

                    "traffic_level": "Unknown",

                    "average_traffic_volume": None,

                    "maximum_traffic_volume": None,

                    "bottleneck_detected": False,

                    "message":
                        "No matching historical pattern found "
                        "for the current feature value."

                }

    # --------------------------------------------------------
    # STEP 7: RETURN COMPLETE RESULTS
    # --------------------------------------------------------

    result = {

        "selected_feature":
            selected_feature,

        "summary":
            summary,

        "high_traffic_patterns":
            high_traffic,

        "moderate_traffic_patterns":
            moderate_traffic,

        "low_traffic_patterns":
            low_traffic,

        "cluster_information":
            cluster_info,

        "bottleneck_insight":
            bottleneck_insight,

        "current_pattern":
            current_pattern,

        "traffic_thresholds": {

            "q33": float(q33),

            "q66": float(q66)

        }

    }

    return result


# ============================================================
# 4. ANALYZE BOTTLENECK USING FEATURE NUMBER
# ============================================================

def analyze_bottleneck_by_number(
    feature_number,
    timestamp_input=None,
    current_features=None
):
    """
    Analyze bottleneck using a 1-based feature number.
    """

    features = get_available_features()

    if feature_number < 1 or feature_number > len(features):

        raise ValueError(
            f"Feature number must be between 1 and {len(features)}."
        )

    selected_feature = features[
        feature_number - 1
    ]

    return analyze_bottleneck(
        selected_feature=selected_feature,
        timestamp_input=timestamp_input,
        current_features=current_features
    )


# ============================================================
# 5. TEST BOTTLENECK ANALYSIS
# ============================================================

if __name__ == "__main__":

    features = get_available_features()

    print("\n")
    print("=" * 70)

    print("AVAILABLE BOTTLENECK ANALYSIS FEATURES")

    print("=" * 70)

    for i, feature in enumerate(features, start=1):

        print(f"{i}. {feature}")

    # Select a feature for testing

    selected_feature = features[0]

    print("\nSelected Feature:", selected_feature)

    result = analyze_bottleneck(
        selected_feature=selected_feature
    )

    print("\n")
    print("=" * 70)

    print("BOTTLENECK ANALYSIS RESULT")

    print("=" * 70)

    print("\nHistorical Traffic Summary:")

    print(
        result["summary"].to_string(
            index=False
        )
    )

    print("\nHigh Traffic Patterns:")

    print(
        result["high_traffic_patterns"].to_string(
            index=False
        )
    )

    print("\nModerate Traffic Patterns:")

    print(
        result["moderate_traffic_patterns"].to_string(
            index=False
        )
    )

    print("\nLow Traffic Patterns:")

    print(
        result["low_traffic_patterns"].to_string(
            index=False
        )
    )

    print("\nBottleneck Insight:")

    print(
        result["bottleneck_insight"]
    )

    print("=" * 70)
