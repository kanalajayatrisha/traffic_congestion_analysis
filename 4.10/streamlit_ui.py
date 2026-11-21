
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Traffic Intelligence System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

from main import (
    run_complete_traffic_analysis,
    display_available_bottleneck_features
)


# ============================================================
# 2. CSS STYLING
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #f6f8fc;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
    opacity: 0.8;
}

.footer {
    text-align: center;
    margin-top: 50px;
    padding: 20px;
    opacity: 0.6;
}

.result-card {
    padding: 25px 20px;
    border-radius: 15px;
    min-height: 170px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.06);
    margin-bottom: 15px;
}

.result-card h3 {
    font-size: 20px;
    margin-bottom: 12px;
}

.result-card p {
    font-size: 15px;
    line-height: 1.6;
}

.traffic-card {
    background-color: #EAF4FF;
    border-left: 5px solid #4A90E2;
}

.bottleneck-card {
    background-color: #FFF4E5;
    border-left: 5px solid #F5A623;
}

.duration-card {
    background-color: #F3EFFF;
    border-left: 5px solid #8E6CEB;
}

.forecast-card {
    background-color: #EAFBF1;
    border-left: 5px solid #34A853;
}

.metric-card {
    padding: 18px 20px;
    border-radius: 14px;
    min-height: 125px;
    border: 1px solid #E5E7EB;
    box-shadow: 0px 3px 10px rgba(0,0,0,0.05);
    margin-bottom: 15px;
}

.metric-title {
    font-size: 15px;
    font-weight: 600;
    color: #4B5563;
    margin-bottom: 10px;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: #1F2937;
}

.traffic-metric {
    background-color: #EFF8FF;
    border-left: 5px solid #60A5FA;
}

.congestion-metric {
    background-color: #FFF7ED;
    border-left: 5px solid #FB923C;
}

.duration-metric {
    background-color: #F5F3FF;
    border-left: 5px solid #A78BFA;
}

.forecast-metric {
    background-color: #F0FDF4;
    border-left: 5px solid #4ADE80;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. HELPER FUNCTIONS
# ============================================================

def get_result(results, key, section_name):

    if key not in results:
        raise KeyError(
            f"{section_name} not found. "
            f"Available keys: {list(results.keys())}"
        )

    return results[key]


def get_value(data, possible_keys, default=None):

    if not isinstance(data, dict):
        return default

    for key in possible_keys:
        if key in data:
            return data[key]

    return default


def format_number(value, decimals=2):

    try:
        return f"{float(value):,.{decimals}f}"

    except (ValueError, TypeError):
        return str(value)


def normalize_dataframe(data):

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if data is None:
        return pd.DataFrame()

    return pd.DataFrame(data)


def normalize_forecast_dataframe(data):

    df = normalize_dataframe(data)

    if df.empty:
        raise ValueError("Forecast dataframe is empty.")

    # Support common timestamp column variations.
    timestamp_column = next(
        (
            col for col in ["TimeStamp", "Timestamp", "timestamp", "DateTime"]
            if col in df.columns
        ),
        None
    )

    if timestamp_column is None:
        raise KeyError(
            f"Forecast dataframe has no timestamp column. "
            f"Available columns: {list(df.columns)}"
        )

    if timestamp_column != "TimeStamp":
        df.rename(
            columns={timestamp_column: "TimeStamp"},
            inplace=True
        )

    # Support common prediction column variations.
    traffic_column = next(
        (
            col for col in [
                "Predicted_Traffic_Volume",
                "predicted_traffic_volume",
                "Traffic_Volume",
                "traffic_volume"
            ]
            if col in df.columns
        ),
        None
    )

    if traffic_column is None:
        raise KeyError(
            "Forecast dataframe has no predicted traffic column. "
            f"Available columns: {list(df.columns)}"
        )

    if traffic_column != "Predicted_Traffic_Volume":
        df.rename(
            columns={
                traffic_column: "Predicted_Traffic_Volume"
            },
            inplace=True
        )

    df["TimeStamp"] = pd.to_datetime(
        df["TimeStamp"],
        dayfirst=True,
        errors="coerce"
    )

    df["Predicted_Traffic_Volume"] = pd.to_numeric(
        df["Predicted_Traffic_Volume"],
        errors="coerce"
    )

    df.dropna(
        subset=["TimeStamp", "Predicted_Traffic_Volume"],
        inplace=True
    )

    if "Traffic_Level" not in df.columns:
        df["Traffic_Level"] = "Unknown"

    df.reset_index(drop=True, inplace=True)

    if df.empty:
        raise ValueError(
            "No valid forecast rows remain after processing."
        )

    return df


def normalize_thresholds(data):

    thresholds = data.get("thresholds", {})

    if not isinstance(thresholds, dict):
        thresholds = {}

    low = data.get(
        "low_threshold",
        thresholds.get("low_threshold")
    )

    moderate = data.get(
        "moderate_threshold",
        thresholds.get("moderate_threshold")
    )

    return low, moderate


# ============================================================
# 4. SIDEBAR INPUTS
# ============================================================

with st.sidebar:

    st.title("🚦 Traffic AI Intelligence")

    st.write(
        "Configure traffic and environmental conditions "
        "to generate predictions."
    )

    st.divider()

    # Environmental conditions
    st.subheader("🌡️ Environmental Conditions")

    temperature = st.number_input(
        "Temperature (°C)",
        value=23.0
    )

    rainfall = st.number_input(
        "Rainfall Last Hour (mm)",
        min_value=0.0,
        value=0.0
    )

    snowfall = st.number_input(
        "Snowfall Last Hour (mm)",
        min_value=0.0,
        value=0.0
    )

    cloud_cover = st.number_input(
        "Cloud Cover (%)",
        min_value=0.0,
        max_value=100.0,
        value=50.0
    )

    st.divider()

    # Weather conditions
    st.subheader("🌦️ Weather Conditions")

    weather = st.selectbox(
        "Weather Condition",
        [
            "Cloudy skies",
            "Clear skies",
            "Rainfall",
            "Light rain",
            "Light fog",
            "Dense fog",
            "Stormy weather",
            "Airborne particles",
            "Snowfall",
            "Sudden windstorm",
            "Airborne smoke"
        ]
    )

    weather_desc = st.selectbox(
        "Weather Description",
        [
            "Partly cloudy skies",
            "Fragmented clouds",
            "Full cloud cover",
            "Clear skies",
            "Scattered clouds",
            "Soft rain",
            "Mild drizzle",
            "Fine fog",
            "Dense fog",
            "Nearby rain showers",
            "Steady rain",
            "Light rain",
            "Intense rainfall",
            "Approaching thunderstorm",
            "Smoky air",
            "Deep snow",
            "Storm with mild rain",
            "Snowfall",
            "Drizzle mixed with snow",
            "Gentle rain shower",
            "Sudden windstorm",
            "Approaching thunderstorm with rain",
            "Storm with rain",
            "Severe storm with heavy rain",
            "Strong drizzle",
            "Torrential downpour",
            "Storm with a light sprinkle",
            "Stormy weather",
            "Flurries",
            "Airborne smoke",
            "Ice rain",
            "Scattered snowflakes",
            "Icy rain",
            "Nearby thunderstorm with light rain"
        ]
    )

    holiday = st.selectbox(
        "Is Holiday?",
        ["no", "yes"]
    )

    st.divider()

    # Prediction time
    st.subheader("📅 Prediction Time")

    selected_date = st.date_input(
        "Select Date"
    )

    selected_time = st.time_input(
        "Select Time"
    )

    timestamp_input = (
        selected_date.strftime("%d/%m/%y")
        + " "
        + selected_time.strftime("%H:%M")
    )

    st.divider()

    # Bottleneck feature
    st.subheader("🚧 Bottleneck Analysis")

    try:
        bottleneck_features = (
            display_available_bottleneck_features()
        )

    except Exception as e:
        st.warning(
            f"Could not load bottleneck features: {e}"
        )
        bottleneck_features = []

    if bottleneck_features:

        bottleneck_feature = st.selectbox(
            "Select Analysis Feature",
            bottleneck_features
        )

    else:

        bottleneck_feature = st.text_input(
            "Bottleneck Feature",
            value="Hour"
        )

    st.divider()

    # Forecast days
    st.subheader("📈 Future Forecast")

    future_days = st.slider(
        "Forecast Days",
        min_value=1,
        max_value=7,
        value=1
    )

    run_prediction = st.button(
        "🚀 Run Analysis",
        use_container_width=True,
        type="primary"
    )


# ============================================================
# 5. MAIN TITLE
# ============================================================

st.markdown("""
<div class="main-title">
    🚦 AI Traffic Intelligence Dashboard
</div>

<div class="subtitle">
    AI-Powered Traffic Volume Prediction, Bottleneck Analysis,
    Congestion Duration Estimation and Future Forecasting
</div>
""", unsafe_allow_html=True)


# ============================================================
# 6. INITIAL DASHBOARD
# ============================================================

if not run_prediction:

    st.info(
        "👈 Configure the traffic and environmental conditions "
        "in the sidebar and click Run Analysis."
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="result-card traffic-card">
            <h3>🚗 Traffic Volume</h3>
            <p>Predict traffic volume using temporal and environmental conditions.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="result-card bottleneck-card">
            <h3>🚧 Bottlenecks</h3>
            <p>Identify recurring high-traffic patterns using DBSCAN analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="result-card duration-card">
            <h3>⏱️ Duration</h3>
            <p>Estimate the expected duration of the predicted congestion category.</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="result-card forecast-card">
            <h3>📈 Forecast</h3>
            <p>Forecast hourly traffic conditions for upcoming days.</p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================
# 7. RUN TRAFFIC ANALYSIS
# ============================================================

else:

    try:

        with st.spinner(
            "🤖 AI models are analyzing traffic conditions..."
        ):

            results = run_complete_traffic_analysis(

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

        # ----------------------------------------------------
        # Validate main result
        # ----------------------------------------------------

        if not isinstance(results, dict):
            raise TypeError(
                "main.py must return a dictionary."
            )

        # ----------------------------------------------------
        # Extract exact keys from main.py
        # ----------------------------------------------------

        current_traffic = get_result(
            results,
            "current_traffic",
            "Current traffic prediction"
        )

        bottleneck = get_result(
            results,
            "bottleneck_analysis",
            "Bottleneck analysis"
        )

        congestion = get_result(
            results,
            "congestion_duration",
            "Congestion duration"
        )

        forecast_result = get_result(
            results,
            "future_traffic_forecast",
            "Future traffic forecast"
        )

        # Validate module results
        for name, value in [
            ("Current traffic", current_traffic),
            ("Bottleneck", bottleneck),
            ("Congestion", congestion),
            ("Future forecast", forecast_result)
        ]:

            if not isinstance(value, dict):
                raise TypeError(
                    f"{name} result must be a dictionary."
                )

        # ----------------------------------------------------
        # Extract forecast dataframe
        # ----------------------------------------------------

        forecast_data = get_value(
            forecast_result,
            ["forecast", "forecast_df", "predictions"]
        )

        if forecast_data is None:
            raise KeyError(
                "Forecast result does not contain a forecast dataframe."
            )

        forecast_df = normalize_forecast_dataframe(
            forecast_data
        )

        # ----------------------------------------------------
        # Extract current traffic
        # ----------------------------------------------------

        traffic_value = float(
            current_traffic["predicted_traffic_volume"]
        )

        # ----------------------------------------------------
        # Extract congestion information
        # ----------------------------------------------------

        congestion_level = get_value(
            congestion,
            ["congestion_level"],
            "Unknown"
        )

        duration_hours = get_value(
            congestion,
            ["duration_hours"],
            0
        )

        # ----------------------------------------------------
        # Extract bottleneck information
        # ----------------------------------------------------

        insight = bottleneck.get(
            "bottleneck_insight",
            {}
        )

        if not isinstance(insight, dict):
            insight = {}

        summary_data = bottleneck.get(
            "traffic_summary"
        )

        summary_df = normalize_dataframe(
            summary_data
        )

        cluster_data = bottleneck.get(
            "cluster_information"
        )

        cluster_df = normalize_dataframe(
            cluster_data
        )

        # ----------------------------------------------------
        # Extract thresholds
        # ----------------------------------------------------

        low_threshold, moderate_threshold = (
            normalize_thresholds(forecast_result)
        )

        # ====================================================
        # 8. TOP METRIC CARDS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.markdown(f"""
            <div class="metric-card traffic-metric">
                <div class="metric-title">
                    🚗 Current Traffic Volume
                </div>
                <div class="metric-value">
                    {traffic_value:,.0f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col2:

            st.markdown(f"""
            <div class="metric-card congestion-metric">
                <div class="metric-title">
                    🚦 Congestion Level
                </div>
                <div class="metric-value">
                    {str(congestion_level).upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col3:

            st.markdown(f"""
            <div class="metric-card duration-metric">
                <div class="metric-title">
                    ⏱️ Congestion Duration
                </div>
                <div class="metric-value">
                    {duration_hours} Hours
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col4:

            st.markdown(f"""
            <div class="metric-card forecast-metric">
                <div class="metric-title">
                    📈 Forecast Window
                </div>
                <div class="metric-value">
                    {future_days} Day(s)
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # ====================================================
        # 9. DASHBOARD TABS
        # ====================================================

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🚗 Current Traffic Analysis",
            "🚧 Bottleneck Analysis",
            "⏱️ Congestion Duration",
            "📈 Future Forecast",
            "🤖 Final Recommendations"
        ])

        # ====================================================
        # TAB 1: CURRENT TRAFFIC
        # ====================================================

        with tab1:

            st.subheader(
                "🚗 Current Traffic Volume Prediction"
            )

            c1, c2 = st.columns([1, 2])

            with c1:

                st.metric(
                    "Predicted Traffic Volume",
                    format_number(traffic_value)
                )

                with st.container(border=True):

                    st.write(
                        f"**Timestamp:** "
                        f"{current_traffic.get('timestamp', timestamp_input)}"
                    )

                    st.write(
                        f"**Weather:** {weather}"
                    )

                    st.write(
                        f"**Weather Description:** {weather_desc}"
                    )

                    st.write(
                        f"**Weekend:** "
                        f"{current_traffic.get('is_weekend', 'N/A')}"
                    )

                    st.write(
                        f"**Holiday:** "
                        f"{current_traffic.get('is_holiday', 'N/A')}"
                    )

            with c2:

                fig = go.Figure(
                    go.Indicator(
                        mode="gauge+number",
                        value=traffic_value,
                        title={
                            "text": "Current Predicted Traffic Volume"
                        },
                        gauge={
                            "axis": {
                                "range": [0, 10000]
                            },
                            "bar": {
                                "color": "#EC4899"
                            },
                            "steps": [
                                {
                                    "range": [0, 3000],
                                    "color": "#DCFCE7"
                                },
                                {
                                    "range": [3000, 7000],
                                    "color": "#FEF3C7"
                                },
                                {
                                    "range": [7000, 10000],
                                    "color": "#FEE2E2"
                                }
                            ]
                        }
                    )
                )

                fig.update_layout(height=400)

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        # ====================================================
        # TAB 2: BOTTLENECK ANALYSIS
        # ====================================================

        with tab2:

            st.subheader(
                "🚧 Historical Traffic Bottleneck Analysis"
            )

            if insight.get("bottleneck_detected", False):

                st.warning(
                    "🚧 Historical high-traffic pattern identified."
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Analysis Feature",
                        str(
                            insight.get(
                                "selected_feature",
                                bottleneck.get("selected_feature", "N/A")
                            )
                        )
                    )

                with col2:

                    st.metric(
                        "Average Traffic",
                        format_number(
                            insight.get(
                                "average_traffic_volume",
                                0
                            )
                        )
                    )

                with col3:

                    st.metric(
                        "Maximum Traffic",
                        format_number(
                            insight.get(
                                "maximum_traffic_volume",
                                0
                            )
                        )
                    )

                st.write(
                    f"**Pattern Value:** "
                    f"{insight.get('feature_value', 'N/A')}"
                )

                st.write(
                    f"**Insight:** "
                    f"{insight.get('message', 'Historical pattern identified.')}"
                )

            else:

                st.info(
                    insight.get(
                        "message",
                        "No significant historical bottleneck pattern detected."
                    )
                )

            # Historical traffic summary
            if not summary_df.empty:

                st.subheader(
                    "Average Traffic Across Historical Patterns"
                )

                feature_name = bottleneck.get(
                    "selected_feature",
                    insight.get("selected_feature")
                )

                if feature_name not in summary_df.columns:
                    feature_name = summary_df.columns[0]

                average_column = next(
                    (
                        col for col in [
                            "Average_Traffic_Volume",
                            "average_traffic_volume"
                        ]
                        if col in summary_df.columns
                    ),
                    None
                )

                if average_column is not None:

                    color_column = (
                        "Traffic_Level"
                        if "Traffic_Level" in summary_df.columns
                        else None
                    )

                    fig = px.bar(
                        summary_df,
                        x=feature_name,
                        y=average_column,
                        color=color_column,
                        hover_data=True,
                        title="Historical Average Traffic Volume"
                    )

                    fig.update_layout(
                        height=450,
                        xaxis_title=feature_name,
                        yaxis_title="Average Traffic Volume"
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                with st.expander(
                    "View Complete Traffic Pattern Data"
                ):

                    st.dataframe(
                        summary_df,
                        use_container_width=True,
                        hide_index=True
                    )

            else:

                st.info(
                    "Historical traffic summary is not available."
                )

            # DBSCAN cluster information
            if not cluster_df.empty:

                st.subheader(
                    "🔵 DBSCAN Cluster Information"
                )

                st.dataframe(
                    cluster_df,
                    use_container_width=True,
                    hide_index=True
                )

        # ====================================================
        # TAB 3: CONGESTION DURATION
        # ====================================================

        with tab3:

            st.subheader(
                "⏱️ Congestion Duration Prediction"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Predicted Traffic Volume",
                    format_number(
                        congestion.get(
                            "predicted_traffic_volume",
                            traffic_value
                        )
                    )
                )

            with col2:

                st.metric(
                    "Congestion Level",
                    str(congestion_level).upper()
                )

            with col3:

                st.metric(
                    "Expected Duration",
                    f"{duration_hours} Hours"
                )

            st.divider()

            duration_forecast = congestion.get(
                "forecast"
            )

            duration_df = normalize_dataframe(
                duration_forecast
            )

            if not duration_df.empty:

                timestamp_column = next(
                    (
                        col for col in [
                            "TimeStamp",
                            "Timestamp",
                            "timestamp"
                        ]
                        if col in duration_df.columns
                    ),
                    None
                )

                traffic_column = next(
                    (
                        col for col in [
                            "Predicted_Traffic_Volume",
                            "predicted_traffic_volume"
                        ]
                        if col in duration_df.columns
                    ),
                    None
                )

                if (
                    timestamp_column is not None
                    and traffic_column is not None
                ):

                    duration_df[timestamp_column] = pd.to_datetime(
                        duration_df[timestamp_column],
                        dayfirst=True,
                        errors="coerce"
                    )

                    fig = px.line(
                        duration_df,
                        x=timestamp_column,
                        y=traffic_column,
                        markers=True,
                        title="Predicted Congestion Timeline"
                    )

                    thresholds = congestion.get(
                        "thresholds",
                        {}
                    )

                    if isinstance(thresholds, dict):

                        low = thresholds.get(
                            "low_threshold"
                        )

                        moderate = thresholds.get(
                            "moderate_threshold"
                        )

                        if low is not None:

                            fig.add_hline(
                                y=low,
                                line_dash="dash",
                                annotation_text="Low Threshold"
                            )

                        if moderate is not None:

                            fig.add_hline(
                                y=moderate,
                                line_dash="dash",
                                annotation_text="Moderate Threshold"
                            )

                    fig.update_layout(height=450)

                    st.plotly_chart(
                        fig,
                        use_container_width=True
                    )

                else:

                    st.dataframe(
                        duration_df,
                        use_container_width=True
                    )

            else:

                st.info(
                    "Detailed congestion forecast is not available."
                )

            st.info(
                f"The predicted {str(congestion_level).upper()} "
                f"traffic category is estimated to continue for "
                f"approximately {duration_hours} hour(s)."
            )

            st.write(
                f"**Selected Timestamp:** "
                f"{congestion.get('selected_timestamp', timestamp_input)}"
            )

            st.write(
                f"**Expected Category Change:** "
                f"{congestion.get('congestion_end_time', 'N/A')}"
            )

            st.write(
                f"**Next Predicted Category:** "
                f"{congestion.get('next_congestion_level', 'N/A')}"
            )

        # ====================================================
        # TAB 4: FUTURE TRAFFIC FORECAST
        # ====================================================

        with tab4:

            st.subheader(
                f"📈 Hourly Traffic Forecast for {future_days} Day(s)"
            )

            st.metric(
                "Total Hourly Predictions",
                len(forecast_df)
            )

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=forecast_df["TimeStamp"],
                    y=forecast_df["Predicted_Traffic_Volume"],
                    mode="lines",
                    name="Predicted Traffic",
                    line=dict(
                        width=3,
                        color="#8B5CF6"
                    ),
                    hovertemplate=(
                        "<b>Date:</b> %{x|%d %b %Y %H:%M}<br>"
                        "<b>Traffic:</b> %{y:.0f}"
                        "<extra></extra>"
                    )
                )
            )

            if low_threshold is not None:

                fig.add_hline(
                    y=low_threshold,
                    line_dash="dash",
                    annotation_text="Low Threshold"
                )

            if moderate_threshold is not None:

                fig.add_hline(
                    y=moderate_threshold,
                    line_dash="dash",
                    annotation_text="Moderate Threshold"
                )

            fig.update_layout(
                xaxis_title="Date and Time",
                yaxis_title="Predicted Traffic Volume",
                height=520,
                hovermode="x unified"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            col1, col2 = st.columns(2)

            with col1:

                st.subheader("Forecast Data")

                st.dataframe(
                    forecast_df,
                    use_container_width=True,
                    hide_index=True,
                    height=350
                )

                csv = forecast_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Traffic Forecast CSV",
                    data=csv,
                    file_name="traffic_forecast.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "Traffic Category Distribution"
                )

                level_count = (
                    forecast_df["Traffic_Level"]
                    .value_counts()
                    .reset_index()
                )

                level_count.columns = [
                    "Traffic_Level",
                    "Hours"
                ]

                fig2 = px.pie(
                    level_count,
                    names="Traffic_Level",
                    values="Hours",
                    hole=0.55,
                    color="Traffic_Level",
                    color_discrete_map={
                        "Low": "#86EFAC",
                        "Moderate": "#FDE68A",
                        "High": "#FDA4AF",
                        "Low Traffic": "#86EFAC",
                        "Moderate Traffic": "#FDE68A",
                        "High Traffic": "#FDA4AF"
                    }
                )

                fig2.update_layout(height=400)

                st.plotly_chart(
                    fig2,
                    use_container_width=True
                )

        # ====================================================
        # TAB 5: FINAL RECOMMENDATIONS
        # ====================================================

        with tab5:

            st.subheader(
                "🤖 Final Traffic Intelligence Summary"
            )

            col1, col2 = st.columns(2)

            with col1:

                with st.container(border=True):

                    st.markdown(
                        "### 🚗 Current Traffic"
                    )

                    st.metric(
                        "Traffic Volume",
                        format_number(traffic_value)
                    )

                    st.metric(
                        "Congestion Category",
                        str(congestion_level).upper()
                    )

                    st.write(
                        f"Expected Duration: {duration_hours} hour(s)"
                    )

                    st.write(
                        f"Expected Until: "
                        f"{congestion.get('congestion_end_time', 'N/A')}"
                    )

                    st.write(
                        f"Next Category: "
                        f"{congestion.get('next_congestion_level', 'N/A')}"
                    )

            with col2:

                with st.container(border=True):

                    st.markdown(
                        "### 🚧 Bottleneck Pattern"
                    )

                    if insight.get("bottleneck_detected", False):

                        st.write(
                            f"Feature: "
                            f"{insight.get('selected_feature', 'N/A')}"
                        )

                        st.write(
                            f"Pattern Value: "
                            f"{insight.get('feature_value', 'N/A')}"
                        )

                        st.write(
                            f"Average Traffic: "
                            f"{format_number(insight.get('average_traffic_volume'))}"
                        )

                        st.write(
                            f"Maximum Traffic: "
                            f"{format_number(insight.get('maximum_traffic_volume'))}"
                        )

                    else:

                        st.success(
                            "No significant historical bottleneck pattern detected."
                        )

            st.divider()

            st.subheader(
                "📈 Upcoming Traffic Conditions"
            )

            forecast_display = forecast_df.head(10).copy()

            def color_traffic_row(row):

                level = str(
                    row["Traffic_Level"]
                ).strip().lower()

                if level in ["low", "low traffic"]:
                    color = (
                        "background-color: #DCFCE7; "
                        "color: #166534;"
                    )

                elif level in ["moderate", "moderate traffic"]:
                    color = (
                        "background-color: #FFEDD5; "
                        "color: #9A3412;"
                    )

                elif level in ["high", "high traffic"]:
                    color = (
                        "background-color: #FEE2E2; "
                        "color: #991B1B;"
                    )

                else:
                    color = ""

                return [color] * len(row)

            styled_forecast = forecast_display.style.apply(
                color_traffic_row,
                axis=1
            )

            st.dataframe(
                styled_forecast,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader("📋 Model Output Information")

            st.write(
                "**Prediction Timestamp:**",
                results.get("timestamp", timestamp_input)
            )

            st.write(
                "**Forecast Records:**",
                len(forecast_df)
            )

            st.write(
                "**Forecast Days:**",
                future_days
            )

            st.caption(
                "Note: Bottleneck results represent historical traffic "
                "patterns. Congestion duration and future traffic values "
                "are model-based estimates, not guaranteed real-world conditions."
            )

    except Exception as e:

        st.error(
            "An error occurred while running the traffic analysis."
        )

        st.exception(e)

        st.warning(
            "Please check the traceback and compare the actual output "
            "from main.py with the dashboard fields."
        )


# ============================================================
# 10. FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    🚦 AI-Powered Traffic Intelligence System<br>
    Traffic Prediction • Bottleneck Analysis •
    Congestion Duration • Future Forecasting
</div>
""", unsafe_allow_html=True)
