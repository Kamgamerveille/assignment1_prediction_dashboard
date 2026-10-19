import matplotlib.pyplot as plt
import seaborn as sns

from src.predictor import Predictor

from pathlib import Path

import pandas as pd
import streamlit as st


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Collision Prediction Dashboard",
    page_icon="🚗",
    layout="wide"
)


# --------------------------------------------------
# Load processed dataset
# --------------------------------------------------

@st.cache_data
def load_data():
    project_root = Path(__file__).resolve().parent.parent

    data_path = (
        project_root
        / "data"
        / "processed"
        / "processed_collisions.csv"
    )

    return pd.read_csv(data_path)

@st.cache_data
def load_roc_data():
    project_root = Path(__file__).resolve().parent.parent

    roc_path = (
        project_root
        / "models"
        / "test_roc_curve.csv"
    )

    return pd.read_csv(roc_path)

df = load_data()

roc_df = load_roc_data()

@st.cache_resource
def load_predictor():
    return Predictor()


predictor = load_predictor()

# Convert collision date to datetime
df["accdate"] = pd.to_datetime(df["accdate"])


# --------------------------------------------------
# Sidebar navigation
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select a page",
    [
        "Data Overview",
        "Interactive Exploration",
        "Prediction",
        "Model Performance"
    ]
)


# --------------------------------------------------
# Data Overview page
# --------------------------------------------------

if page == "Data Overview":
    st.title("Motor Vehicle Collision Prediction Dashboard")

    st.write(
        "This dashboard explores City of Toronto motor vehicle "
        "collisions involving killed or seriously injured persons."
    )

    st.header("Data Overview")

    # Summary metrics
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Number of Collisions",
        f"{len(df):,}"
    )

    col2.metric(
        "Number of Features",
        df.shape[1]
    )

    fatal_rate = df["target"].mean() * 100

    col3.metric(
        "Fatal Injury Rate",
        f"{fatal_rate:.2f}%"
    )

    # Dataset preview
    st.subheader("Dataset Preview")

    preview_rows = st.slider(
        "Number of rows to display",
        min_value=5,
        max_value=50,
        value=10,
        step=5
    )

    st.dataframe(
        df.head(preview_rows),
        use_container_width=True
    )

    # Summary statistics
    st.subheader("Numeric Summary Statistics")

    numeric_summary = df.describe(
        include="number"
    ).transpose()

    st.dataframe(
        numeric_summary,
        use_container_width=True
    )


# --------------------------------------------------
# Interactive Exploration page
# --------------------------------------------------

elif page == "Interactive Exploration":
    st.title("Interactive Exploration")

    st.write(
        "Use the controls below to explore patterns in the collision dataset."
    )

    # --------------------------------------------------
    # Date range filter
    # --------------------------------------------------

    min_date = df["accdate"].min().date()
    max_date = df["accdate"].max().date()

    col1, col2 = st.columns(2)

    with col1:
        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
            key="start_date"
        )

    with col2:
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
            key="end_date"
        )

    if start_date > end_date:
        st.error("Start date must be before or equal to end date.")
        st.stop()

    filtered_df = df[
        (df["accdate"].dt.date >= start_date)
        & (df["accdate"].dt.date <= end_date)
    ].copy()

    st.metric(
        "Collisions in Selected Date Range",
        f"{len(filtered_df):,}"
    )

    st.subheader("Collision Trend by Year")

    filtered_df["year"] = filtered_df["accdate"].dt.year

    yearly_collisions = (
        filtered_df.groupby("year")
        .size()
        .reset_index(name="collisions")
    )

    st.line_chart(
        yearly_collisions,
        x="year",
        y="collisions"
    )

    st.caption(
        "This chart shows the number of collisions recorded each year "
        "within the selected date range."
    )
        
    st.subheader("Fatal Injury Rate by Selected Feature")

    feature_options = {
        "Impact Type": "impactype",
        "Lighting Condition": "light",
        "Road Surface Condition": "rdsfcond",
        "Road Class": "road_class"
    }

    selected_feature_label = st.selectbox(
        "Select a feature to compare",
        options=list(feature_options.keys())
    )

    selected_feature = feature_options[selected_feature_label]

    feature_fatality_rate = (
        filtered_df.groupby(selected_feature)["target"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index(name="fatality_rate")
    )

    st.bar_chart(
        feature_fatality_rate,
        x=selected_feature,
        y="fatality_rate"
    )

    st.caption(
        f"This chart shows the fatal injury rate by "
        f"{selected_feature_label.lower()} within the selected date range."
    )
    
    st.subheader("Collision Outcomes by Selected Involvement Factor")

    factor_options = {
        "Pedestrian Involved": "pedestrian",
        "Cyclist Involved": "cyclist",
        "Motorcyclist Involved": "motorcyclist",
        "Heavy Truck Involved": "heavy_truck",
        "Older Adult Involved": "older_adult",
        "Aggressive Driving": "aggressive",
        "Distracted": "distracted"
    }

    selected_factor_label = st.selectbox(
        "Select an involvement factor",
        options=list(factor_options.keys())
    )

    selected_factor = factor_options[selected_factor_label]

    factor_df = filtered_df[
        filtered_df[selected_factor] == True
    ]

    factor_outcomes = (
        factor_df["acclass"]
        .value_counts()
        .reset_index()
    )

    factor_outcomes.columns = [
        "injury_class",
        "collision_count"
    ]

    st.bar_chart(
        factor_outcomes,
        x="injury_class",
        y="collision_count"
    )

    st.caption(
        f"This chart compares Fatal and Non-Fatal Injury collisions "
        f"where {selected_factor_label.lower()} is present."
    )

# --------------------------------------------------
# Prediction page
# --------------------------------------------------

elif page == "Prediction":
    st.title("Collision Injury Prediction")

    st.write(
        "Enter collision characteristics below to predict whether "
        "the collision is more likely to result in a Fatal or "
        "Non-Fatal Injury."
    )

    with st.form("prediction_form"):

        st.subheader("Collision Details")

        col1, col2 = st.columns(2)

        with col1:
            per_inv = st.number_input(
                "Number of People Involved",
                min_value=1,
                max_value=20,
                value=2
            )

            accloc = st.selectbox(
                "Collision Location Type",
                sorted(df["accloc"].unique())
            )

            traffictl = st.selectbox(
                "Traffic Control",
                sorted(df["traffictl"].unique())
            )

            impactype = st.selectbox(
                "Impact Type",
                sorted(df["impactype"].unique())
            )

            visible = st.selectbox(
                "Visibility / Weather",
                sorted(df["visible"].unique())
            )

        with col2:
            light = st.selectbox(
                "Lighting Condition",
                sorted(df["light"].unique())
            )

            rdsfcond = st.selectbox(
                "Road Surface Condition",
                sorted(df["rdsfcond"].unique())
            )

            road_class = st.selectbox(
                "Road Class",
                sorted(df["road_class"].unique())
            )

            wardname = st.selectbox(
                "Ward",
                sorted(df["wardname"].unique())
            )

            division = st.selectbox(
                "Police Division",
                sorted(df["division"].unique())
            )

        neighbourhood = st.selectbox(
            "Neighbourhood",
            sorted(df["neighbourhood"].unique())
        )

        st.subheader("Location")

        location_col1, location_col2 = st.columns(2)

        with location_col1:
            longitude = st.number_input(
                "Longitude",
                value=float(df["longitude"].median()),
                format="%.6f"
            )

        with location_col2:
            latitude = st.number_input(
                "Latitude",
                value=float(df["latitude"].median()),
                format="%.6f"
            )

        st.subheader("Date and Time")

        date_col1, date_col2 = st.columns(2)

        with date_col1:
            collision_date = st.date_input(
                "Collision Date"
            )

        with date_col2:
            collision_time = st.time_input(
                "Collision Time"
            )

        st.subheader("Collision Factors")

        factor_col1, factor_col2, factor_col3 = st.columns(3)

        with factor_col1:
            failtorem = st.checkbox("Failed to Remain")
            aggressive = st.checkbox("Aggressive Driving")
            distracted = st.checkbox("Distracted")

        with factor_col2:
            cyclist = st.checkbox("Cyclist Involved")
            motorcyclist = st.checkbox("Motorcyclist Involved")
            other_micromobility = st.checkbox(
                "Other Micromobility Involved"
            )
            older_adult = st.checkbox("Older Adult Involved")

        with factor_col3:
            pedestrian = st.checkbox("Pedestrian Involved")
            red_light = st.checkbox("Red Light Violation")
            school_child = st.checkbox("School Child Involved")
            heavy_truck = st.checkbox("Heavy Truck Involved")

        submitted = st.form_submit_button(
            "Predict Collision Outcome"
        )

    if submitted:

        collision_datetime = pd.Timestamp.combine(
            collision_date,
            collision_time
        )

        input_data = pd.DataFrame(
            [{
                "per_inv": per_inv,
                "accloc": accloc,
                "traffictl": traffictl,
                "impactype": impactype,
                "visible": visible,
                "light": light,
                "rdsfcond": rdsfcond,
                "road_class": road_class,
                "failtorem": failtorem,
                "longitude": longitude,
                "latitude": latitude,
                "wardname": wardname,
                "division": division,
                "neighbourhood": neighbourhood,
                "aggressive": aggressive,
                "distracted": distracted,
                "cyclist": cyclist,
                "motorcyclist": motorcyclist,
                "other_micromobility": other_micromobility,
                "older_adult": older_adult,
                "pedestrian": pedestrian,
                "red_light": red_light,
                "school_child": school_child,
                "heavy_truck": heavy_truck,
                "year": collision_datetime.year,
                "month": collision_datetime.month,
                "day_of_week": collision_datetime.weekday(),
                "hour": collision_datetime.hour
            }]
        )

        prediction, probabilities = predictor.predict(
            input_data
        )

        predicted_class = int(prediction[0])

        non_fatal_probability = probabilities[0][0] * 100
        fatal_probability = probabilities[0][1] * 100

        if predicted_class == 1:
            st.error("Predicted Outcome: Fatal Injury")
        else:
            st.success("Predicted Outcome: Non-Fatal Injury")

        prob_col1, prob_col2 = st.columns(2)

        prob_col1.metric(
            "Non-Fatal Probability",
            f"{non_fatal_probability:.1f}%"
        )

        prob_col2.metric(
            "Fatal Probability",
            f"{fatal_probability:.1f}%"
        )


# --------------------------------------------------
# Model Performance page
# --------------------------------------------------

elif page == "Model Performance":
    st.title("Model Performance")

    st.write(
        "This page summarizes the performance of the selected "
        "Random Forest classifier on the final test set."
    )

    # --------------------------------------------------
    # Final test metrics
    # --------------------------------------------------

    st.subheader("Final Test Metrics")

    metrics_df = pd.DataFrame(
        {
            "Metric": [
                "Accuracy",
                "Precision",
                "Recall",
                "F1-score",
                "ROC AUC"
            ],
            "Score": [
                0.772,
                0.273,
                0.375,
                0.316,
                0.687
            ]
        }
    )

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------

    st.subheader("Confusion Matrix")

    confusion_df = pd.DataFrame(
        [
            [822, 160],
            [100, 60]
        ]
    )

    confusion_df.index = [
        "Actual Non-Fatal",
        "Actual Fatal"
    ]

    confusion_df.columns = [
        "Predicted Non-Fatal",
        "Predicted Fatal"
    ]

    st.caption(
        "The model correctly classified 822 Non-Fatal collisions "
        "and 60 Fatal collisions. It produced 160 false positives "
        "and 100 false negatives."
    )
    fig, ax = plt.subplots(figsize=(2, 1.5))

    sns.heatmap(
        confusion_df,
        annot=True,
        fmt="d",    

        cmap="Blues",
        cbar=False,
        ax=ax,
        annot_kws={"size": 5}
    )

    ax.set_title(
    "Random Forest Test Confusion Matrix",
    fontsize=7
    )
    ax.set_xlabel(
        "Predicted Class",
        fontsize=5
    )
    ax.set_ylabel(
        "Actual Class",
        fontsize=6
    )

    ax.tick_params(
        axis="both",
        labelsize=4
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    st.caption(
        "The confusion matrix shows 822 correct Non-Fatal predictions, "
        "60 correct Fatal predictions, 160 false positives, "
        "and 100 false negatives."
    )
    
    st.subheader("ROC Curve")

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.plot(
        roc_df["false_positive_rate"],
        roc_df["true_positive_rate"],
        label="Random Forest (AUC = 0.687)"
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier"
    )

    ax.set_title(
        "Random Forest Test ROC Curve",
        fontsize=8
    )

    ax.set_xlabel(
        "False Positive Rate",
        fontsize=6
    )

    ax.set_ylabel(
        "True Positive Rate",
        fontsize=6
    )

    ax.tick_params(
        axis="both",
        labelsize=5
    )

    ax.legend(
        fontsize=5,
        loc="lower right"
    )

    ax.grid(
        alpha=0.3
    )

    st.pyplot(
        fig,
        use_container_width=False
    )

    st.caption(
        "The Random Forest achieved a test ROC AUC of 0.687, "
        "showing moderate ability to distinguish Fatal from "
        "Non-Fatal collisions."
    )

    # --------------------------------------------------
    # Model comparison
    # --------------------------------------------------

    st.subheader("Model Comparison")

    comparison_df = pd.DataFrame(
    {
            "Model": [
                "Decision Tree",
                "Tuned Random Forest"
            ],
            "Accuracy": [
                0.625,
                0.772
            ],
            "Precision": [
                0.187,
                0.273
            ],
            "Recall": [
                0.500,
                0.375
            ],
            "F1-score": [
                0.272,
                0.316
            ]
        }
    )
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )
    
    st.info(
        "Selected model: Tuned Random Forest. "
        "It was chosen because it achieved the best overall balance "
        "of accuracy, precision, and F1-score on the test set, "
        "although the Decision Tree achieved higher recall for Fatal collisions."
    )
