
import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Industrial Fault Monitoring",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# PROFESSIONAL DASHBOARD STYLING
# --------------------------------------------------
st.markdown("""
<style>
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 2.25rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #64748b;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 650;
        margin-top: 0.8rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.18);
        padding: 16px;
        border-radius: 12px;
    }

    .footer {
        color: #808080;
        font-size: 0.85rem;
        text-align: center;
        padding-top: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fault_model.pkl")


st.markdown(
    '<div class="main-title">⚙️ Industrial Fault Monitoring</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Machine condition classification using '
    'sensor readings and a Random Forest model</div>',
    unsafe_allow_html=True
)

try:
    model = load_model()
except Exception as e:
    st.error(
        "Unable to load fault_model.pkl. "
        "Check that the file exists in the GitHub repository "
        "and that its dependencies are compatible."
    )
    st.exception(e)
    st.stop()

# --------------------------------------------------
# SIDEBAR: SENSOR INPUTS
# --------------------------------------------------
st.sidebar.title("🛠️ Machine Inputs")
st.sidebar.caption("Enter the current machine readings.")

temperature = st.sidebar.number_input(
    "Temperature (°C)",
    min_value=0.0,
    max_value=500.0,
    value=92.0,
    step=1.0
)

vibration = st.sidebar.number_input(
    "Vibration (mm/s)",
    min_value=0.0,
    max_value=100.0,
    value=2.5,
    step=0.1
)

rpm = st.sidebar.number_input(
    "Motor Speed (RPM)",
    min_value=0,
    max_value=100000,
    value=1450,
    step=50
)

current = st.sidebar.number_input(
    "Current (A)",
    min_value=0.0,
    max_value=1000.0,
    value=6.2,
    step=0.1
)

predict_button = st.sidebar.button(
    "🔍 Analyze Machine",
    type="primary",
    use_container_width=True
)

st.sidebar.divider()
st.sidebar.info(
    "This demonstration uses manually entered readings. "
    "Live sensor monitoring requires hardware integration."
)

# --------------------------------------------------
# SENSOR SUMMARY
# --------------------------------------------------
st.markdown("### 📊 Sensor Overview")

m1, m2, m3, m4 = st.columns(4)

m1.metric("🌡️ Temperature", f"{temperature:.1f} °C")
m2.metric("〰️ Vibration", f"{vibration:.2f} mm/s")
m3.metric("⚙️ Motor Speed", f"{rpm:,} RPM")
m4.metric("🔌 Current", f"{current:.2f} A")

st.divider()

# --------------------------------------------------
# MODEL PREDICTION
# --------------------------------------------------
st.markdown("### 🧠 Machine Condition Analysis")

# Use the feature order saved by the trained model when available.
default_features = [
    "Temperature",
    "Vibration",
    "RPM",
    "Current"
]

if hasattr(model, "feature_names_in_"):
    feature_names = list(model.feature_names_in_)
else:
    feature_names = default_features

input_values = {
    "Temperature": temperature,
    "Vibration": vibration,
    "RPM": rpm,
    "Current": current
}

try:
    missing_features = [
        feature for feature in feature_names
        if feature not in input_values
    ]

    if missing_features:
        st.error(
            "The model expects features that are not provided by this app: "
            + ", ".join(missing_features)
        )
        st.stop()

    input_data = pd.DataFrame(
        [[input_values[feature] for feature in feature_names]],
        columns=feature_names
    )

except Exception as e:
    st.error(f"Could not prepare the sensor readings: {e}")
    st.stop()

# Run prediction when the button is clicked.
if predict_button:
    try:
        prediction = model.predict(input_data)[0]
        prediction_text = str(prediction)

        probabilities = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]

        # Confidence for the predicted class.
        confidence = None

        if probabilities is not None:
            classes = list(model.classes_)
            predicted_index = classes.index(prediction)
            confidence = float(probabilities[predicted_index]) * 100

        # Status display: based on the model's predicted class.
        class_name = prediction_text.strip().lower()

        if class_name == "normal":
            status_icon = "🟢"
            status_label = "NORMAL CLASSIFICATION"
            status_message = (
                "The model classified these readings as Normal."
            )
        elif class_name == "overheating":
            status_icon = "🔴"
            status_label = "OVERHEATING CLASSIFICATION"
            status_message = (
                "The model classified these readings as Overheating."
            )
        elif class_name == "excessive vibration":
            status_icon = "🟠"
            status_label = "EXCESSIVE VIBRATION CLASSIFICATION"
            status_message = (
                "The model classified these readings as Excessive Vibration."
            )
        elif class_name == "bearing fault":
            status_icon = "🔴"
            status_label = "BEARING FAULT CLASSIFICATION"
            status_message = (
                "The model classified these readings as Bearing Fault."
            )
        else:
            status_icon = "🔵"
            status_label = "CLASSIFICATION RESULT"
            status_message = "The model returned a classification."

        st.markdown(f"#### {status_icon} {status_label}")
        st.write(status_message)

        left, right = st.columns([1, 1])

        with left:
            st.markdown("##### Predicted Condition")
            st.metric(
                label="Model Prediction",
                value=prediction_text
            )

            if confidence is not None:
                st.metric(
                    label="Model Confidence",
                    value=f"{confidence:.2f}%"
                )
                st.progress(
                    max(0.0, min(confidence / 100.0, 1.0))
                )

        with right:
            st.markdown("##### 📈 Class Confidence Chart")

            if probabilities is not None:
                probability_df = pd.DataFrame({
                    "Condition": [str(c) for c in model.classes_],
                    "Confidence (%)": [
                        float(p) * 100 for p in probabilities
                    ]
                }).sort_values(
                    "Confidence (%)",
                    ascending=False
                )

                chart_data = probability_df.set_index("Condition")
                st.bar_chart(
                    chart_data,
                    y="Confidence (%)",
                    height=260
                )
            else:
                st.info(
                    "This model does not provide class probabilities, "
                    "so the confidence chart is unavailable."
                )

        # Detailed results
        st.divider()
        st.markdown("##### 📋 Detailed Classification Results")

        if probabilities is not None:
            results_df = pd.DataFrame({
                "Condition": [str(c) for c in model.classes_],
                "Confidence (%)": [
                    round(float(p) * 100, 2)
                    for p in probabilities
                ]
            }).sort_values(
                "Confidence (%)",
                ascending=False
            ).reset_index(drop=True)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True
            )

            st.caption(
                "Confidence values are model-estimated probabilities, "
                "not guarantees of real-world machine condition."
            )

        # Inputs used for the prediction
        with st.expander("View sensor readings used in this prediction"):
            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption(
            "Check that the saved model was trained using the expected "
            "four input features and compatible library versions."
        )

else:
    st.info(
        "Enter the machine sensor readings in the sidebar, "
        "then click **Analyze Machine** to see the classification."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.markdown(
    '<div class="footer">Industrial Fault Classification | '
    'Academic Mini Project | Random Forest + Streamlit</div>',
    unsafe_allow_html=True
)

