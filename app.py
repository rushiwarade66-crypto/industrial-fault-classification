
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Industrial Fault Classification",
    page_icon="⚙️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("fault_model.pkl")

st.title("⚙️ Industrial Fault Classification")
st.write("Predict industrial machine faults using sensor readings and a Random Forest model.")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load the trained model: {e}")
    st.stop()

st.subheader("Enter Machine Sensor Readings")

temperature = st.number_input("Temperature (°C)", min_value=0.0, value=92.0)
vibration = st.number_input("Vibration (mm/s)", min_value=0.0, value=2.5)
rpm = st.number_input("RPM", min_value=0, value=1450)
current = st.number_input("Current (A)", min_value=0.0, value=6.2)

if st.button("Predict Fault", type="primary"):
    input_data = pd.DataFrame(
        [[temperature, vibration, rpm, current]],
        columns=["Temperature", "Vibration", "RPM", "Current"]
    )

    try:
        prediction = model.predict(input_data)[0]
        st.subheader("Prediction Result")
        st.success(f"Predicted condition: {prediction}")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_data)[0]
            results = pd.DataFrame({
                "Condition": model.classes_,
                "Confidence (%)": probabilities * 100
            })
            results["Confidence (%)"] = results["Confidence (%)"].round(2)
            st.dataframe(results, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.caption("Academic mini project | Random Forest classifier")
 
