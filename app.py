import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bird Strike Prediction System",
    layout="centered"
)

# --------------------------------------------------
# Load models and encoders
# --------------------------------------------------
@st.cache_resource
def load_models():
    birdstrike_model = joblib.load("models/xgb_birdstrike_model.pkl")
    species_model = joblib.load("models/xgb_species_model.pkl")
    phase_model = joblib.load("models/xgb_phase_model.pkl")
    species_encoder = joblib.load("models/species_encoder.pkl")
    phase_encoder = joblib.load("models/phase_encoder.pkl")
    return birdstrike_model, species_model, phase_model, species_encoder, phase_encoder

birdstrike_model, species_model, phase_model, species_encoder, phase_encoder = load_models()

# --------------------------------------------------
# Helper function
# --------------------------------------------------
def create_user_input(date, timeofday, airport, visibility):
    date = pd.to_datetime(date)
    return pd.DataFrame([{
        "incident_year": date.year,
        "incident_month": date.month,
        "timeofday": timeofday,
        "airport": airport,
        "visibility": visibility
    }])

# --------------------------------------------------
# App UI
# --------------------------------------------------
st.title("✈️ Bird Strike Prediction System")

st.markdown(
    """
    Enter **departure flight details** to estimate:
    - **Bird strike probability**
    - **Most likely bird species**
    - **Most likely flight phase**
    """
)

with st.form("prediction_form"):
    date = st.date_input("Date of Flight")

    timeofday = st.selectbox(
        "Time of Day",
        ["Dawn", "Day", "Dusk", "Night"]
    )

    departure_airport = st.text_input(
        "Departure Airport (e.g., JFK Intl)"
    )

    visibility = st.selectbox(
        "Visibility",
        ["Clear", "Fog", "Rain", "Snow"]
    )

    submitted = st.form_submit_button("Run Prediction")

# --------------------------------------------------
# Prediction output
# --------------------------------------------------
if submitted:
    user_input = create_user_input(
        date, timeofday, departure_airport, visibility
    )

    # Predictions
    strike_prob = birdstrike_model.predict_proba(user_input)[:, 1][0]

    species_encoded = species_model.predict(user_input)[0]
    species = species_encoder.inverse_transform([species_encoded])[0]

    phase_encoded = phase_model.predict(user_input)[0]
    phase = phase_encoder.inverse_transform([phase_encoded])[0]

    # Display results
    st.success("✅ Prediction Results")

    st.metric("Bird Strike Probability (Departure)", f"{strike_prob:.2%}")
    st.metric("Most Likely Bird Species", species)
    st.metric("Most Likely Flight Phase", phase)

