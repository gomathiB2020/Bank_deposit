import streamlit as st
import pandas as pd
import joblib
import json
import os
import shap
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "final_model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.json")

# -----------------------------
# Load model + threshold
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_threshold():
    with open(THRESHOLD_PATH) as f:
        data = json.load(f)
        return data["threshold"] if isinstance(data, dict) else data

model = load_model()
threshold = load_threshold()

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Bank Deposit Predictor", layout="centered")

st.title("🏦 Term Deposit Prediction App")
st.markdown("Enter customer details to predict subscription probability.")

# -----------------------------
# Inputs
# -----------------------------
age = st.slider("Age", 18, 80, 30)
balance = st.number_input("Balance", value=1000.0)
duration = st.number_input("Call Duration (seconds)", value=100.0)
campaign = st.number_input("Campaign Contacts", min_value=1, max_value=50, value=1)

housing = st.selectbox("Housing Loan", ["yes", "no"])
loan = st.selectbox("Personal Loan", ["yes", "no"])
default = st.selectbox("Default", ["yes", "no"])

job = st.selectbox("Job", ["admin", "blue-collar", "technician", "services", "management"])
marital = st.selectbox("Marital Status", ["single", "married", "divorced"])
education = st.selectbox("Education", ["primary", "secondary", "tertiary"])

contact = st.selectbox("Contact Type", ["unknown", "telephone", "cellular"])
month = st.selectbox("Month", ["jan", "feb", "mar", "apr", "may", "jun",
                               "jul", "aug", "sep", "oct", "nov", "dec"])

poutcome = st.selectbox("Previous Outcome", ["unknown", "failure", "other", "success"])

pdays = st.number_input("Days since last contact", value=999)
previous = st.number_input("Previous contacts", value=0)
day = st.number_input("Last contact day", value=1)

# -----------------------------
# Feature Engineering
# -----------------------------
age_group = "young" if age < 30 else "adult" if age < 60 else "senior"

campaign_intensity = campaign / (duration + 1)
balance_per_age = balance / (age + 1)
duration_balance = duration * balance

# -----------------------------
# Prepare input dataframe
# -----------------------------
input_df = pd.DataFrame({
    "age": [age],
    "balance": [balance],
    "duration": [duration],
    "campaign": [campaign],
    "housing": [housing],
    "loan": [loan],
    "default": [default],
    "job": [job],
    "marital": [marital],
    "education": [education],
    "contact": [contact],
    "month": [month],
    "poutcome": [poutcome],
    "pdays": [pdays],
    "previous": [previous],
    "day": [day],
    "age_group": [age_group],
    "campaign_intensity": [campaign_intensity],
    "balance_per_age": [balance_per_age],
    "duration_balance": [duration_balance]
})

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    prob = model.predict_proba(input_df)[0][1]
    pred = 1 if prob >= threshold else 0

    st.subheader("Result")

    st.metric("Probability of Subscription", f"{prob:.2%}")

    if pred == 1:
        st.success("Customer likely to SUBSCRIBE")
    else:
        st.error("Customer NOT likely to subscribe")

    st.write(f"Threshold used: {threshold:.2f}")

    # -----------------------------
    # SHAP EXPLANATION
    # -----------------------------
    st.subheader("Why this prediction? (SHAP Explanation)")

    try:
        if hasattr(model, "named_steps"):
            steps = list(model.named_steps.values())
            preprocessor = steps[0]
            ml_model = steps[-1]

            # Transform input
            X_transformed = preprocessor.transform(input_df)

            # SHAP
            explainer = shap.Explainer(ml_model)
            shap_values = explainer(X_transformed)

            # Plot
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], show=False)
            st.pyplot(fig)

        else:
            st.warning("SHAP not supported for this model structure.")

    except Exception as e:
        st.error(f"SHAP explanation failed: {e}")
