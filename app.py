import streamlit as st
import pandas as pd
import joblib

# Load files
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

# UI
st.title("🫀 Heart Disease Prediction By DANISH")
st.markdown("### 💉 Provide the following details")

age = st.slider("🎂 Age", 18, 100, 40)

sex = st.selectbox("🚻 Sex", ["M", "F"])

chest_pain = st.selectbox(
    "💔 Chest Pain Type",
    ["ATA", "NAP", "TA", "ASY"]
)

resting_bp = st.number_input(
    "🩺 Resting Blood Pressure (mm Hg)",
    80, 200, 120
)

cholesterol = st.number_input(
    "🧈 Cholesterol (mg/dL)",
    100, 600, 200
)

fasting_bs = st.selectbox(
    "🩸 Fasting Blood Sugar (>120 mg/dL)",
    [0, 1]
)

resting_ecg = st.selectbox(
    "📈 Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "❤️ Max Heart Rate",
    60, 220, 150
)

exercise_angina = st.selectbox(
    "🏃 Exercise-Induced Angina",
    ["Y", "N"]
)

oldpeak = st.slider(
    "📉 Oldpeak (ST Depression)",
    0.0, 6.0, 1.0
)

st_slope = st.selectbox(
    "📊 ST Slope",
    ["Up", "Flat", "Down"]
)

# Prediction
if st.button("🔍 Predict"):

    # Create dataframe with all expected columns
    input_df = pd.DataFrame(
        0,
        index=[0],
        columns=expected_columns
    )

    # Numerical features
    numeric_data = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak
    }

    for col, value in numeric_data.items():
        if col in input_df.columns:
            input_df.loc[0, col] = value

    # Categorical features
    dummy_columns = [
        f"Sex_{sex}",
        f"ChestPainType_{chest_pain}",
        f"RestingECG_{resting_ecg}",
        f"ExerciseAngina_{exercise_angina}",
        f"ST_Slope_{st_slope}"
    ]

    for col in dummy_columns:
        if col in input_df.columns:
            input_df.loc[0, col] = 1

    # Scale
    scaled_input = scaler.transform(input_df)

    # Predict
    prediction = model.predict(scaled_input)[0]

    # Debug info
    st.write("Raw Prediction:", prediction)

    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")