
import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ---------------- HEADER ----------------
st.title("❤️ Heart Disease Prediction")
st.subheader("🔬 ML Powered Prediction System")
st.write("Enter your health details below to check the predicted heart disease risk.")

st.info(
    "⚠️ This application is for educational purposes only and should not "
    "be used as a medical diagnosis."
)

# ---------------- PERSONAL DETAILS ----------------
st.markdown("### 👤 Personal Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("🎂 Age", 18, 100, 40)

with col2:
    sex = st.selectbox("👤 Sex", ["M", "F"])

# ---------------- HEART INFORMATION ----------------
st.markdown("### ❤️ Heart & Chest Information")

col1, col2 = st.columns(2)

with col1:
    chest_pain = st.selectbox(
        "💓 Chest Pain Type",
        ["ATA", "NAP", "TA", "ASY"]
    )

    resting_bp = st.number_input(
        "🩸 Resting Blood Pressure",
        min_value=80,
        max_value=200,
        value=120
    )

    cholesterol = st.number_input(
        "🧪 Cholesterol (mg/dL)",
        min_value=100,
        max_value=600,
        value=200
    )

with col2:
    fasting_bs = st.selectbox(
        "🍬 Fasting Blood Sugar > 120 mg/dL",
        [0, 1]
    )

    resting_ecg = st.selectbox(
        "📈 Resting ECG",
        ["Normal", "ST", "LVH"]
    )

    max_hr = st.slider(
        "💗 Maximum Heart Rate",
        60,
        220,
        150
    )

# ---------------- EXERCISE INFORMATION ----------------
st.markdown("### 🏃 Exercise & ECG Information")

col1, col2 = st.columns(2)

with col1:
    exercise_angina = st.selectbox(
        "🏃 Exercise-Induced Angina",
        ["Y", "N"]
    )

    oldpeak = st.slider(
        "📉 Oldpeak (ST Depression)",
        0.0,
        6.0,
        1.0,
        0.1
    )

with col2:
    st_slope = st.selectbox(
        "📊 ST Slope",
        ["Up", "Flat", "Down"]
    )

# ---------------- INPUT SUMMARY ----------------
with st.expander("📋 View Your Entered Information"):

    summary = pd.DataFrame({
        "Parameter": [
            "Age",
            "Sex",
            "Chest Pain",
            "Resting BP",
            "Cholesterol",
            "Fasting Blood Sugar",
            "Resting ECG",
            "Maximum Heart Rate",
            "Exercise Angina",
            "Oldpeak",
            "ST Slope"
        ],
        "Value": [
            age,
            sex,
            chest_pain,
            resting_bp,
            cholesterol,
            fasting_bs,
            resting_ecg,
            max_hr,
            exercise_angina,
            oldpeak,
            st_slope
        ]
    })

    st.table(summary)

# ---------------- PREDICTION ----------------
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "🔍 Predict Risk",
        use_container_width=True
    )

if predict_button:

    with st.spinner("🤖 Analyzing your health information..."):

        # Create raw input
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create dataframe
        input_df = pd.DataFrame([raw_input])

        # Add missing columns
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(scaled_input)[0]

    # ---------------- RESULT ----------------
    st.markdown("## 📊 Prediction Result")

    if prediction == 1:

        st.error(
            "⚠️ HIGH RISK OF HEART DISEASE"
        )

        st.warning(
            "The machine learning model predicts a higher risk. "
            "Please consult a qualified healthcare professional for "
            "proper evaluation."
        )

    else:

        st.success(
            "✅ LOW RISK OF HEART DISEASE"
        )

        st.info(
            "The machine learning model predicts a lower risk based "
            "on the information provided."
        )

    # ---------------- RESULT DETAILS ----------------
    st.markdown("### 🧠 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Prediction",
            "High Risk" if prediction == 1 else "Low Risk"
        )

    with col2:
        st.metric(
            "Model",
            "KNN"
        )

# ---------------- FOOTER ----------------
st.markdown("---")

st.caption(
    "❤️ Heart Disease Prediction System | "
    "Built with Python, Streamlit & Machine Learning | By Animesh"
)

