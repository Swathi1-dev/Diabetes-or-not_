import streamlit as st
import requests

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient details to predict diabetes.")

st.divider()

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)

glucose = st.number_input("Glucose Level", min_value=0, max_value=300, value=120)

blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)

skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

insulin = st.number_input("Insulin", min_value=0, max_value=1000, value=80)

bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)

diabetes_pedigree = st.number_input(
    "Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5
)

age = st.number_input("Age", min_value=1, max_value=120, value=30)

st.divider()

# Prediction button
if st.button("Predict Diabetes", use_container_width=True):
    data = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age,
    }

    try:
        response = requests.post("http://127.0.0.1:8001/predict", json=data)

        if response.status_code == 200:
            result = response.json()

            if result["prediction"].lower() == "Diabetes":
                st.error("Prediction: Diabetes")
            else:
                st.success("Prediction: No Diabetes")

        else:
            st.error(f"API Error: {response.text}")

    except requests.exceptions.ConnectionError:
        st.error("Unable to connect to FastAPI. Please start the backend.")
