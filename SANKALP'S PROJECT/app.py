import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("house_model.pkl")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Sankalp's House Price Prediction")
st.markdown("Predict the selling price of a house using Machine Learning.")

# User Inputs
col1, col2 = st.columns(2)

with col1:
    overallqual = st.slider("Overall Quality", 1, 10, 5)
    grlivarea = st.number_input("Living Area (sq ft)", 500, 5000, 1500)
    garagecars = st.slider("Garage Capacity", 0, 5, 2)
    garagearea = st.number_input("Garage Area", 0, 1500, 500)

with col2:
    totalbsmtsf = st.number_input("Basement Area", 0, 3000, 800)
    fullbath = st.slider("Bathrooms", 0, 5, 2)
    bedroom = st.slider("Bedrooms", 1, 10, 3)
    yearbuilt = st.number_input("Year Built", 1870, 2025, 2000)

if st.button("Predict Price"):

    input_data = pd.DataFrame({
        "OverallQual": [overallqual],
        "GrLivArea": [grlivarea],
        "GarageCars": [garagecars],
        "GarageArea": [garagearea],
        "TotalBsmtSF": [totalbsmtsf],
        "FullBath": [fullbath],
        "BedroomAbvGr": [bedroom],
        "YearBuilt": [yearbuilt]
    })

    prediction = model.predict(input_data)

    st.metric(
    "Estimated House Price",
    f"${prediction[0]:,.0f}"
)