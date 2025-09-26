import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib  
import numpy as np
@st.cache_data
def load_data():
    return pd.read_csv("raw_data_sampled.csv", low_memory=False)
df_raw = load_data()
# Loading Model and Metadata
model = xgb.Booster()
model.load_model("model.json")
preprocessor = joblib.load("preprocessor.pkl")
feature_names = list(joblib.load("feature_names.pkl"))


st.title("🚗 Car Price Prediction Dashboard")
st.write("Pick your car details to get a predicted price and explanation.")


# User Inputs

col1, col2 = st.columns(2)

with col1:
    year = st.selectbox("Year", sorted(df_raw['year'].dropna().unique().tolist()))
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500000, step=1000)
    condition = st.selectbox("Condition", df_raw['condition'].dropna().unique())

with col2:
    fuel = st.selectbox("Fuel Type", df_raw['fuel'].dropna().unique().tolist())
    transmission = st.selectbox("Transmission", df_raw['transmission'].dropna().unique())
    manufacturer = st.selectbox("Manufacturer", sorted(df_raw['manufacturer'].dropna().unique()))
# Dynamic Model Selection based on other inputs
if manufacturer:
    filtered_df = df_raw.loc[
        (df_raw['year'] == year) & 
        (df_raw['fuel'] == fuel) &
        (df_raw['transmission'] == transmission) &
        (df_raw['manufacturer'] == manufacturer)
    ]

    # Get the unique models
    models_list = sorted(filtered_df['model'].dropna().unique())

    if not models_list:
        st.warning("No models found for the selected criteria.")
        car_model = "" 
    else:
        car_model = st.selectbox("Car Model", models_list)
else:
    car_model = st.selectbox("Car Model", ["Select Manufacturer First"])

# Prediction Time
if st.button("Predict Price")and car_model:
     
    raw_input = pd.DataFrame([{col: None for col in preprocessor.feature_names_in_}])

    # Fill in the ones we actually collect from the user
    raw_input.loc[0, "year"] = year
    raw_input.loc[0, "odometer"] = mileage
    raw_input.loc[0, "Age"] = 2024 - year
    raw_input.loc[0, "fuel"] = fuel
    raw_input.loc[0, "transmission"] = transmission
    raw_input.loc[0, "manufacturer"] = manufacturer
    raw_input.loc[0, "model"] = car_model
    raw_input.loc[0, "condition"] = condition
    
    # Apply preprocessing
    X_user = np.expm1(preprocessor.transform(raw_input))

    duser = xgb.DMatrix(X_user)

# Predict
    log_pred = model.predict(duser)[0]
    prediction = np.expm1(log_pred)
    st.success(f"💰 Estimated Price: ${prediction:,.2f}")
    
