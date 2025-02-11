import streamlit as st
import pickle
import numpy as np
import pandas as pd

# load model
pipe = pickle.load(open("data/car_price_model.pkl", "rb"))

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗")
st.title("Car Price Predictor")
st.write("Enter the car details below to get an estimated resale price.")

col1, col2 = st.columns(2)

with col1:
    name = st.text_input("Car Name (e.g. Maruti Swift Dzire)")
    company = st.text_input("Company (e.g. Maruti)")
    year = st.number_input("Year of Purchase", min_value=1990,
                            max_value=2023, value=2015)

with col2:
    kms_driven = st.number_input("Kilometers Driven", min_value=0,
                                  max_value=500000, value=30000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG",
                                             "LPG", "Electric"])

if st.button("Predict Price", use_container_width=True):
    query = pd.DataFrame([[name, company, year, kms_driven, fuel_type]],
                          columns=["name", "company", "year",
                                   "kms_driven", "fuel_type"])
    predicted_price = int(pipe.predict(query)[0])

    if predicted_price < 0:
        st.warning("Could not estimate price for this combination. "
                   "Please check your inputs.")
    else:
        st.success(f"Estimated Resale Price: Rs {predicted_price:,}")
