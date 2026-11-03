import streamlit as st
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

# load the model
BASE_DIR = Path(__file__).resolve().parent

with open(BASE_DIR / 'house_price_model.pkl', 'rb') as file:
    model, scaler = pickle.load(file)

st.title("House price Prediction")
st.write("Enter the details of the house to pridict its price.")


#  input features
# Input features

longitude = st.number_input("Longitude", value=-122.0)

latitude = st.number_input("Latitude", value=37.0)

housing_median_age = st.number_input("Housing Median Age", min_value=1.0, value=20.0)

total_rooms = st.number_input("Total Rooms", min_value=1.0, value=1000.0)

total_bedrooms = st.number_input("Total Bedrooms", min_value=1.0, value=200.0)

population = st.number_input("Population", min_value=1.0, value=500.0)

households = st.number_input("Households", min_value=1.0, value=150.0)

median_income = st.number_input("Median Income", min_value=0.0, value=3.0)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
)

#  button 
if st.button("Predict Price"):
    # prepare the input data 
        input_data = pd.DataFrame([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income
    ]], columns=[
        'longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income'
    ])
        input_data = pd.get_dummies(input_data)

# Ensure same columns as training data
        input_data = input_data.reindex(
         columns=scaler.feature_names_in_,
            fill_value=0
        )

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
# log transformation of input data
        input_data[['total_rooms', 'total_bedrooms',
                'population', 'households']] = np.log1p(
        input_data[['total_rooms', 'total_bedrooms',
                    'population', 'households']]
    )
# scale the input data
        input_scaled = scaler.transform(input_data)

# make prediction 
        prediction = model.predict(input_scaled)

        st.success(f"Predicted House Price: ${prediction[0]:,.2f}")