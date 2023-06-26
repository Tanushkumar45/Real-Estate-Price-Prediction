import numpy as np
import pandas as pd
import pickle
import json
import streamlit as st

# Load the saved model
model = pickle.load(open('model.pkl', 'rb'))

# Load the column names
with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']

def predict_price(location, sqft, bath, bhk):
    loc_index = data_columns.index(location.lower())

    x = np.zeros(len(data_columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return model.predict([x])[0]

def main():
    st.title("House Price Prediction")

    sqft = st.number_input("Enter the square feet area:", min_value=1.0)
    bhk = st.number_input("Enter the number of bedrooms (BHK):", min_value=1)
    bath = st.number_input("Enter the number of bathrooms:", min_value=1)

    location = st.selectbox("Select the location:", data_columns)

    if st.button("Predict"):
        price = predict_price(location, sqft, bath, bhk)
        st.success(f"The predicted price of the house is: {price:.2f} Lakhs")

main()