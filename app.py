import streamlit as st
import pandas as pd
import joblib

# Load Model
model = joblib.load('model.pkl')

st.title("Stock Price Prediction Using ML")

st.write("Enter Stock Details")

open_price = st.number_input("Open Price")
high_price = st.number_input("High Price")
low_price = st.number_input("Low Price")
volume = st.number_input("Volume")

# Prediction Button
if st.button("Predict Closing Price"):

    input_data = pd.DataFrame({
        'Open': [open_price],
        'High': [high_price],
        'Low': [low_price],
        'Volume': [volume]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Closing Price: {prediction[0]:.2f}")