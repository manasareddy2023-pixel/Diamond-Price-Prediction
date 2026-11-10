
import streamlit as st
import pickle
import pandas as pd

# Load Trained Model
with open("knn_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# Page Configuration
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎"
)

# Application Title
st.title("💎 Diamond Price Prediction")

st.write("Enter diamond details to predict its price.")

# User Inputs
carat = st.number_input("Carat", min_value=0.1, max_value=5.0, value=1.0)
cut = st.selectbox("Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"])
color = st.selectbox("Color", ["D", "E", "F", "G", "H", "I", "J"])
clarity = st.selectbox("Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"])

depth = st.number_input("Depth", min_value=40.0, max_value=80.0, value=61.5)
table = st.number_input("Table", min_value=40.0, max_value=80.0, value=57.0)

x = st.number_input("Length (x)", min_value=0.0, value=5.0)
y = st.number_input("Width (y)", min_value=0.0, value=5.0)
z = st.number_input("Height (z)", min_value=0.0, value=3.0)

# Prediction Button
if st.button("Predict Diamond Price"):

    input_data = pd.DataFrame({
        "carat": [carat],
        "cut": [cut],
        "color": [color],
        "clarity": [clarity],
        "depth": [depth],
        "table": [table],
        "x": [x],
        "y": [y],
        "z": [z]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Diamond Price: ${prediction[0]:.2f}")
