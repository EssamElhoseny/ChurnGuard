import streamlit as st
import pandas as pd
import joblib

# Load model
rfc_model = joblib.load('RFC-9910.joblib')  # Update with the correct model file name

# Load original features
original_features = joblib.load('feature_names.joblib')

# Title
st.title("Customer Churn Prediction")

# User Inputs
def user_input_features():
    inputs = {}
    for feature in original_features:
        inputs[feature] = st.number_input(feature)
    features = pd.DataFrame(inputs, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Features')
st.write(input_df)

# Prediction
if st.button('Predict'):
    prediction = rfc_model.predict(input_df)
    # Display prediction
    st.subheader('Prediction')
    st.write('Churn' if prediction[0] == 1 else 'No Churn')
