import streamlit as st
import pandas as pd
import joblib
import warnings 
warnings.filterwarnings("ignore")

# Load model
rfc_model = joblib.load('RFC-9910.joblib')

# Load original features
original_features = joblib.load('feature_names.joblib')

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Title
st.markdown('<div class="title">Customer Churn Prediction</div>', unsafe_allow_html=True)

# Container for content
st.markdown('<div class="container">', unsafe_allow_html=True)

# Instructions
st.markdown("""
**Welcome to the Customer Churn Prediction app!**

In this app, you can input various customer attributes to predict whether a customer is likely to churn or not.

<div class="instructions">
            
### Instructions:
            
- **Numerical Features:** For features like `MonthlyCharges` and `TotalCharges`, use the slider to set the value.
- **Binary Features:** For binary features, select either 0 or 1:
  - **0:** Represents `No` or `False`.
  - **1:** Represents `Yes` or `True`.
</div>

<div class="prediction-interpretation">
            
### Prediction Interpretation:
            
- **Churn = 1:** The customer is likely to churn.
- **Churn = 0:** The customer is not likely to churn.
</div>
            
""", unsafe_allow_html=True)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)


# User Inputs
def user_input_features():
    inputs = {}
    for feature in original_features:
        if 'Charges' in feature:
            inputs[feature] = st.slider(feature, min_value=0.0, max_value=10000.0, step=0.1)  # Customize the range as needed
        elif 'tenure_group' in feature:
            inputs[feature] = st.selectbox(feature, options=[0, 1])
        else:
            inputs[feature] = st.selectbox(feature, options=[0, 1])
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

# Close container
st.markdown('</div>', unsafe_allow_html=True)
