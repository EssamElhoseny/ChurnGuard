import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

# Set page config (add this line)
st.set_page_config(page_title="Customer Churn Prediction", page_icon="favicon.ico")

# Load model
rfc_model = joblib.load('RFC-9910.joblib')

# Load original features
original_features = joblib.load('feature_names.joblib')

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("styles.css")

# Add custom images
banner = Image.open("Designer.png")
st.image(banner, use_column_width=True)

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
            
- **Numerical Features:** For features like `MonthlyCharges` and `TotalCharges`, enter the value manually in the text box.
- **Binary Features:** For binary features, select either "Yes" or "No":
  
  - **No:** Represents `0` or `False`.
  - **Yes:** Represents `1` or `True`.
</div>

<div class="prediction-interpretation">
            
### Prediction Interpretation:
            
- **Churn = Yes:** The customer is likely to churn.
- **Churn = No:** The customer is not likely to churn.
</div>
            
""", unsafe_allow_html=True)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# User Inputs
def user_input_features():
    inputs = {}
    for feature in original_features:
        if 'Charges' in feature:
            inputs[feature] = st.text_input(feature)  # Use text input for numerical features
        elif 'tenure_group' in feature:
            inputs[feature] = st.selectbox(feature, options=["No", "Yes"], format_func=lambda x: 'Yes' if x == "Yes" else 'No')
        else:
            inputs[feature] = st.selectbox(feature, options=["No", "Yes"], format_func=lambda x: 'Yes' if x == "Yes" else 'No')
    features = pd.DataFrame(inputs, index=[0])
    return features

input_df = user_input_features()

# Display user inputs
st.subheader('User Input Features')
st.write(input_df)

# Prediction
if st.button('Predict'):
    # Convert 'Yes'/'No' to 1/0
    input_df = input_df.replace({'Yes': 1, 'No': 0})
    prediction = rfc_model.predict(input_df)
    # Display prediction
    st.markdown('<div class="prediction-container"><h2>Prediction</h2><p>' + ('Churn' if prediction[0] == 1 else 'No Churn') + '</p></div>', unsafe_allow_html=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
