import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import warnings
import os

warnings.filterwarnings("ignore")

# Define the path to the assets folder
assets_path = os.path.join(os.path.dirname(__file__), 'assets')

# Set page config
icon_path = os.path.join(assets_path, 'images', 'icon2.ico')
st.set_page_config(page_title="Customer Churn Prediction", page_icon=icon_path)

# Load model 
stacking_clf = joblib.load(os.path.join(assets_path, 'models', 'rfc&gb-965.joblib'))


# Load original features
original_features = pd.read_csv(os.path.join(assets_path, 'data', 'top_18_features.csv'))
original_features = original_features.columns.tolist()

# Load custom CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css(os.path.join(assets_path, 'styles', 'styles.css'))

# Add custom images
banner = Image.open(os.path.join(assets_path, 'images', 'customers.png'))
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

# Add this line here, just before the user input section
st.markdown('<h2 class="input-title">Customer Information</h2>', unsafe_allow_html=True)

# User Inputs
def user_input_features():
    inputs = {}
    for feature in original_features:
        if 'Charges' in feature:  # Numeric input for charges
            inputs[feature] = st.text_input(feature)  # Keep as a text input for numeric values like MonthlyCharges
        elif 'tenure_group' in feature:  # Binary input for tenure group (Yes/No)
            inputs[feature] = st.selectbox(feature, options=["No", "Yes"])
        else:  # Binary input for other features (Yes/No)
            inputs[feature] = st.selectbox(feature, options=["No", "Yes"])
    
    # Convert 'Yes'/'No' inputs to 1/0 internally for prediction
    for key in inputs:
        if inputs[key] == 'Yes':
            inputs[key] = 1
        elif inputs[key] == 'No':
            inputs[key] = 0
    
    features = pd.DataFrame(inputs, index=[0])
    return features

input_df = user_input_features()

# Display user inputs for transparency
st.subheader('User Input Features')
st.write(input_df.replace({1: 'Yes', 0: 'No'}))  # Display 'Yes'/'No' to the user

# Prediction
if st.button('Predict'):
    # Convert DataFrame to numpy array for model input
    input_array = input_df.values
    
    # Reshape if necessary (but this step might not be needed depending on the model)
    input_reshaped = input_array.reshape(1, -1)
    
    # Make prediction
    prediction = stacking_clf.predict(input_reshaped)
    
    # Display prediction result
    st.markdown('<div class="prediction-container"><h2>Prediction</h2><p>' + ('Churn' if prediction[0] == 1 else 'No Churn') + '</p></div>', unsafe_allow_html=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
