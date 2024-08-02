import streamlit as st
import pandas as pd
import joblib
import warnings
from PIL import Image
warnings.filterwarnings("ignore")

# Load model and features
rfc_model = joblib.load('RFC-9910.joblib')
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

# Sidebar for user inputs
st.sidebar.markdown("### Input Customer Data")

def user_input_features():
    inputs = {}
    for feature in original_features:
        if 'Charges' in feature:
            inputs[feature] = st.sidebar.text_input(feature, "Enter value")
        elif 'tenure_group' in feature:
            inputs[feature] = st.sidebar.selectbox(feature, options=["No", "Yes"], index=0)
        else:
            inputs[feature] = st.sidebar.selectbox(feature, options=["No", "Yes"], index=0)
    features = pd.DataFrame(inputs, index=[0])
    return features

input_df = user_input_features()

# Display user inputs in main section
st.markdown('<div class="input-container">', unsafe_allow_html=True)
st.subheader('User Input Features')
st.write(input_df)
st.markdown('</div>', unsafe_allow_html=True)

# Prediction
if st.sidebar.button('Predict'):
    input_df = input_df.replace({'Yes': 1, 'No': 0})
    prediction = rfc_model.predict(input_df)
    
    st.markdown('<div class="prediction-container">', unsafe_allow_html=True)
    st.subheader('Prediction')
    st.write('Churn' if prediction[0] == 1 else 'No Churn')
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="separator"></div>', unsafe_allow_html=True)

# Additional Information
st.markdown("""
<div class="additional-info">
    <h3>About the Model</h3>
    <p>This application uses a Random Forest Classifier to predict customer churn based on various input features. 
    Adjust the input values in the sidebar to see the prediction change.</p>
</div>
""", unsafe_allow_html=True)

# Close container
st.markdown('</div>', unsafe_allow_html=True)
