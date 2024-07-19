# ChurnGuard

ChurnGuard is a customer churn prediction tool built using a Random Forest Classifier model. The application is deployed using Streamlit to provide an interactive interface where users can input customer features and get predictions on whether the customer will churn or not.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Customer churn prediction is a critical task for businesses aiming to retain their customers. ChurnGuard uses a machine learning model to predict the likelihood of customer churn based on input features. The model is deployed using Streamlit to create an easy-to-use web application.

## Features

- **User-Friendly Interface**: Enter customer details and get instant churn predictions.
- **Machine Learning Model**: Utilizes a Random Forest Classifier for accurate predictions.
- **Interactive Visualizations**: Displays the input features used for the prediction.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/churnguard.git
   cd churnguard
   
2. **Install dependencies:**
Make sure you have Python installed. Then, install the required Python packages using the requirements.txt file:
   ```sh
    pip install -r requirements.txt

3. **Run the Streamlit app:**
   ```sh
   streamlit run app.py

## Usage

Once the application is running, open your web browser and go to http://localhost:8501. You will see the input fields for various customer features. Enter the values and click on the "Predict" button to see if the customer is likely to Churn.

## Project Structure

app.py: The main Streamlit application file.
feature_names.joblib: Serialized file containing the names of the features used by the model.
RFC_model.joblib: The pre-trained Random Forest Classifier model.
requirements.txt: List of Python dependencies required for the project.
Procfile: Configuration file for deploying the app on platforms like Streamlit.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, feel free to fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
