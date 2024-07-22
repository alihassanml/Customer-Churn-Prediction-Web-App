# Customer Churn Prediction Web App

This project is a web application for predicting customer churn using a pre-trained machine learning model. The application takes various customer features as input and predicts whether the customer is likely to churn or not.

## Features

- **User Input**: Users can input customer details such as credit score, geography, gender, age, tenure, balance, number of products, whether they have a credit card, if they are an active member, and estimated salary.
- **Real-time Prediction**: The app preprocesses the input data and uses a trained neural network model to predict the likelihood of customer churn.
- **User-friendly Interface**: Built with Streamlit, the app provides an easy-to-use interface for entering data and viewing predictions.

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Add the model and encoders files**:
    Ensure you have the following files in the project directory:
    - `model.h5`
    - `standard_scalar.pkl`
    - `label_encoder_gender.pkl`
    - `one_hot_encoder.pkl`

## Running the App

Run the Streamlit app with the following command:
```bash
streamlit run app.py
