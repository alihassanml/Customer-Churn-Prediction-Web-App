import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Load the trained model and encoders
my_model = load_model('model.h5')
with open('standard_scalar.pkl', 'rb') as file:
    standard_scalar = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

with open('one_hot_encoder.pkl', 'rb') as file:
    one_hot_encoder = pickle.load(file)

# Streamlit app
st.title('Customer Churn Prediction')

# Input fields
credit_score = st.number_input('Credit Score', min_value=0, max_value=1000, value=600)
geography = st.selectbox('Geography', ['France', 'Spain', 'Germany'])
gender = st.selectbox('Gender', ['Male', 'Female'])
age = st.number_input('Age', min_value=18, max_value=100, value=40)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=3)
balance = st.number_input('Balance', min_value=0, value=600000)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=10, value=2)
has_cr_card = st.selectbox('Has Credit Card?', [0, 1])
is_active_member = st.selectbox('Is Active Member?', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0, value=50000)

# Create input data dictionary
input_data = {
    'CreditScore': credit_score,
    'Geography': geography,
    'Gender': gender,
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

if st.button('Predict'):
    # Process the input data
    geo_encoder = one_hot_encoder.transform([[input_data['Geography']]]).toarray()
    data = pd.DataFrame([input_data])
    geo_df = pd.DataFrame(geo_encoder, columns=one_hot_encoder.get_feature_names_out(['Geography']))

    input_data_processed = pd.concat([data.drop('Geography', axis=1), geo_df], axis=1)
    input_data_processed['Gender'] = label_encoder.transform(input_data_processed['Gender'])
    input_data_processed = standard_scalar.transform(input_data_processed)

    # Predict
    prediction = my_model.predict(input_data_processed)

    # Display the prediction
    if prediction[0][0] > 0.5:
        st.write('The Customer is likely to churn.')
    else:
        st.write('The Customer is not likely to churn.')

