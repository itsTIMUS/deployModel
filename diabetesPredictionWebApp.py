# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 13:44:50 2025

@author: sumit
"""

import numpy as np
import pickle
import streamlit as st

# Load the saved model and scaler together
loaded_objects = pickle.load(open('trained_model_and_scaler.sav', 'rb'))

# Extract model and scaler
loaded_model = loaded_objects['model']
scaler = loaded_objects['scaler']

# Creating a function for prediction 
def diabetes_prediction(input_data):
    # Convert input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data, dtype=float)

    # Reshape the array as we are predicting only one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Standardise the input data
    std_data = scaler.transform(input_data_reshaped)

    # Make prediction
    prediction = loaded_model.predict(std_data)

    if prediction[0] == 0:
        return "Non-diabetic"
    else:
        return "Diabetic"
    
    
def main():
    # Giving a title 
    st.title('Diabetes Prediction Web App')

    # Getting the input data from users 
    Pregnancies = st.text_input('Number of pregnancies')
    Glucose = st.text_input('Blood glucose level')
    BloodPressure = st.text_input('Blood pressure')
    SkinThickness = st.text_input('Skin thickness')
    Insulin = st.text_input('Blood insulin level')
    BMI = st.text_input('BMI')
    DiabetesPedigreeFunction = st.text_input('Diabetes pedigree function')
    Age = st.text_input('Your age')
    
    # Code for prediction 
    diagnosis = ''
    
    # Creating a button 
    if st.button('Diabetes test results'):
        try:
            input_values = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            input_values = [float(i) for i in input_values]  # Convert all inputs to float
            diagnosis = diabetes_prediction(input_values)
        except ValueError:
            diagnosis = '⚠ Please enter valid numerical values!'
    
    st.success(diagnosis)
    

if __name__ == '__main__':
    main()
