# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Load the saved model and scaler together
loaded_objects = pickle.load(open('C:/Users/sumit/Desktop/deployModel/trained_model_and_scaler.sav', 'rb'))

# Extract model and scaler
loaded_model = loaded_objects['model']
scaler = loaded_objects['scaler']

# Make a predictive system
input_data = (7, 137, 90, 41, 0, 32, 0.391, 39)

# Change data to numpy
input_data_as_numpy_array = np.asarray(input_data)

# Reshape the array as we are predicting only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# Standardise the input data
std_data = scaler.transform(input_data_reshaped)
print("Standardized input data:", std_data)

# Make prediction
prediction = loaded_model.predict(std_data)
print("Prediction result:", prediction)

if prediction[0] == 0:
    print("non diabetic")
else:
    print("diabetic")
