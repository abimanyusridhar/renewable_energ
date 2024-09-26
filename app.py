from flask import Flask, request, render_template
import pickle
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the trained model (HDF5 format, since Keras models use this format)
model_path = 'energy_generation_model.h5'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

try:
    model = load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Load the scalers and encoders
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('state_encoder.pkl', 'rb') as f:
    state_encoder = pickle.load(f)

with open('station_encoder.pkl', 'rb') as f:
    station_encoder = pickle.load(f)

with open('type_encoder.pkl', 'rb') as f:
    type_encoder = pickle.load(f)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        energy_type = request.form['type']
        state_name = request.form['state_name']
        station = request.form['station']
        installed_capacity = float(request.form['installed_capacity'])
        date = request.form['date']
        sector = request.form['sector']
        owner = request.form['owner']

        # Extract date components
        year = int(date.split('-')[0])
        month = int(date.split('-')[1])
        day_of_week = pd.to_datetime(date).dayofweek

        # Apply the necessary cyclical transformations for the date components
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)
        quarter = (month - 1) // 3 + 1

        # Encode the categorical inputs
        state_encoded = state_encoder.transform([state_name])[0]
        station_encoded = station_encoder.transform([station])[0]
        type_encoded = type_encoder.transform([energy_type])[0]


        # Create an input array with all features (make sure feature order matches the model)
        input_features = np.array([[state_encoded, station_encoded, sector, owner, type_encoded, installed_capacity, 
                                    month_sin, month_cos, day_of_week_sin, day_of_week_cos, year, quarter]])

        # Scale the input features
        input_features_scaled = scaler.transform(input_features)

        # Make a prediction
        prediction = model.predict(input_features_scaled)

        # Format the prediction to two decimal places
        predicted_value = round(prediction[0][0], 2)

        # Pass the prediction to the template
        return render_template('index.html', prediction=predicted_value)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)

