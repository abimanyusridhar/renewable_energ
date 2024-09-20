from flask import Flask, request, jsonify, render_template
import logging
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize global variables for model, scaler, and label encoders
model = None
scaler = None
label_encoders = None
feature_columns = None

# Load the trained model, scaler, label encoders, and feature columns
def load_artifacts():
    global model, scaler, label_encoders, feature_columns
    try:
        with open('house_price_model.pkl', 'rb') as model_file:
            model = joblib.load(model_file)
            app.logger.info("Model loaded successfully")
        
        with open('scaler.pkl', 'rb') as scaler_file:
            scaler = joblib.load(scaler_file)
            app.logger.info("Scaler loaded successfully")

        with open('label_encoders.pkl', 'rb') as encoders_file:
            label_encoders = joblib.load(encoders_file)
            app.logger.info("Label encoders loaded successfully")

        with open('feature_columns.pkl', 'rb') as columns_file:
            feature_columns = joblib.load(columns_file)
            app.logger.info("Feature columns loaded successfully")

    except Exception as e:
        app.logger.error(f"Error loading artifacts: {e}")
        raise e

# Load artifacts when the application starts
load_artifacts()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.info('Received data: %s', data)

        if not data or 'features' not in data:
            raise ValueError("Invalid input data: 'features' key not found.")

        features = data['features']
        app.logger.info('Raw features: %s', features)

        # Ensure the correct number of features are passed
        if len(features) != len(feature_columns):
            raise ValueError(f"Invalid number of features provided. Expected {len(feature_columns)}, got {len(features)}.")

        # Extract and encode location (first feature)
        location = features[0]
        if location not in label_encoders['location'].classes_:
            raise ValueError(f"Location '{location}' not recognized.")
        
        location_encoded = label_encoders['location'].transform([location])[0]

        # Extract numerical features (excluding location)
        numerical_features = np.array(features[1:], dtype=float).reshape(1, -1)

        # Scale numerical features
        numerical_features_scaled = scaler.transform(numerical_features)

        # Combine encoded location and scaled numerical features
        final_features = np.concatenate([[location_encoded], numerical_features_scaled[0]])

        # Predict using the model
        prediction = model.predict([final_features])
        app.logger.info('Prediction: %s', prediction)

        # Convert log prediction back to normal price
        predicted_price = np.expm1(prediction[0])  # Using expm1 since we log-transformed the price

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except ValueError as ve:
        error_message = f"Value Error: {str(ve)}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 400
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 500

# Analytics route
@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

# Run the app locally
if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=False)
