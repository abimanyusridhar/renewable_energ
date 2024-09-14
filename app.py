from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model, scaler, and label encoders
def load_artifacts():
    try:
        with open('fertilizer_model.pkl', 'rb') as model_file:
            model = pickle.load(model_file)
            app.logger.info("Model loaded successfully")

        with open('fertilizer_scaler.pkl', 'rb') as scaler_file:
            scaler = pickle.load(scaler_file)
            app.logger.info("Scaler loaded successfully")

        with open('label_encoders.pkl', 'rb') as le_file:
            label_encoders = pickle.load(le_file)
            app.logger.info("Label encoders loaded successfully")
            
        return model, scaler, label_encoders
    except Exception as e:
        app.logger.error(f"Error loading artifacts: {e}")
        raise e

# Load artifacts
model, scaler, label_encoders = load_artifacts()

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        app.logger.info('Received data: %s', data)

        if not data or 'features' not in data:
            raise ValueError("Invalid input data: 'features' key not found.")

        features = data['features']
        app.logger.info('Raw features: %s', features)

        if len(features) != 8:
            raise ValueError("Invalid number of features provided.")

        # Extract and encode soil_type and crop_type (categorical features)
        soil_type, crop_type = features[0], features[1]

        if soil_type not in label_encoders['Soil Type'].classes_:
            raise ValueError(f"Soil Type '{soil_type}' not recognized.")
        if crop_type not in label_encoders['Crop Type'].classes_:
            raise ValueError(f"Crop Type '{crop_type}' not recognized.")

        soil_type_encoded = label_encoders['Soil Type'].transform([soil_type])[0]
        crop_type_encoded = label_encoders['Crop Type'].transform([crop_type])[0]

        # Extract numerical features
        numerical_features = np.array(features[2:]).reshape(1, -1)

        # Scale only the numerical features (Temperature, Humidity, etc.)
        numerical_features_scaled = scaler.transform(numerical_features)

        # Combine encoded categorical and scaled numerical features
        final_features = np.concatenate([[soil_type_encoded, crop_type_encoded], numerical_features_scaled[0]])

        # Predict using the model
        prediction = model.predict([final_features])
        app.logger.info('Prediction: %s', prediction)

        fertilizer_recommendation = label_encoders['Fertilizer Name'].inverse_transform([int(prediction[0])])[0]

        return jsonify({'recommendation': fertilizer_recommendation})
    except ValueError as ve:
        error_message = f"Value Error: {str(ve)}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 400
    except Exception as e:
        error_message = f"Error during prediction: {str(e)}"
        app.logger.error(error_message)
        return jsonify({'error': error_message}), 500




@app.route('/analytics')
def analytics():
    return render_template('analytics.html')

if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=False)  # Change port here

