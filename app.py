from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

def clean(text):
    # Example of a simple preprocessing function
    return text.lower()

# Home page route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction page route
@app.route('/prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        reviewer = request.form.get("reviewer_name")
        product = request.form.get('product_name')
        review = request.form.get("review")

        # Validate that a review was provided
        if not review:
            message = 'Enter a review.'
            return render_template('home.html', message=message)

        try:
            # Load the model
            model = joblib.load('Predictive_Model/random_forest_model.pkl')
            vectorize = joblib.load('Predictive_Model/vectorizer.pkl')
            transformed_review = vectorize.transform([review])

            # Make the prediction using the transformed review
            prediction = model.predict(transformed_review)[0]
            return render_template('output.html', reviewer=reviewer, product=product, prediction=prediction)

        except Exception as e:
            # Handle model loading or prediction errors
            message = f"An error occurred: {str(e)}"
            return render_template('home.html', message=message)

    # For GET requests, render the home page
    return render_template('home.html')

if __name__ == "__main__":
    # Disable the reloader
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)
