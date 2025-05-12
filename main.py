from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load data and model
data = pd.read_csv('Clean_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Retrieve the input values from the form
        location = request.form.get('location')
        bhk = request.form.get('bhk')
        bathrooms = request.form.get('bath')
        sqft = request.form.get('total_sqft')

        # Ensure the inputs are valid
        if not location or not bhk or not bathrooms or not sqft:
            return jsonify({'error': 'All fields are required.'})

        # Convert the inputs to appropriate types
        bhk = int(bhk)
        bathrooms = int(bathrooms)
        sqft = float(sqft)

        # Validation for positive numbers
        if bhk <= 0 or bathrooms <= 0 or sqft <= 0:
            return jsonify({'error': 'All values must be greater than zero.'})

        # Prepare the input data
        input_data = pd.DataFrame([[location, sqft, bhk, bathrooms]], columns=['location', 'total_sqft', 'bhk', 'bath'])

        # Make the prediction
        prediction = pipe.predict(input_data)[0] * 1e5

        # Format the predicted price
        price_in_lakhs = prediction / 1e5  # Convert to lakhs
        formatted_price = f"The predicted price of the house is ₹{round(price_in_lakhs, 2)} lakhs"

        return jsonify({'price': formatted_price})

    except Exception as e:
        # In case of any error, return a detailed error message
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True, port=50001)
