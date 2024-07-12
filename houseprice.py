from flask import Flask, request, jsonify
import pandas as pd

import joblib

app = Flask(__name__)

# Load the saved model
best_gb_model = joblib.load('best_gb_model.pkl')

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict_1():
    # Get JSON data from the request
    data = request.get_json(force=True)
    
    # Convert JSON to DataFrame
    input_df = pd.DataFrame(data, index=[0])
    
    # Define the features used for prediction
    features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'TotalSF', 'GarageArea', 'WoodDeckSF',
                'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']
    
    # Ensure the input data matches the expected format
    input_df = input_df[features].astype(float)
    
    # Make prediction
    prediction = best_gb_model.predict(input_df)
    
    # Return the prediction result
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
