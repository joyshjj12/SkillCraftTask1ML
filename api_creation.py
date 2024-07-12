# from flask import Flask, request, jsonify
# import pandas as pd
# import pickle

# app = Flask(__name__)

# # Load the saved model
# with open('best_gb_model.pkl', 'rb') as model_file:
#     best_gb_model = pickle.load(model_file)

# # Define the prediction route
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Get JSON data from POST request
#     data = request.get_json(force=True)

#     # Convert JSON to DataFrame
#     input_df = pd.DataFrame([data])

#     # Define the features used for prediction (ensure they match the model's trained features)
#     features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'TotalSF', 'GarageArea', 'WoodDeckSF',
#                 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

#     # Ensure input DataFrame has the correct columns and fill missing values
#     input_df = input_df[features].fillna(0)

#     # Make prediction using the model
#     prediction = best_gb_model.predict(input_df)

#     # Return prediction as JSON response
#     return jsonify({'Cost Prediction': prediction[0]})

# if __name__ == '__main__':
#     app.run(debug=True)



from flask import Flask, request, jsonify
import pandas as pd
import pickle

app = Flask(__name__)

# Load the saved model
with open('best_gb_model.pkl', 'rb') as model_file:
    best_gb_model = pickle.load(model_file)

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from POST request
        data = request.get_json(force=True)

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Define the features used for prediction (ensure they match the model's trained features)
        features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'TotalSF', 'GarageArea', 'WoodDeckSF',
                    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']

        # Ensure input DataFrame has the correct columns and fill missing values
        input_df = input_df[features].fillna(0)

        # Make prediction using the model
        prediction = best_gb_model.predict(input_df)

        # Return prediction as JSON response
        return jsonify({'Cost Prediction': float(prediction[0])})  # Ensure prediction is converted to float if necessary
    
    except Exception as e:
        return jsonify({'error': str(e)})  # Return error message in JSON format

if __name__ == '__main__':
    app.run(debug=True)
