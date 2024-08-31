import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template
import os

app = Flask(__name__)

# Get the directory of the current script
base_dir = os.path.dirname(__file__)

# Construct file paths relative to the script location
svr_path = os.path.join(base_dir, 'svr.pkl')
encoder_path = os.path.join(base_dir, 'encoder.pkl')
scaler_path = os.path.join(base_dir, 'scaler.pkl')

# Load the models and preprocessing objects
try:
    with open(svr_path, 'rb') as file:
        svr = pickle.load(file)
    with open(encoder_path, 'rb') as file:
        encoder = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except Exception as e:
    print(f"Error loading files: {e}")
    exit(1)

@app.route('/')
def index():
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = {
            'Credit Card Limit': int(request.form.get('Credit Card Limit')),
            'emi_active': int(request.form.get('emi_active')),
            'age': int(request.form.get('age')),
            'Emp_Tenure_Years': int(request.form.get('Emp_Tenure_Years')),
            'Tenure_with_Bank': int(request.form.get('Tenure_with_Bank')),
            'NetBanking': int(request.form.get('NetBanking')),
            'Monthly Viewership': int(request.form.get('Monthly Viewership')),
            'Monthly Expense': int(request.form.get('Monthly Expense')),
            'Monthly Income': int(request.form.get('Monthly Income')),
            'Marital Status': request.form.get('Marital Status'),
            'Target Audience': request.form.get('Target Audience'),
            'Channel': request.form.get('Channel')
        }

        # Encode 'Channel'
        Channel_encoding_map = {'Star Plus': 0, 'Sab TV': 1, 'Colors': 2}
        form_data['encoded_Channel'] = Channel_encoding_map.get(form_data['Channel'], -1)

        # Convert form data to DataFrame
        data = pd.DataFrame([form_data])

        # One-Hot Encode categorical variables
        categorical_features = data[['Marital Status', 'Target Audience']]
        encoded_features = encoder.transform(categorical_features)
        encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Marital Status', 'Target Audience']))

        # Drop original categorical columns and the 'Channel' column
        data = data.drop(['Marital Status', 'Target Audience', 'Channel'], axis=1)

        # Concatenate the encoded features
        data = pd.concat([data, encoded_features_df], axis=1)

        # Reorder the DataFrame to match the expected column order
        expected_columns = [
            'Credit Card Limit', 'emi_active', 'age', 'Emp_Tenure_Years', 
            'Tenure_with_Bank', 'NetBanking', 'Monthly Viewership', 
            'Monthly Expense', 'Monthly Income','encoded_Channel',
            'Marital Status_Married', 'Marital Status_Unmarried', 
            'Target Audience_Female', 'Target Audience_Male'
        ]
        data = data.reindex(columns=expected_columns, fill_value=0)

        # Apply StandardScaler
        data_scaled = scaler.transform(data)

        # Make prediction
        prediction = svr.predict(data_scaled)

        audience_selection = int(round(prediction[0]))
        result = 'Yes' if audience_selection == 1 else 'No'

        return jsonify({
            
            'Audience Should be targeted': result
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
