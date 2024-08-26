import pandas as pd
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the model and preprocessing objects
try:
    with open('C:\\Users\\Sham Sunder Chawla\\Desktop\\MLDEPLOYMENT\\svr.pkl', 'rb') as file:
        svr = pickle.load(file)
    with open('C:\\Users\\Sham Sunder Chawla\\Desktop\\MLDEPLOYMENT\\encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('C:\\Users\\Sham Sunder Chawla\\Desktop\\MLDEPLOYMENT\\scaler.pkl', 'rb') as file:
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
            'card_lim': float(request.form.get('card_lim')),
            'emi_active': int(request.form.get('emi_active')),
            'age': float(request.form.get('age')),
            'Emp_Tenure_Years': float(request.form.get('Emp_Tenure_Years')),
            'Tenure_with_Bank': float(request.form.get('Tenure_with_Bank')),
            'NetBanking_Flag': int(request.form.get('NetBanking_Flag')),
            'Total Credit Consumption': float(request.form.get('Total_Credit_Consumption')),
            'Total debit Consumption': float(request.form.get('Total_debit_Consumption')),
            'Total Investment': float(request.form.get('Total_Investment')),
            'account_type': request.form.get('account_type'),
            'gender': request.form.get('gender'),
            'Income': request.form.get('Income')
        }

        # Encode 'Income'
        income_encoding_map = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        form_data['encoded_Income'] = income_encoding_map.get(form_data['Income'].upper(), -1)

        # Convert form data to DataFrame
        data = pd.DataFrame([form_data])

        # One-Hot Encode categorical variables
        categorical_features = data[['account_type', 'gender']]
        encoded_features = encoder.transform(categorical_features)
        encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['account_type', 'gender']))

        # Drop original categorical columns
        data = data.drop(['account_type', 'gender'], axis=1)
        
        # Concatenate the encoded features
        data = pd.concat([data, encoded_features_df], axis=1)

        # Drop original 'Income' column since it is now encoded
        data = data.drop(['Income'], axis=1)

        # Define the expected columns
        expected_columns = [
            'card_lim', 'emi_active', 'age', 'Emp_Tenure_Years', 'Tenure_with_Bank', 
            'NetBanking_Flag', 'Total Credit Consumption', 'Total debit Consumption', 
            'Total Investment', 'encoded_Income', 'account_type_Current', 'account_type_Savings', 
            'gender_Female', 'gender_Male'
        ]

        # Reorder the DataFrame to match the expected column order
        data = data[expected_columns]

        # Apply StandardScaler
        data_scaled = scaler.transform(data)

        # Make prediction
        prediction = svr.predict(data_scaled)

        # Format prediction as an integer
        predicted_credit_limit = int(round(prediction[0]))

        return jsonify({
            'message': 'Credit limit prediction successful',
            'predicted_credit_limit': predicted_credit_limit
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
