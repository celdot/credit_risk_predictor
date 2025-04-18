# app.py
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, send_from_directory
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def home():
    return send_from_directory('.', 'index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        form = request.form
        data = {
            "age": int(form['age']),
            "income": int(form['income']),
            "living": int(form['living']),
            "employment": int(form['employment']),
            "purpose": int(form['purpose']),
            "amount": int(form['amount']),
            "interest": float(form['interest']),
            "percentage": float(form['percentage']),
            "status": int(form['status']),
            "bank_years": int(form['bank_years'])
        }

        print(f"Form submitted successfully with data: {data}")
        
        input_data = np.array([list(data.values())])

        # Load dataset for scaling
        dataset = pd.read_csv("data/credit_risk_data_processed.csv")
        X = dataset.drop(['default', 'id'], axis=1)

        scaler = StandardScaler()
        scaler.fit(X)
        input_data_scaled = scaler.transform(input_data)

        model_name = "Random Forest"
        model = joblib.load(f"models/{model_name}.joblib")
        print(f"Model {model_name} loaded.")

        prediction = model.predict(input_data_scaled)
        result = "You will default on your loan." if prediction[0] == 1 else "You will not default on your loan."
        print(f"Prediction made: {result}")

        return f"<h2>Prediction Result</h2><p>{result}</p>"

    except Exception as e:
        return f"<h2>Error</h2><p>{str(e)}</p>"

if __name__ == '__main__':
    app.run(debug=True)
