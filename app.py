# app.py
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    try:
        age = int(request.form['age'])
        income = int(request.form['income'])
        living = int(request.form['living'])  # already passed as int from select
        employment = int(request.form['employment'])
        purpose = int(request.form['purpose'])  # already passed as int from select
        amount = int(request.form['amount'])
        interest = int(request.form['interest'])
        percentage = int(request.form['percentage'])
        status = int(request.form['status'])  # already passed as int from select
        bank_years = int(request.form['bank_years'])

        # Process or store the data as needed
        result = {
            "age": age,
            "income": income,
            "living": living,
            "employment": employment,
            "purpose": purpose,
            "amount": amount,
            "interest": interest,
            "percentage": percentage,
            "status": status,
            "bank_years": bank_years
        }
        
        print(f"Form submitted successfully with data: {result}")
        
        customer_id = 123456
        input_data = np.array([[customer_id, age, income, living, employment, purpose, amount, interest, percentage, status, bank_years]])
        scaler = StandardScaler()
        dataset = pd.read_csv("data/credit_risk_data_processed.csv")
        scaler = StandardScaler()
        X = dataset.drop('default', axis=1)
        X = scaler.fit_transform(X)
        input_data = scaler.transform(input_data)

        model_name = "Random Forest"
        # Load the model and make predictions here
        model = joblib.load(f"models/{model_name}.joblib")
        print(f"Model {model_name} loaded from models/{model_name}.joblib")
        print("Got information from the form, now predicting...")
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            prediction = "You will default on your loan."
        else:
            prediction = "You will not default on your loan."
        print(f"Prediction made: {prediction}")
        

        return f"<h2>Form submitted successfully!</h2><pre>{prediction}</pre>"

    except ValueError:
        return "Invalid input. Please ensure all numeric fields are filled with valid numbers."

if __name__ == '__main__':
    app.run(debug=True)
