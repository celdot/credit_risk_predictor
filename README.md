# Credit Default Prediction

This project predicts the likelihood of a loan applicant defaulting based on personal and financial attributes. It features a user-facing web form and a backend inference pipeline deployed on Google Cloud. The frontend is hosted using GitHub Pages.

## Project Overview

- **Objective**: Predict whether a loan applicant is likely to default.
- **Interface**: A web form where users can input loan and personal details.
- **Backend**: A Flask server processes input and feeds it to a trained model.
- **Deployment**:
  - Backend hosted on Google Cloud
  - Frontend (HTML form) hosted with GitHub Pages

## Dataset

The dataset is sourced from Kaggle and includes the following attributes:

| Feature         | Description                                                  |
|-----------------|--------------------------------------------------------------|
| ID              | Unique identifier for each loan applicant                    |
| Age             | Age of the loan applicant                                    |
| Income          | Annual income of the applicant                               |
| Home            | Home ownership status (Own, Mortgage, Rent)                  |
| Emp_Length      | Length of employment in years                                |
| Intent          | Purpose of the loan (e.g., education, home improvement)      |
| Amount          | Loan amount requested                                        |
| Rate            | Interest rate on the loan                                    |
| Status          | Loan status (Fully Paid, Charged Off, Current)               |
| Percent_Income  | Loan amount as a percentage of income                        |
| Default         | Whether the applicant previously defaulted (Yes, No)         |
| Cred_Length     | Length of the applicant’s credit history                     |

## Results

Here are the evaluation results (before hyperparameter tuning) for various models:

| Model                     | Accuracy | F1 Score | Precision | Recall |
|---------------------------|----------|----------|-----------|--------|
| Logistic Regression       | 0.8211   | 0.3820   | 0.5154    | 0.3035 |
| Decision Tree             | 0.8262   | 0.5289   | 0.5223    | 0.5357 |
| Random Forest             | 0.8221   | 0.4942   | 0.5126    | 0.4771 |
| Support Vector Classifier | 0.8173   | 0.3043   | 0.4964    | 0.2194 |
| K-Nearest Neighbors       | 0.8105   | 0.4104   | 0.4735    | 0.3621 |
| Naive Bayes               | 0.8175   | 0.4632   | 0.4988    | 0.4324 |
| Stochastic Gradient Descent | 0.8194 | 0.3521   | 0.5080    | 0.2694 |
| XGBoost                   | 0.8192   | 0.4989   | 0.5038    | 0.4941 |
| AdaBoost                  | 0.8171   | 0.5196   | 0.4980    | 0.5431 |
| LightGBM                  | 0.8229   | 0.5167   | 0.5137    | 0.5197 |
| Gradient Boosting         | 0.8231   | 0.5118   | 0.5145    | 0.5091 |
| CatBoost                  | 0.8231   | 0.5107   | 0.5146    | 0.5069 |

### Final Evaluation Metrics

**Confusion Matrix:**

```
[[2479  199]
 [ 368  213]]
```

- **Test Accuracy**: 0.8260  
- **Validation Accuracy**: 0.8223  
- **Training set size**: 20,619  
- **Validation set size**: 5,155  
- **Test set size**: 2,864  

**Classification Report:**

```
              precision    recall  f1-score   support
           0       0.87      0.93      0.90      2678
           1       0.52      0.37      0.43       581
    accuracy                           0.83      3259
   macro avg       0.69      0.65      0.66      3259
weighted avg       0.81      0.83      0.81      3259
```

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/celdot/credit-default-prediction.git
cd credit-default-prediction
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the Flask server:

```bash
python app.py
```

5. Access the form in your browser:

```
https://celdot.github.io/credit_risk_predictor/
```

## Try it Live

- **Frontend** (GitHub Pages): [Visit Website](https://celdot.github.io/credit_risk_predictor/)
- **Backend** (Google Cloud): Receives user input and returns model predictions

