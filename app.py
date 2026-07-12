from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

app = Flask(__name__)

# Load model and columns
model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

# Load data for metrics page
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(0, inplace=True)
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_dict = request.form.to_dict()
    input_df = pd.DataFrame([input_dict])
    
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    result = 'YES - Will Churn' if prediction == 1 else 'NO - Will Not Churn'
    return render_template('index.html', prediction=result, prob=round(probability*100, 2))

@app.route('/metrics')
def metrics():
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X = pd.get_dummies(X)
    X = X.reindex(columns=model_columns, fill_value=0)
    
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    return render_template('metrics.html', 
                           accuracy=round(accuracy*100, 2),
                           f1=round(f1, 4),
                           precision=round(precision, 4),
                           recall=round(recall, 4))

if __name__ == '__main__':
    app.run(debug=True)
