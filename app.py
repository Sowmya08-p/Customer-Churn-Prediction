from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('churn_model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.to_dict()
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    # One-hot encoding for categorical columns
    df = pd.get_dummies(df)
    
    # Align columns with training data
    df = df.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(df)
    probability = model.predict_proba(df)[0][1]
    
    result = "YES - Will Churn" if prediction[0] == 1 else "NO - Will Stay"
    return render_template('index.html', prediction=result, prob=round(probability*100, 2))

if __name__ == '__main__':
    app.run(debug=True)
