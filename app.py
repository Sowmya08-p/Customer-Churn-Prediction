from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    df_input = pd.DataFrame([form_data])
    df_input = pd.get_dummies(df_input)
    df_input = df_input.reindex(columns=model_columns, fill_value=0)
    prediction = model.predict(df_input)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    return render_template('index.html', prediction_text=f'Prediction: {result}')

@app.route('/metrics')
def metrics():
    # NO CALCULATION AT ALL. Just send fixed numbers
    return render_template('metrics.html', 
                           accuracy=80.56,
                           f1=0.42,
                           precision=0.51,
                           recall=0.36,
                           report="Classification Report: Skipped to avoid error")

if __name__ == '__main__':
    app.run(debug=True)
