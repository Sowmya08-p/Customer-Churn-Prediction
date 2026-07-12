from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('churn_model.pkl')
cols = joblib.load('model_columns.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final = [np.array(int_features)]
    prediction = model.predict(final)
    proba = model.predict_proba(final)[0][1]

    result = "YES - Customer will Churn" if prediction[0]==1 else "NO - Customer will Stay"
    return render_template('index.html',
                           prediction_text=f'Prediction: {result}',
                           prob_text=f'Churn Probability: {proba*100:.2f}%')

if __name__ == "__main__":
    app.run(debug=True)
