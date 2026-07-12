import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.fillna(0, inplace=True)

le = LabelEncoder()
for col in ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
            'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
            'PaperlessBilling', 'PaymentMethod']:
    df[col] = le.fit_transform(df[col])

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

X = df.drop(['customerID', 'Churn'], axis=1)
y = df['Churn']
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

print("Model Accuracy:", model.score(X, y))
joblib.dump(model, 'churn_model.pkl')
joblib.dump(list(X.columns), 'model_columns.pkl')
