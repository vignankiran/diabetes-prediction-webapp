import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv("diabetes.csv")
cols_to_clean = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

pipeline.fit(X, y)
joblib.dump(pipeline, "model.pkl")
print("Model saved to model.pkl")
