from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(request.form[feature]) for feature in [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]]
    prediction = model.predict([input_data])[0]
    probability = model.predict_proba([input_data])[0][1]

    result = "Positive (High chance of diabetes)" if prediction == 1 else "Negative (Low chance of diabetes)"

    return render_template('index.html', prediction=result, probability=f"{probability:.2%}")

if __name__ == '__main__':
    app.run(debug=True)
