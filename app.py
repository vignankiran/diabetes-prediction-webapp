import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.caption(
    "Built using Random Forest Classifier and the Pima Indians Diabetes Dataset"
)
st.markdown(
    "Machine Learning application built using the Pima Indians Diabetes Dataset."
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_excel("diabetes.csv")
    return df

df = load_data()

# -----------------------------
# DATA PREPROCESSING
# -----------------------------
cols_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

df[cols_to_clean] = df[cols_to_clean].replace(0, np.nan)
df.fillna(df.median(numeric_only=True), inplace=True)

# -----------------------------
# TRAIN MODEL
# -----------------------------
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Project Information")

st.sidebar.success(f"Model Accuracy: {accuracy:.2%}")

st.sidebar.write("Dataset Records:", len(df))
st.sidebar.write("Features:", len(X.columns))

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head())

# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
st.subheader("Feature Importance")

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(
    by="Importance",
    ascending=False
)

st.bar_chart(
    importance_df.set_index("Feature")
)

# -----------------------------
# PREDICTION SECTION
# -----------------------------
st.subheader("Patient Details")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies (0-20)",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose (70-180 mg/dL)",
        min_value=0,
        max_value=300,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure (60-120 mmHg)",
        min_value=0,
        max_value=200,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness (10-50 mm)",
        min_value=0,
        max_value=100,
        value=20
    )

with col2:
    insulin = st.number_input(
        "Insulin (15-276 μU/mL)",
        min_value=0,
        max_value=900,
        value=80
    )

    bmi = st.number_input(
        "BMI (18.5-24.9 Normal)",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function (0-3)",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age (1-120 years)",
        min_value=1,
        max_value=120,
        value=30
    )

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Diabetes Risk"):

    input_data = np.array([
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("⚠️ High Risk of Diabetes")
    else:
        st.success("✅ Low Risk of Diabetes")

    st.write(f"Probability: {probability:.2%}")

    st.progress(float(probability))

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown(
    "**Developed by Vignan Kiran Gali** | AI & Data Science"
)
