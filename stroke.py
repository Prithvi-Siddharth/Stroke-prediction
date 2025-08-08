import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Set page title
st.title("Stroke Risk Prediction App")
st.markdown("Enter patient details to predict stroke risk using a Logistic Regression model trained on stroke data.")

# Load and preprocess data (for training the model)
@st.cache_data
def load_and_train_model():
    # Load data
    df = pd.read_csv("stroke-data.csv").replace('N/A', np.nan)
    df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')
    df.drop('id', axis=1, inplace=True)

    # Handle outliers
    def handle_outliers(df, columns, method='clip'):
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if method == 'clip':
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        return df_clean

    numeric_cols = ['age', 'avg_glucose_level', 'bmi']
    df_clean = handle_outliers(df, numeric_cols, method='clip')

    # Feature engineering
    df_clean['age_glucose'] = df_clean['age'] * df_clean['avg_glucose_level']
    df_clean['bmi_glucose'] = df_clean['bmi'] * df_clean['avg_glucose_level']
    df_clean['age_bmi'] = df_clean['age'] * df_clean['bmi']
    df_clean['glucose_bmi_ratio'] = df_clean['avg_glucose_level'] / (df_clean['bmi'] + 1)
    df_clean['age_bin'] = pd.cut(df_clean['age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])
    df_clean['bmi_category'] = pd.cut(df_clean['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
    df_clean['glucose_category'] = pd.cut(df_clean['avg_glucose_level'], bins=[0, 100, 125, 200, 1000], labels=['normal', 'prediabetic', 'diabetic', 'severe'])
    df_clean['risk_score'] = (
        df_clean['age'] * 0.3 +
        df_clean['avg_glucose_level'] * 0.2 +
        df_clean['bmi'] * 0.1 +
        df_clean['hypertension'] * 0.2 +
        df_clean['heart_disease'] * 0.2
    )
    df_clean['age_risk'] = df_clean['age'] * df_clean['risk_score']
    df_clean['glucose_risk'] = df_clean['avg_glucose_level'] * df_clean['risk_score']
    df_clean['bmi_risk'] = df_clean['bmi'] * df_clean['risk_score']
    df_clean['age_squared'] = df_clean['age'] ** 2
    df_clean['glucose_squared'] = df_clean['avg_glucose_level'] ** 2
    df_clean['bmi_squared'] = df_clean['bmi'] ** 2

    # Define features and target
    X = df_clean.drop('stroke', axis=1)
    y = df_clean['stroke'].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Define feature types
    num_features = ['age', 'avg_glucose_level', 'bmi', 'age_glucose', 'bmi_glucose', 'age_bmi',
                    'glucose_bmi_ratio', 'risk_score', 'age_risk', 'glucose_risk', 'bmi_risk',
                    'age_squared', 'glucose_squared', 'bmi_squared']
    cat_features = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                    'Residence_type', 'smoking_status', 'age_bin', 'bmi_category', 'glucose_category']

    # Create pipeline
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])
    cat_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('resample', SMOTE(random_state=42, k_neighbors=3)),
        ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Find optimal threshold
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
    optimal_threshold = thresholds[np.argmax(f1_scores)]

    return pipeline, optimal_threshold, num_features, cat_features

# Load model
pipeline, optimal_threshold, num_features, cat_features = load_and_train_model()

# User input form
st.subheader("Enter Patient Details")
with st.form("patient_form"):
    col1, col2 = st.columns(2)
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        age = st.number_input("Age", min_value=0, max_value=120, value=50)
        hypertension = st.selectbox("Hypertension", ["No", "Yes"])
        heart_disease = st.selectbox("Heart Disease", ["No", "Yes"])
        ever_married = st.selectbox("Ever Married", ["No", "Yes"])
    with col2:
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
        avg_glucose_level = st.number_input("Average Glucose Level", min_value=0.0, max_value=500.0, value=100.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
        smoking_status = st.selectbox("Smoking Status", ["formerly smoked", "never smoked", "smokes", "Unknown"])
    
    submitted = st.form_submit_button("Predict Stroke Risk")

# Process input and predict
if submitted:
    st.subheader("Prediction Results")
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        'gender': [gender],
        'age': [age],
        'hypertension': [1 if hypertension == "Yes" else 0],
        'heart_disease': [1 if heart_disease == "Yes" else 0],
        'ever_married': [ever_married],
        'work_type': [work_type],
        'Residence_type': [residence_type],
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'smoking_status': [smoking_status]
    })

    # Handle outliers
    def handle_outliers(df, columns, method='clip'):
        df_clean = df.copy()
        for col in columns:
            if col in df_clean.columns and df_clean[col].dtype in ['float64', 'int64']:
                Q1 = df_clean[col].quantile(0.25) if not df_clean[col].isna().all() else df_clean[col].median()
                Q3 = df_clean[col].quantile(0.75) if not df_clean[col].isna().all() else df_clean[col].median()
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                if method == 'clip':
                    df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        return df_clean

    input_data = handle_outliers(input_data, ['age', 'avg_glucose_level', 'bmi'], method='clip')

    # Feature engineering
    input_data['age_glucose'] = input_data['age'] * input_data['avg_glucose_level']
    input_data['bmi_glucose'] = input_data['bmi'] * input_data['avg_glucose_level']
    input_data['age_bmi'] = input_data['age'] * input_data['bmi']
    input_data['glucose_bmi_ratio'] = input_data['avg_glucose_level'] / (input_data['bmi'] + 1)
    input_data['age_bin'] = pd.cut(input_data['age'], bins=[0, 30, 50, 70, 100], labels=['young', 'middle', 'senior', 'elderly'])
    input_data['bmi_category'] = pd.cut(input_data['bmi'], bins=[0, 18.5, 25, 30, 100], labels=['underweight', 'normal', 'overweight', 'obese'])
    input_data['glucose_category'] = pd.cut(input_data['avg_glucose_level'], bins=[0, 100, 125, 200, 1000], labels=['normal', 'prediabetic', 'diabetic', 'severe'])
    input_data['risk_score'] = (
        input_data['age'] * 0.3 +
        input_data['avg_glucose_level'] * 0.2 +
        input_data['bmi'] * 0.1 +
        input_data['hypertension'] * 0.2 +
        input_data['heart_disease'] * 0.2
    )
    input_data['age_risk'] = input_data['age'] * input_data['risk_score']
    input_data['glucose_risk'] = input_data['avg_glucose_level'] * input_data['risk_score']
    input_data['bmi_risk'] = input_data['bmi'] * input_data['risk_score']
    input_data['age_squared'] = input_data['age'] ** 2
    input_data['glucose_squared'] = input_data['avg_glucose_level'] ** 2
    input_data['bmi_squared'] = input_data['bmi'] ** 2

    # Make prediction
    prob = pipeline.predict_proba(input_data)[:, 1][0]
    pred = 1 if prob >= optimal_threshold else 0

    # Display results
    st.write(f"**Prediction**: {'Stroke' if pred >= 0. else 'No Stroke'}")
    st.write(f"**Stroke Probability**: {prob:.3f}")
    st.write(f"**Risk Score**: {input_data['risk_score'].iloc[0]:.2f}")

    # Visualize risk score components
    st.subheader("Risk Score Breakdown")
    risk_components = {
        'Age': input_data['age'].iloc[0] * 0.3,
        'Glucose': input_data['avg_glucose_level'].iloc[0] * 0.2,
        'BMI': input_data['bmi'].iloc[0] * 0.1,
        'Hypertension': input_data['hypertension'].iloc[0] * 0.2,
        'Heart Disease': input_data['heart_disease'].iloc[0] * 0.2
    }
    fig, ax = plt.subplots()
    ax.bar(risk_components.keys(), risk_components.values(), color='skyblue')
    ax.set_ylabel('Contribution to Risk Score')
    ax.set_title('Risk Score Components')
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | Model based on Logistic Regression with SMOTE | Data: stroke-data.csv")