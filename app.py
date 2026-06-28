import streamlit as st
import pandas as pd
import numpy as np
import pickle

from tensorflow.keras.models import load_model

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------

st.set_page_config(
    page_title="AI Salary Prediction",
    page_icon="💼",
    layout="wide"
)

# -------------------------------------------------
# Load Files
# -------------------------------------------------

@st.cache_resource
def load_files():

    model = load_model("model.keras")

    with open("preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)

    with open("y_scaler.pkl", "rb") as f:
        y_scaler = pickle.load(f)

    return model, preprocessor, y_scaler


model, preprocessor, y_scaler = load_files()

# -------------------------------------------------
# Title
# -------------------------------------------------

st.title("💼 AI Salary Prediction using ANN")

st.markdown(
"""
Predict employee salary using an Artificial Neural Network.

Fill all employee details and click **Predict Salary**.
"""
)

st.divider()

# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------

st.sidebar.header("Employee Details")

job_title = st.sidebar.selectbox(
    "Job Title",
    [
        "AI Engineer",
        "Data Scientist",
        "Data Analyst",
        "ML Engineer",
        "Business Analyst",
        "Software Engineer",
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "DevOps Engineer",
        "Product Manager"
    ]
)

experience_years = st.sidebar.slider(
    "Experience (Years)",
    0,
    30,
    5
)

education_level = st.sidebar.selectbox(
    "Education Level",
    [
        "High School",
        "Bachelor",
        "Master",
        "PhD"
    ]
)

skills_count = st.sidebar.slider(
    "Skills Count",
    1,
    30,
    10
)

industry = st.sidebar.selectbox(
    "Industry",
    [
        "Healthcare",
        "Finance",
        "Retail",
        "Manufacturing",
        "Telecom",
        "IT",
        "Education",
        "Media"
    ]
)

company_size = st.sidebar.selectbox(
    "Company Size",
    [
        "Small",
        "Medium",
        "Large"
    ]
)

location = st.sidebar.selectbox(
    "Location",
    [
        "India",
        "USA",
        "Canada",
        "Germany",
        "Australia",
        "Singapore",
        "Sweden",
        "UK"
    ]
)

certifications = st.sidebar.slider(
    "Certifications",
    0,
    15,
    2
)

# -------------------------------------------------
# Input Data
# -------------------------------------------------

input_df = pd.DataFrame({

    "job_title":[job_title],

    "experience_years":[experience_years],

    "education_level":[education_level],

    "skills_count":[skills_count],

    "industry":[industry],

    "company_size":[company_size],

    "location":[location],

    "certifications":[certifications]

})

# -------------------------------------------------
# Prediction
# -------------------------------------------------

if st.button("🚀 Predict Salary", use_container_width=True):

    try:

        X = preprocessor.transform(input_df)

        pred_scaled = model.predict(X)

        prediction = y_scaler.inverse_transform(pred_scaled)

        salary = prediction[0][0]

        st.success(
            f"### 💰 Predicted Salary : ₹ {salary:,.2f}"
        )

        st.balloons()

        st.subheader("Employee Details")

        st.dataframe(
            input_df,
            use_container_width=True
        )

    except Exception as e:

        st.error(f"Prediction Error : {e}")

# -------------------------------------------------
# Footer
# -------------------------------------------------

st.markdown("---")

st.markdown(
"""
### Project Information

**Model**
- Artificial Neural Network (ANN)

**Preprocessing**
- StandardScaler
- OneHotEncoder
- ColumnTransformer

**Hyperparameter Tuning**
- Optuna

**Evaluation Metric**
- R² Score
"""
)