# 💼 AI Salary Prediction using Artificial Neural Network (ANN)

## 📌 Project Overview

This project predicts employee salaries using an **Artificial Neural Network (ANN)**. It leverages employee-related features such as job title, experience, education level, skills, industry, company size, location, and certifications to estimate salary accurately.

The project includes complete data preprocessing, ANN model development, hyperparameter tuning using Optuna, model evaluation, and deployment through a Streamlit web application.

---

## 🚀 Features

- Employee Salary Prediction
- Artificial Neural Network (ANN)
- Data Preprocessing Pipeline
- Hyperparameter Tuning with Optuna
- One-Hot Encoding for Categorical Features
- Standard Scaling for Numerical Features
- Target Variable Scaling
- Streamlit Web Application
- Model Persistence using Pickle & Keras

---

## 📂 Dataset Features

| Feature | Description |
|----------|-------------|
| Job Title | Employee designation |
| Experience Years | Total years of experience |
| Education Level | Highest qualification |
| Skills Count | Number of technical skills |
| Industry | Industry sector |
| Company Size | Small / Medium / Large |
| Location | Employee working country |
| Certifications | Number of certifications |
| Salary | Target Variable |

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Optuna
- Streamlit
- Pickle
- Matplotlib

---

## 📊 Data Preprocessing

### Numerical Features

- Experience Years
- Skills Count
- Certifications

Technique Used

- StandardScaler

### Categorical Features

- Job Title
- Education Level
- Industry
- Company Size
- Location

Technique Used

- OneHotEncoder

Pipeline Used

- ColumnTransformer

---

## 🧠 ANN Architecture

```
Input Layer

↓

Dense Layer (96 Neurons)
Activation : Tanh

↓

Batch Normalization

↓

Dropout (0.0873)

↓

Output Layer
1 Neuron
Activation : Linear
```

---

## ⚙️ Hyperparameter Tuning

Hyperparameter optimization was performed using **Optuna**.

### Best Parameters

| Hyperparameter | Value |
|---------------|-------|
| Learning Rate | 0.0001194595 |
| Hidden Layers | 1 |
| Hidden Units | 96 |
| Activation | Tanh |
| Optimizer | RMSprop |
| Batch Size | 32 |
| Dropout | 0.0873 |
| L1/L2 Regularization | 1.334e-06 |

---

## 📈 Model Performance

| Metric | Score |
|---------|-------|
| Best Validation R² Score | **0.9766** |

### Regression Metrics

- R² Score
- MAE
- RMSE
- MSE

---

## 📁 Project Structure

```
AI_Salary_Prediction/

│── app.py
│── model.keras
│── preprocessor.pkl
│── y_scaler.pkl
│── requirements.txt
│── README.md
│── salary_dataset.csv
│── AI_Salary_Prediction.ipynb
```

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/AI_Salary_Prediction.git
```

Move into the project directory

```bash
cd AI_Salary_Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Streamlit App

```bash
streamlit run app.py
```

---

## 💻 Streamlit Application

The application allows users to:

- Select Job Title
- Enter Experience
- Select Education Level
- Enter Skills Count
- Select Industry
- Select Company Size
- Select Location
- Enter Certifications
- Predict Employee Salary instantly

---

## 📦 Saved Model Files

### Model

```
model.keras
```

### Preprocessor

```
preprocessor.pkl
```

### Target Scaler

```
y_scaler.pkl
```

---

## 📈 Workflow

```
Employee Data

        │

        ▼

Data Preprocessing

        │

        ▼

Feature Encoding

        │

        ▼

Feature Scaling

        │

        ▼

ANN Model

        │

        ▼

Optuna Hyperparameter Tuning

        │

        ▼

Salary Prediction

        │

        ▼

Streamlit Deployment
```

---

## 🎯 Future Improvements

- Explainable AI using SHAP
- Docker Containerization
- CI/CD Deployment
- Cloud Deployment (AWS/Azure/GCP)
- REST API using FastAPI
- Model Monitoring
- MLflow Experiment Tracking

---

## 👨‍💻 Author

**Palli Vinodh**

📍 Hyderabad, India

### Connect with me

- LinkedIn: https://www.linkedin.com/in/palli-vinodh/
- GitHub: https://github.com/vinodhpalli

---

## ⭐ If you found this project useful, please consider giving it a Star on GitHub!
