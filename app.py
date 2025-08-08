import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for reference visuals (optional)
try:
    data = pd.read_csv("loan_data.csv")  # must have same structure as training data
except FileNotFoundError:
    data = None

# App title
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk and view related visuals.")

# Input fields
col1, col2 = st.columns(2)
with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0)
with col2:
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
    loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
    loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0)
    credit_history = st.selectbox("Credit History", ["Good (1)", "Bad (0)"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Convert inputs to numeric codes
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# Create input array
input_data = np.array([[gender, married, dependents, education,
                        self_employed, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history, property_area]])

# Scale numeric features
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    if prediction[0] == 1:
        st.error("❌ High Risk: Loan Likely to Default.")
    else:
        st.success("✅ Low Risk: Loan Likely to be Approved.")

    # Show probability if available
    if prob is not None:
        st.write(f"**Probability of Default:** {prob*100:.2f}%")

    st.markdown("---")
    st.subheader("📊 Visual Insights")

    colA, colB = st.columns(2)

    # 1. Feature Importance
    with colA:
        if hasattr(model, "feature_importances_"):
            importance = pd.DataFrame({
                'Feature': ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                            'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanAmountTerm',
                            'CreditHistory', 'PropertyArea'],
                'Importance': model.feature_importances_
            }).sort_values(by="Importance", ascending=False)
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.barplot(data=importance, x="Importance", y="Feature", ax=ax, palette="viridis")
            ax.set_title("Feature Importance")
            st.pyplot(fig)

    # 2. Applicant Income vs Loan Amount
    with colB:
        if data is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(data=data, x="ApplicantIncome", y="LoanAmount", alpha=0.6)
            ax.scatter(applicant_income, loan_amount, color="red", s=100, label="You")
            ax.legend()
            ax.set_title("Applicant Income vs Loan Amount")
            st.pyplot(fig)

    colC, colD = st.columns(2)

    # 3. Loan Amount Distribution
    with colC:
        if data is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(data["LoanAmount"], bins=20, kde=True, ax=ax, color="skyblue")
            ax.axvline(loan_amount, color="red", linestyle="--", label="You")
            ax.legend()
            ax.set_title("Loan Amount Distribution")
            st.pyplot(fig)

    # 4. Prediction Probability Gauge (Horizontal Bar)
    with colD:
        if prob is not None:
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(["Default Probability"], [prob], color="red" if prob > 0.5 else "green")
            ax.set_xlim(0, 1)
            for i, v in enumerate([prob]):
                ax.text(v + 0.02, i, f"{v*100:.1f}%", color='black', va='center')
            ax.set_title("Prediction Probability")
            st.pyplot(fig)
