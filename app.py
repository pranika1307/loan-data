import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter

# ---------- Background with overlay ----------
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white;
        }}
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.5);
            z-index: 0;
        }}
        h1, h2, h3, h4, h5, h6, p, label, span {{
            color: white !important;
        }}
        </style>
    """, unsafe_allow_html=True)

# ---------- Simple dark theme for plots ----------
def dark_plot():
    plt.style.use("dark_background")

# ---------- Page setup ----------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
set_background("bank1.jpg")
dark_plot()

# ---------- Load model & data ----------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
try:
    data = pd.read_csv("loan_data.csv")
except:
    data = None

# ---------- Title ----------
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk and view related visuals.")

# ---------- Inputs ----------
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

# ---------- Convert inputs ----------
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# ---------- Model prediction ----------
input_data = np.array([[gender, married, dependents, education,
                        self_employed, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history, property_area]])
input_scaled = scaler.transform(input_data)

if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    result_text = "✅ Low Risk: Loan Likely to be Approved." if prediction[0] == 0 else "❌ High Risk: Loan Likely to Default."
    st.success(result_text) if prediction[0] == 0 else st.error(result_text)

    prob_text = f"Probability of Default: {prob*100:.2f}%" if prob is not None else "Probability not available."
    st.write(f"**{prob_text}**")

    st.markdown("---")
    st.subheader("📊 Visual Insights")

    plots = []  # store plots for PDF

    # --- Plot 1: Feature Importance ---
    if hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots()
        importance = pd.DataFrame({
            'Feature': ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                        'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'LoanAmountTerm',
                        'CreditHistory', 'PropertyArea'],
            'Importance': model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        sns.barplot(data=importance, x="Importance", y="Feature", ax=ax, palette="viridis")
        ax.set_title("Feature Importance")
        st.pyplot(fig)
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        plots.append(("Feature Importance", buf))

    # --- Plot 2: Applicant Income vs Loan Amount ---
    if data is not None:
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x="ApplicantIncome", y="LoanAmount", alpha=0.6, ax=ax)
        ax.scatter(applicant_income, loan_amount, color="red", s=100, label="You")
        ax.legend()
        ax.set_title("Applicant Income vs Loan Amount")
        st.pyplot(fig)
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        plots.append(("Applicant Income vs Loan Amount", buf))

    # --- Plot 3: Loan Amount Distribution ---
    if data is not None:
        fig, ax = plt.subplots()
        sns.histplot(data["LoanAmount"], bins=20, kde=True, ax=ax, color="skyblue")
        ax.axvline(loan_amount, color="red", linestyle="--", label="You")
        ax.legend()
        ax.set_title("Loan Amount Distribution")
        st.pyplot(fig)
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        plots.append(("Loan Amount Distribution", buf))

    # --- Plot 4: Prediction Probability ---
    if prob is not None:
        fig, ax = plt.subplots(figsize=(5, 1.5))
        ax.barh(["Default Probability"], [prob], color="red" if prob > 0.5 else "green")
        ax.set_xlim(0, 1)
        ax.text(prob + 0.02, 0, f"{prob*100:.1f}%", color='white', va='center')
        ax.set_title("Prediction Probability")
        st.pyplot(fig)
        buf = BytesIO(); fig.savefig(buf, format="png"); buf.seek(0)
        plots.append(("Prediction Probability", buf))

    # ---------- PDF Export ----------
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Loan Default Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(result_text, styles["Normal"]))
    elements.append(Paragraph(prob_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    details = f"""
    Gender: {gender}, Married: {married}, Dependents: {dependents}, Education: {education},
    Self Employed: {self_employed}, Applicant Income: {applicant_income},
    Coapplicant Income: {coapplicant_income}, Loan Amount: {loan_amount},
    Loan Term: {loan_amount_term}, Credit History: {credit_history},
    Property Area: {property_area}
    """
    elements.append(Paragraph("Applicant Details:", styles["Heading2"]))
    elements.append(Paragraph(details, styles["Normal"]))
    elements.append(Spacer(1, 12))

    for title, img_buf in plots:
        elements.append(Paragraph(title, styles["Heading2"]))
        elements.append(Image(img_buf, width=400, height=250))
        elements.append(Spacer(1, 12))

    doc.build(elements)
    pdf_buffer.seek(0)

    st.download_button(
        label="📥 Download Prediction Report (PDF)",
        data=pdf_buffer,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )
