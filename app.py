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

# ==== Function to set background with overlay ====
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: white !important;
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
    main, header, footer {{
        position: relative;
        z-index: 1;
    }}
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: white !important;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# ==== Dark theme helper for plots ====
def apply_dark_theme(fig, ax):
    fig.patch.set_facecolor('#0e1117')     # Dark outer background
    ax.set_facecolor('#0e1117')            # Dark plot area
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color('white')
    return fig, ax

# ==== Page config & background ====
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
set_background("bank1.jpg")

# ==== Load model and scaler ====
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Load dataset for visuals if available
try:
    data = pd.read_csv("loan_data.csv")
except FileNotFoundError:
    data = None

# ==== Title ====
st.title("🏦 Loan Default Prediction App")
st.write("Enter applicant details to predict loan default risk and view related visuals.")

# ==== Input layout ====
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

# ==== Convert inputs ====
gender = 1 if gender == "Male" else 0
married = 1 if married == "Yes" else 0
dependents = 3 if dependents == "3+" else int(dependents)
education = 0 if education == "Graduate" else 1
self_employed = 1 if self_employed == "Yes" else 0
credit_history = 1 if credit_history == "Good (1)" else 0
property_area = {"Urban": 2, "Semiurban": 1, "Rural": 0}[property_area]

# ==== Prepare for model ====
input_data = np.array([[gender, married, dependents, education,
                        self_employed, applicant_income, coapplicant_income,
                        loan_amount, loan_amount_term, credit_history, property_area]])
input_scaled = scaler.transform(input_data)

# ==== Prediction and visuals ====
if st.button("Predict Loan Default"):
    prediction = model.predict(input_scaled)
    prob = model.predict_proba(input_scaled)[0][1] if hasattr(model, "predict_proba") else None

    if prediction[0] == 1:
        result_text = "❌ High Risk: Loan Likely to Default."
        st.error(result_text)
    else:
        result_text = "✅ Low Risk: Loan Likely to be Approved."
        st.success(result_text)

    prob_text = f"Probability of Default: {prob*100:.2f}%" if prob is not None else "Probability not available."
    st.write(f"**{prob_text}**")

    st.markdown("---")
    st.subheader("📊 Visual Insights")

    plot_images = []
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
            ax.set_title("Feature Importance", color="white")
            fig, ax = apply_dark_theme(fig, ax)
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
            buf.seek(0)
            plot_images.append(("Feature Importance", buf))

    # 2. Applicant Income vs Loan Amount
    with colB:
        if data is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.scatterplot(data=data, x="ApplicantIncome", y="LoanAmount", alpha=0.6, ax=ax)
            ax.scatter(applicant_income, loan_amount, color="red", s=100, label="You")
            ax.legend()
            ax.set_title("Applicant Income vs Loan Amount", color="white")
            fig, ax = apply_dark_theme(fig, ax)
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
            buf.seek(0)
            plot_images.append(("Applicant Income vs Loan Amount", buf))

    colC, colD = st.columns(2)
    # 3. Loan Amount Distribution
    with colC:
        if data is not None:
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.histplot(data["LoanAmount"], bins=20, kde=True, ax=ax, color="skyblue")
            ax.axvline(loan_amount, color="red", linestyle="--", label="You")
            ax.legend()
            ax.set_title("Loan Amount Distribution", color="white")
            fig, ax = apply_dark_theme(fig, ax)
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
            buf.seek(0)
            plot_images.append(("Loan Amount Distribution", buf))

    # 4. Prediction Probability Gauge
    with colD:
        if prob is not None:
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.barh(["Default Probability"], [prob], color="red" if prob > 0.5 else "green")
            ax.set_xlim(0, 1)
            ax.text(prob + 0.02, 0, f"{prob*100:.1f}%", color='white', va='center')
            ax.set_title("Prediction Probability", color="white")
            fig, ax = apply_dark_theme(fig, ax)
            st.pyplot(fig)
            buf = BytesIO()
            fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
            buf.seek(0)
            plot_images.append(("Prediction Probability", buf))

    # ==== Generate PDF ====
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    elements.append(Paragraph("Loan Default Prediction Report", styles["Title"]))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Result: {result_text}", styles["Normal"]))
    elements.append(Paragraph(prob_text, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Applicant details
    details = f"""
    Gender: {gender} | Married: {married} | Dependents: {dependents} | Education: {education} |
    Self Employed: {self_employed} | Applicant Income: {applicant_income} |
    Coapplicant Income: {coapplicant_income} | Loan Amount: {loan_amount} |
    Loan Term: {loan_amount_term} | Credit History: {credit_history} |
    Property Area: {property_area}
    """
    elements.append(Paragraph("Applicant Details:", styles["Heading2"]))
    elements.append(Paragraph(details, styles["Normal"]))
    elements.append(Spacer(1, 12))

    # Add plots to PDF
    for title, img_buf in plot_images:
        elements.append(Paragraph(title, styles["Heading2"]))
        img = Image(img_buf, width=400, height=250)
        elements.append(img)
        elements.append(Spacer(1, 12))

    doc.build(elements)
    pdf_buffer.seek(0)

    # Download button
    st.download_button(
        label="📥 Download Prediction Report (PDF)",
        data=pdf_buffer,
        file_name="loan_prediction_report.pdf",
        mime="application/pdf"
    )
