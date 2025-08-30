# 🏦 Loan Default Prediction App  

An interactive web app built with **Streamlit** that predicts whether a loan applicant is likely to default or not. The app uses a trained **Machine Learning model** and provides **visual insights** along with a downloadable **PDF report**.  

---

## 🌟 Features  
✅ Predicts loan default risk based on applicant details  
✅ Shows probability of default in percentage  
✅ Provides interactive **visual insights** (charts & distributions)  
✅ Highlights applicant’s data point in visuals for better understanding  
✅ Generates a **PDF report** (prediction + visuals)  
✅ User-friendly interface with background styling  

---

## 🛠️ Technologies Used  
- **Python** 🐍  
- **Streamlit** (for web app)  
- **scikit-learn** (for ML model)  
- **pandas, numpy** (data processing)  
- **matplotlib, seaborn** (visualizations)  
- **joblib** (model saving/loading)  
- **reportlab** (PDF generation)  

---

## 🚀 How to Run Locally  

1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. Create a virtual environment & activate it  
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app  
   ```bash
   streamlit run app.py
   ```

5. Open in browser 👉 `http://localhost:8501`  

---

## 📊 Dataset  
The model is trained on **loan applicant data** (features like gender, income, dependents, education, loan amount, etc.).  
> You can replace `loan_data.csv` with your own dataset for retraining.  

---

## 🌍 Live Demo  
🔗 [Streamlit Deployed App](https://your-streamlit-link.streamlit.app)  

---

## 📽️ Demo Video  
🎥 [YouTube / Google Drive Demo](https://your-demo-link.com)  

---

## 📑 Project Presentation  
📂 [View Presentation Slides](https://your-slides-link.com)  

---

## 📸 Screenshots  

### Home Screen  
![Home](screenshots/home.png)  

### Prediction Result  
![Result](screenshots/result.png)  

### Visual Insights  
![Charts](screenshots/charts.png)  

---

## 👨‍💻 Author  
Developed by **[Your Name]**  
For Hackathon Submission 🚀  
