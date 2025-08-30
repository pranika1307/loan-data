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
   git clone https://github.com/pranika1307/loan-default-prediction.git
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
🔗 [Streamlit Deployed App]
(loan-data-56npspdvjxnubrbggwt9ac.streamlit.app)  

---

## 📽️ Demo Video  
🎥 [YouTube / Google Drive Demo]
(https://drive.google.com/file/d/1D3YdXS_VtP7OROo9TCmp3tv6Xz1sY09d/view?usp=sharing)  

---

## 📑 Project Presentation  
📂 [View Presentation Slides]
(https://docs.google.com/presentation/d/1wBi68XpP6KO9WmZN0kYgKvH_g4guwxvS/edit?usp=sharing&ouid=105611852932502833748&rtpof=true&sd=true)  

---

## 📸 Screenshots  

- **Home Page**
  ![Home Page](screenshot1.png)

- **Prediction Result**
  ![Prediction Result](screenshot2.png)

- **Visual Insights**
  ![Visual Insights](screenshot3.png)

- **PDF Report Download**
  ![PDF Report Download](screenshot4.png)


---

## 💡 Future Enhancements

-Deploy on cloud (AWS/GCP) for scalability
-Add more features for prediction (employment type, assets, etc.)
-Improve model accuracy with hyperparameter tuning
-Mobile-friendly UI


## 👨‍💻 Author  
Developed by Pranika Seth 
For Hackathon Submission 🚀  
