# Bank Term Deposit Prediction

An end-to-end Machine Learning project that predicts whether a customer will subscribe to a term deposit based on demographic, financial, and campaign-related data. The project includes data preprocessing, feature engineering, model training, evaluation, explainability using SHAP, and deployment via a Streamlit web application.

---

## Objective

- Predict whether a customer will subscribe to a term deposit (Yes/No)
- Improve marketing campaign efficiency
- Build and compare multiple machine learning models
- Deploy an interactive web app for real-time predictions
- Provide model interpretability using SHAP

---

## Dataset

- Source: Bank Marketing Dataset
- Contains customer and campaign-related information such as:
  - Age, job, marital status, education
  - Account balance, housing loan, personal loan
  - Campaign details (duration, contacts)
  - Previous campaign outcomes

> Dataset is not included in this repository due to privacy and system restrictions.  
> The project uses the publicly available Bank Marketing dataset.

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn
- SHAP
- Matplotlib, Seaborn
- Streamlit

---

##  Machine Learning Pipeline

1. Data Cleaning and preprocessing  
2. Feature engineering:
   - age_group  
   - balance_per_age  
   - campaign_intensity  
   - duration_balance  
3. Encoding categorical variables  
4. Train-test split  
5. Model training:
   - Logistic Regression  
   - Random Forest  
   - XGBoost  
6. Model evaluation using ROC-AUC  
7. Selection of best model  
8. Model saving using Joblib  

---

## Model Performance

- Best Model: **XGBoost**
- ROC-AUC Score: ~0.93

### Evaluation Metrics:
- Confusion Matrix  
- ROC Curve  
- Classification Report  

---

##  Explainability

SHAP (SHapley Additive Explanations) is used to interpret model predictions and understand feature importance. It helps explain why a prediction is made for a specific customer.

---

## Streamlit Application

The web app allows users to:

- Input customer details
- Get prediction probability
- View classification result
- Visualize SHAP explanations

---

##  How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/Bank_Term_Deposit_Prediction.git
cd Bank_Term_Deposit_Prediction
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit app
streamlit run app/streamlit_app.py
 Key Insights
Call duration is a strong predictor of subscription
Customers with higher balances are more likely to subscribe
Previous campaign outcomes significantly influence decisions
Excessive campaign contacts reduce effectiveness
 
 Author
Gomathi Boopathy

 Future Improvements
Deploy on cloud (AWS / Streamlit Cloud)
Add FastAPI backend
Hyperparameter tuning (Optuna)
Real-time data integration
Improve model calibration
