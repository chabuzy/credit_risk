import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- LOAD ASSETS ---
# Loading from the 'models/' folder as we planned
model = joblib.load("models/best_credit_model.pkl")
cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
encoders = {col: joblib.load(f"models/{col}_encoder.pkl") for col in cat_cols}

st.set_page_config(page_title="Credit Risk AI", layout="centered")
st.title("🏦 Credit Risk Prediction App")
st.write("This app uses an **Extra Trees Classifier** (72.5% Accuracy) to predict credit default risk.")

# --- USER INPUTS ---
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 100, 30)
    sex = st.selectbox("Sex", ["male", "female"])
    housing = st.selectbox("Housing", ["own", "rent", "free"])
    saving = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich", "unknown"])

with col2:
    duration = st.number_input("Duration (months)", 1, 72, 12)
    credit_amt = st.number_input("Credit Amount ($)", 100, 20000, 1000)
    checking = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
    purpose = st.selectbox("Purpose", list(encoders['Purpose'].classes_))

# --- FEATURE ENGINEERING (Must match Notebook) ---
# 1. Log Transform
credit_log = np.log(credit_amt)
# 2. Monthly Installment
monthly_burden = credit_amt / duration

# --- PREDICTION LOGIC ---
if st.button("Analyze Risk"):
    # Ensure these names match the X.columns from your Notebook EXACTLY
    input_df = pd.DataFrame({
        "Age": [age],
        "Sex": [encoders["Sex"].transform([sex])[0]],
        "Job": [1], 
        "Housing": [encoders["Housing"].transform([housing])[0]],
        "Saving accounts": [encoders["Saving accounts"].transform([saving])[0]],
        "Checking account": [encoders["Checking account"].transform([checking])[0]],
        "Purpose": [encoders["Purpose"].transform([purpose])[0]],
        "Duration": [duration], # Added because it was likely in your X_train
        "Monthly_Installment": [monthly_burden],
        "Credit amount_log": [credit_log]
    })
    
    # Reorder columns to match the training set order
    # model.feature_names_in_ exists if using newer scikit-learn
    try:
        input_df = input_df[model.feature_names_in_]
    except:
        pass

    prediction = model.predict(input_df)[0]
    
    if prediction == 1:
        st.success("✅ **Prediction: GOOD RISK** - Likely to repay.")
    else:
        st.error("❌ **Prediction: BAD RISK** - High probability of default.")