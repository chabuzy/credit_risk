import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- LOAD ASSETS ---
model = joblib.load("models/best_credit_model.pkl")
feat_importance = joblib.load("models/feature_importance.pkl")
cat_cols = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
encoders = {col: joblib.load(f"models/{col}_encoder.pkl") for col in cat_cols}

st.set_page_config(page_title="Credit Risk AI", layout="wide")

# --- SIDEBAR / NAVIGATION ---
tab1, tab2 = st.tabs(["🚀 Make a Prediction", "📊 Data Insights"])

# --- TAB 1: PREDICTION ---
with tab1:
    st.title("🏦 Credit Risk Prediction")
    st.write("This app uses an **Extra Trees Classifier** (72.5% Accuracy) to predict credit default risk.")

    # --- USER INPUTS (Now correctly inside Tab 1) ---
    ui_col1, ui_col2 = st.columns(2)

    with ui_col1:
        age = st.number_input("Age", 18, 100, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        housing = st.selectbox("Housing", ["own", "rent", "free"])
        saving = st.selectbox("Saving accounts", ["little", "moderate", "rich", "quite rich", "unknown"])

    with ui_col2:
        duration = st.number_input("Duration (months)", 1, 72, 12)
        credit_amt = st.number_input("Credit Amount ($)", 100, 20000, 1000)
        checking = st.selectbox("Checking account", ["little", "moderate", "rich", "unknown"])
        purpose = st.selectbox("Purpose", list(encoders['Purpose'].classes_))

    # --- FEATURE ENGINEERING ---
    credit_log = np.log(credit_amt)
    monthly_burden = credit_amt / duration

    # --- PREDICTION LOGIC ---
    if st.button("Analyze Risk"):
        input_df = pd.DataFrame({
            "Age": [age],
            "Sex": [encoders["Sex"].transform([sex])[0]],
            "Job": [1], 
            "Housing": [encoders["Housing"].transform([housing])[0]],
            "Saving accounts": [encoders["Saving accounts"].transform([saving])[0]],
            "Checking account": [encoders["Checking account"].transform([checking])[0]],
            "Purpose": [encoders["Purpose"].transform([purpose])[0]],
            "Duration": [duration],
            "Monthly_Installment": [monthly_burden], # Make sure this matches your X_train name!
            "Credit amount_log": [credit_log]
        })
        
        # Match column order
        try:
            input_df = input_df[model.feature_names_in_]
        except:
            pass
        # --- DEBUG CHECK ---
        st.write("### Debugging Column Match")
        model_columns = list(model.feature_names_in_)
        app_columns = list(input_df.columns)

        if model_columns == app_columns:
         st.success("✅ Column names and order match perfectly!")
        else:
         st.error("❌ Mismatch detected!")
         st.write("Model expected:", model_columns)
         st.write("App sent:", app_columns)
        prediction = model.predict(input_df)[0]
        
        st.divider()
        if prediction == 1:
            st.success("✅ **Prediction: GOOD RISK** - Likely to repay.")
        else:
            st.error("❌ **Prediction: BAD RISK** - High probability of default.")

# --- TAB 2: DATA INSIGHTS ---
with tab2:
    st.title("🔍 Strategic Model & Data Insights")
    st.write("Below are the key findings from our SQL-driven analysis of the German Credit Dataset.")

    # Section 1: Feature Importance & Business Logic
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Key Drivers of Risk")
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=feat_importance, palette='viridis', ax=ax)
        st.pyplot(fig)
    
    with col_b:
        st.subheader("Why these features?")
        st.info("""
        - **Checking Account:** The strongest predictor of immediate liquidity.
        - **Monthly_Installment:** (New Feature) High monthly burdens increase default probability.
        - **Age:** Older applicants often show more financial stability.
        """)

    st.divider()

    # Section 2: Strategic Insights (The SQL Findings)
    st.subheader("⚠️ High-Risk Segments & Generational Trends")
    col_c, col_d = st.columns(2)

    with col_c:
        # Display the Strategic Insight Image 1
        st.image("images/strategic_insight_1.png", caption="High-Risk Housing & Job Segments")
        st.write("**Finding:** Renters with skilled jobs show a default rate > 40%.")

    with col_d:
        # Display the Strategic Insight Image 2
        st.image("images/strategic_insight_2.png", caption="Default Risk by Age Group")
        st.write("**Finding:** Gen Z / Young Adults are significantly higher risk than Seniors.")

    # Section 3: Feature Engineering Explanation
    st.divider()
    with st.expander("🛠 View Feature Engineering Details"):
        st.write("""
        We improved model accuracy by engineering two main features:
        1. **Log Transformation:** Normalizing 'Credit Amount' to handle outliers.
        2. **Monthly Installment:** Calculated as `Credit Amount / Duration` to measure the borrower's monthly pressure.
        """)