German Credit Risk Prediction
This project uses Machine Learning to predict whether a loan applicant is a "Good" or "Bad" credit risk. It includes a full data science pipeline: Exploratory Data Analysis (EDA), Feature Engineering, Multi-Model Comparison, and a live web application built with Streamlit.

Project Overview
The goal of this project is to minimize financial loss for banks by identifying high-risk applicants. We performed extensive "Risk Checks" to see how features like Age, Credit Amount, and Housing correlate with defaults.
Key Insights from EDA:
Monthly Burden: Applicants with a higher ratio of credit amount to loan duration (Monthly Installments) showed a significantly higher probability of being labeled "Bad Risk."

Skewness: Credit amounts were heavily right-skewed, requiring a Log Transformation to improve model performance.

Housing Stability: Applicants who "Own" their homes generally represent a lower risk profile compared to those in "Rent" or "Free" housing.

Performance ComparisonWe tested four different classifiers using GridSearchCV to find the optimal hyperparameters.ModelAccuracyExtra Trees Classifier72.50%XGBoost71.50%Random Forest70.00%Decision Tree67.00%