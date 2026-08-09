📋 Project Overview
This is a web-based application for predicting 3-year all-cause mortality risk in patients with Cardiovascular-Kidney-Metabolic (CKM) syndrome. The application is built using the NHANES (National Health and Nutrition Examination Survey) database and employs a machine learning model (Random Forest) to provide personalized risk assessments.

🎯 Key Features
Risk Prediction: Predicts 3-year mortality probability based on 13 clinical features

Interactive Interface: User-friendly web interface built with Streamlit

Model Interpretability: SHAP (SHapley Additive exPlanations) values to explain prediction results

Real-time Feedback: Instant risk assessment with visual indicators

Feature Importance: Displays global feature importance for model transparency

📊 Input Features
Categorical Variables:

Gender (Male/Female)

Cancer History (Yes/No)

CKM Stage (Stage 1-4)

Continuous Variables:

Age (years)

MCV (Mean Corpuscular Volume, fL)

RDW (Red Cell Distribution Width, %)

PLT (Platelet Count, ×10⁹/L)

Albumin (g/L)

Globulin (g/L)

AST (Aspartate Aminotransferase, U/L)

CRP (C-Reactive Protein, mg/L)

ABSI (A Body Shape Index)

SII (Systemic Immune-Inflammation Index, ×10⁹/L)

🚀 Model Performance
Algorithm: Random Forest Classifier

External Validation AUC: 0.835

Balanced Accuracy: 76.6%

Features: 13 clinical parameters
