import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="CKM 3-Year Mortality Risk Prediction",
    page_icon="🏥",
    layout="wide"
)


# ==================== Load Model ====================
@st.cache_resource
def load_model():
    """Load trained model and preprocessors"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Try multiple possible paths
    possible_paths = [
        os.path.join(current_dir, 'ckm_risk_model.pkl'),
        r"C:\Users\admin\PycharmProjects\PythonProject9\CKM_3 year death\ckm_risk_model.pkl",
        r"C:\Users\admin\PycharmProjects\PythonProject9\ckm_risk_model.pkl"
    ]

    for model_path in possible_paths:
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as file:
                    artifacts = pickle.load(file)
                st.success(f"✅ Model loaded successfully: {os.path.basename(model_path)}")
                return artifacts
            except Exception as e:
                st.warning(f"⚠️ Failed to load {model_path}: {e}")
                continue

    st.error("❌ Model file not found")
    st.info("Please ensure 'ckm_risk_model.pkl' exists in one of the following paths:\n" + "\n".join(possible_paths))
    return None


# Load model
artifacts = load_model()

if artifacts is None:
    st.stop()

# Extract model components
model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['features']
categorical_features = artifacts['categorical_features']
continuous_features = artifacts['continuous_features']
threshold = artifacts.get('threshold', 0.5)
model_info = artifacts.get('model_info', {})

# ==================== Page Title ====================
st.title("🏥 CKM 3-Year All-Cause Mortality Risk Prediction")
st.markdown("---")

# Model information display
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model Type", model_info.get('type', 'Random Forest'))
with col2:
    st.metric("AUC", f"{model_info.get('auc', 0.835):.3f}")
with col3:
    st.metric("Balanced Accuracy", f"{model_info.get('balanced_accuracy', 0.766):.3f}")
with col4:
    st.metric("Prediction Threshold", f"{threshold:.4f}")

st.markdown("---")
st.markdown("""
This tool uses a Random Forest model to predict 3-year all-cause mortality risk in CKM 
(Cardiovascular-Kidney-Metabolic) patients. Please enter patient clinical characteristics 
in the sidebar below.
""")

# ==================== Sidebar Input ====================
st.sidebar.header("📋 Patient Characteristics Input")

# Categorical Variables
st.sidebar.subheader("📌 Categorical Variables")
gender = st.sidebar.selectbox("Gender", options=["Male", "Female"])
cancer = st.sidebar.selectbox("Cancer History", options=["No", "Yes"])
ckm = st.sidebar.selectbox("CKM Stage", options=["Stage 1", "Stage 2", "Stage 3", "Stage 4"])

# Categorical variable encoding mapping
gender_map = {"Male": 1, "Female": 0}
cancer_map = {"No": 0, "Yes": 1}
ckm_map = {"Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "Stage 4": 4}

# Continuous Variables
st.sidebar.subheader("📊 Continuous Variables")
age = st.sidebar.slider("Age (years)", min_value=18, max_value=100, value=65, step=1)
mcv = st.sidebar.slider("MCV (fL)", min_value=50.0, max_value=120.0, value=90.0, step=0.5)
rdw = st.sidebar.slider("RDW (%)", min_value=10.0, max_value=25.0, value=13.5, step=0.1)
plt_val = st.sidebar.slider("PLT (×10⁹/L)", min_value=10, max_value=800, value=200, step=5)
alb = st.sidebar.slider("Albumin (g/L)", min_value=15.0, max_value=55.0, value=40.0, step=0.5)
glb = st.sidebar.slider("Globulin (g/L)", min_value=10.0, max_value=50.0, value=25.0, step=0.5)
ast = st.sidebar.slider("AST (U/L)", min_value=1, max_value=200, value=25, step=1)
crp = st.sidebar.slider("CRP (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
absi = st.sidebar.slider("ABSI Index", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
sii = st.sidebar.slider("SII Index (×10⁹/L)", min_value=0, max_value=5000, value=500, step=50)


# ==================== Data Processing Functions ====================
def create_input_data():
    """Create input dataframe"""
    input_dict = {
        'GENDER': gender_map[gender],
        'AGE': age,
        'MCV': mcv,
        'RDW': rdw,
        'PLT': plt_val,
        'ALB': alb,
        'GLB': glb,
        'AST': ast,
        'CRP': crp,
        'CANCER': cancer_map[cancer],
        'CKM': ckm_map[ckm],
        'ABSI': absi,
        'SII': sii
    }
    return pd.DataFrame([input_dict])


def predict_risk(input_df):
    """Predict mortality risk"""
    # Separate continuous and categorical variables
    cont_df = input_df[continuous_features]
    cat_df = input_df[categorical_features]

    # Standardize continuous variables
    cont_scaled = scaler.transform(cont_df)

    # Combine features
    X_scaled = np.hstack([cont_scaled, cat_df.values])

    # Predict probability
    prob = model.predict_proba(X_scaled)[0, 1]

    # Use optimal threshold for prediction
    prediction = 1 if prob >= threshold else 0

    return prob, prediction, X_scaled


# ==================== Main Layout ====================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Input Features Summary")
    input_data = create_input_data()

    # Display input features (decoded)
    display_data = input_data.copy()
    display_data['GENDER'] = display_data['GENDER'].map({1: 'Male', 0: 'Female'})
    display_data['CANCER'] = display_data['CANCER'].map({1: 'Yes', 0: 'No'})
    display_data['CKM'] = display_data['CKM'].map({1: 'Stage 1', 2: 'Stage 2', 3: 'Stage 3', 4: 'Stage 4'})

    # Transpose and display
    display_df = display_data.T.rename(columns={0: 'Value'})
    display_df['Value'] = display_df['Value'].astype(str)
    st.dataframe(display_df, width='stretch')

with col2:
    st.subheader("🎯 Prediction Results")

    if st.button("🔍 Predict", type="primary", use_container_width=True):
        prob, pred, X_scaled = predict_risk(input_data)

        # Display risk probability
        risk_color = '#DC3545' if prob > 0.5 else '#28A745'
        st.markdown(f"""
        <div style="text-align: center; padding: 10px; background-color: {risk_color}; 
                    border-radius: 10px; color: white;">
            <h1 style="margin: 0; color: white;">{prob * 100:.1f}%</h1>
            <p style="margin: 0; color: white;">3-Year Mortality Risk</p>
        </div>
        """, unsafe_allow_html=True)

        # Display prediction result
        if pred == 1:
            st.error("⚠️ HIGH RISK: Predicted 3-year mortality risk is elevated")
        else:
            st.success("✅ LOW RISK: Predicted 3-year mortality risk is low")

        # Risk level
        if prob < 0.2:
            risk_level = "Low Risk"
            color = "#28A745"
        elif prob < 0.5:
            risk_level = "Moderate Risk"
            color = "#FFC107"
        else:
            risk_level = "High Risk"
            color = "#DC3545"

        # Custom progress bar
        st.markdown(f"""
        <div style="margin: 10px 0;">
            <div style="background-color: #e9ecef; border-radius: 10px; height: 30px; position: relative;">
                <div style="background-color: {color}; width: {min(prob * 100, 100)}%; border-radius: 10px; height: 30px; 
                            display: flex; align-items: center; justify-content: center; color: white; font-weight: bold;">
                    {risk_level} ({prob * 100:.1f}%)
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.caption(f"Prediction Threshold: {threshold:.3f}")

        # ==================== SHAP Explanation ====================
        st.markdown("---")
        st.subheader("🔍 Model Explanation (SHAP)")

        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)

            # For classification models, take positive class SHAP values
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Feature names
            feature_names = continuous_features + categorical_features

            # Create feature contribution bar chart
            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'SHAP Value': shap_values[0]
            }).sort_values('SHAP Value', ascending=True)

            # Plot SHAP bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            colors_shap = ['#DC3545' if x < 0 else '#007BFF' for x in shap_df['SHAP Value']]
            ax.barh(shap_df['Feature'], shap_df['SHAP Value'], color=colors_shap)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('SHAP Value')
            ax.set_title('Feature Impact on Prediction')

            # Add legend
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor='#DC3545', label='Increases Risk ↑'),
                Patch(facecolor='#007BFF', label='Decreases Risk ↓')
            ]
            ax.legend(handles=legend_elements, loc='lower right')

            st.pyplot(fig)

            # Display detailed SHAP table
            with st.expander("📋 View Detailed SHAP Values"):
                shap_detail = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values[0],
                    'Direction': ['↑ Increases Risk' if x > 0 else '↓ Decreases Risk' for x in shap_values[0]],
                    '|SHAP Value|': np.abs(shap_values[0])
                }).sort_values('|SHAP Value|', ascending=False)
                st.dataframe(shap_detail, width='stretch')

        except Exception as e:
            st.warning(f"⚠️ SHAP explanation generation failed: {e}")

# ==================== Feature Importance ====================
st.markdown("---")
st.subheader("📈 Global Feature Importance")

try:
    # Get feature importance
    importance = model.feature_importances_
    feature_names = continuous_features + categorical_features

    # Create DataFrame and sort
    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    # Plot horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_imp = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Global Feature Importance')
    st.pyplot(fig)

    # Display detailed table
    with st.expander("📋 View Feature Importance Details"):
        st.dataframe(imp_df.sort_values('Importance', ascending=False), width='stretch')

except Exception as e:
    st.warning(f"⚠️ Unable to display feature importance: {e}")

# ==================== User Guide ====================
st.markdown("---")
st.subheader("📖 User Guide")

col_help1, col_help2, col_help3 = st.columns(3)

with col_help1:
    st.markdown("""
    **📝 Input Steps**
    1. Enter patient characteristics in sidebar
    2. Categorical variables: Select options
    3. Continuous variables: Adjust sliders
    4. Click "Predict" button
    """)

with col_help2:
    st.markdown("""
    **📊 Result Interpretation**
    - **Risk Probability**: 3-year mortality probability
    - **Risk Level**: Low/Moderate/High
    - **SHAP Values**: Feature impact direction
    - **Red**: Increases risk
    - **Blue**: Decreases risk
    """)

with col_help3:
    st.markdown("""
    **💡 Model Information**
    - Algorithm: Random Forest
    - Features: 13 clinical variables
    - External Validation AUC: 0.835
    - Balanced Accuracy: 76.6%
    - For clinical research reference only
    """)

# ==================== Footer ====================
st.markdown("---")
st.caption(
    "⚠️ This tool is for clinical research reference only, not for final diagnosis | Model Version: v1.0 | Best Random Forest Model")

# ==================== Run Instructions ====================
if __name__ == "__main__":
    # Run command: streamlit run ckm_3years_death.py
    pass