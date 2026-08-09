import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="CKM 3-Year Mortality Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    .main-header {
        font-size: 2.0rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #6c757d;
        margin-bottom: 1.2rem;
    }
    .divider {
        border-top: 1px solid #e9ecef;
        margin: 1.2rem 0;
    }

    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.5rem 0.8rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-box .label {
        font-size: 0.65rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .metric-box .value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
    }

    .input-card {
        background-color: white;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        border: 1px solid #e9ecef;
        margin-bottom: 0.8rem;
    }
    .input-card-title {
        font-size: 0.75rem;
        font-weight: 600;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.8rem;
    }

    .result-card {
        border-radius: 12px;
        padding: 1.8rem 1.5rem;
        text-align: center;
        margin-bottom: 0.8rem;
        border: 2px solid #e9ecef;
        background: linear-gradient(135deg, #fafbfc, #ffffff);
    }
    .result-number {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1a1a2e;
        line-height: 1.2;
    }
    .result-number .percent {
        font-size: 1.8rem;
        color: #6c757d;
    }
    .result-label {
        font-size: 0.95rem;
        color: #6c757d;
        margin-top: 0.2rem;
    }
    .result-outcome {
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 0.5rem;
        padding: 0.4rem 1.5rem;
        border-radius: 8px;
        display: inline-block;
    }
    .outcome-death {
        color: #dc3545;
        background-color: #f8d7da;
    }
    .outcome-survive {
        color: #28a745;
        background-color: #d4edda;
    }
    .result-prob-detail {
        margin-top: 0.4rem;
        font-size: 0.8rem;
        color: #6c757d;
    }

    .probability-bar {
        margin-top: 0.8rem;
        background-color: #e9ecef;
        border-radius: 20px;
        height: 12px;
        overflow: hidden;
        position: relative;
    }
    .probability-bar .death-bar {
        background: linear-gradient(90deg, #dc3545, #b02a37);
        height: 100%;
        border-radius: 20px;
        transition: width 0.5s ease;
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4a6cf7, #6a3de8);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.6rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.4);
    }

    .placeholder {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 2.5rem 1.5rem;
        text-align: center;
        border: 2px dashed #dee2e6;
    }
    .placeholder-icon {
        font-size: 2.5rem;
        margin: 0;
    }
    .placeholder-text {
        font-size: 0.95rem;
        color: #6c757d;
        margin-top: 0.3rem;
    }

    .interpret-box {
        border-radius: 8px;
        padding: 0.7rem 1rem;
        border-left: 4px solid;
    }
    .interpret-death {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .interpret-survive {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .interpret-title {
        font-weight: 600;
        margin: 0;
    }
    .interpret-text {
        font-size: 0.85rem;
        margin: 0.2rem 0 0 0;
    }

    .info-note {
        text-align: center;
        font-size: 0.75rem;
        color: #6c757d;
        margin-top: 0.5rem;
    }
    .threshold-note {
        text-align: center;
        color: #6c757d;
        font-size: 0.75rem;
        margin-bottom: 0.8rem;
        padding: 0.3rem;
        background-color: #f8f9fa;
        border-radius: 6px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Load Model ====================
@st.cache_resource
def load_model():
    """Load trained model and preprocessors"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

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
                return artifacts
            except Exception as e:
                continue

    st.error("❌ Model file not found")
    return None


artifacts = load_model()

if artifacts is None:
    st.stop()

model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['features']
categorical_features = artifacts['categorical_features']
continuous_features = artifacts['continuous_features']
model_threshold = artifacts.get('threshold', 0.5)

# ===== IMPORTANT: Override threshold to 0.50 for binary classification display =====
# This means only patients with ≥ 50% mortality probability will be classified as "Mortality"
DISPLAY_THRESHOLD = 0.50

model_info = artifacts.get('model_info', {})

# ==================== Header ====================
st.markdown('<p class="main-header">🏥 CKM 3-Year All-Cause Mortality Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Cardiovascular-Kidney-Metabolic Syndrome Risk Assessment Tool</p>',
            unsafe_allow_html=True)

# ==================== Model Metrics ====================
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Model</div>
        <div class="value">{model_info.get('type', 'Random Forest')}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">AUC</div>
        <div class="value">{model_info.get('auc', 0.835):.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Balanced Accuracy</div>
        <div class="value">{model_info.get('balanced_accuracy', 0.766):.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col_m4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Threshold</div>
        <div class="value">{DISPLAY_THRESHOLD:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ==================== Mapping ====================
gender_map = {"Male": 1, "Female": 0}
cancer_map = {"No": 0, "Yes": 1}
ckm_map = {"Stage 1": 1, "Stage 2": 2, "Stage 3": 3, "Stage 4": 4}

# ==================== Main Layout ====================
col_input, col_result = st.columns([1.3, 1], gap="large")

# ==================== Input Section ====================
with col_input:
    st.markdown("### 📋 Patient Characteristics")

    # ===== Categorical Variables =====
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">📌 Demographics</div>', unsafe_allow_html=True)

    col_gender, col_cancer, col_ckm = st.columns(3)
    with col_gender:
        gender = st.selectbox("Gender", options=["Male", "Female"])
    with col_cancer:
        cancer = st.selectbox("Cancer History", options=["No", "Yes"])
    with col_ckm:
        ckm = st.selectbox("CKM Stage", options=["Stage 1", "Stage 2", "Stage 3", "Stage 4"])
    st.markdown('</div>', unsafe_allow_html=True)

    # ===== Continuous Variables =====
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown('<div class="input-card-title">📊 Clinical Measurements</div>', unsafe_allow_html=True)

    col_age, col_mcv, col_rdw = st.columns(3)
    with col_age:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=65, step=1)
    with col_mcv:
        mcv = st.number_input("MCV (fL)", min_value=50.0, max_value=120.0, value=90.0, step=0.5)
    with col_rdw:
        rdw = st.number_input("RDW (%)", min_value=10.0, max_value=25.0, value=13.5, step=0.1)

    col_plt, col_alb, col_glb = st.columns(3)
    with col_plt:
        plt_val = st.number_input("PLT (×10⁹/L)", min_value=10, max_value=800, value=200, step=5)
    with col_alb:
        alb = st.number_input("Albumin (g/L)", min_value=15.0, max_value=55.0, value=40.0, step=0.5)
    with col_glb:
        glb = st.number_input("Globulin (g/L)", min_value=10.0, max_value=50.0, value=25.0, step=0.5)

    col_ast, col_crp = st.columns(2)
    with col_ast:
        ast = st.number_input("AST (U/L)", min_value=1, max_value=200, value=25, step=1)
    with col_crp:
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

    col_absi, col_sii = st.columns(2)
    with col_absi:
        absi = st.number_input("ABSI Index", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    with col_sii:
        sii = st.number_input("SII (×10⁹/L)", min_value=0, max_value=5000, value=500, step=50)

    st.markdown('</div>', unsafe_allow_html=True)

    predict_clicked = st.button("🔍 Predict", type="primary", width='stretch')


# ==================== Data Processing ====================
def create_input_data():
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
    cont_df = input_df[continuous_features]
    cat_df = input_df[categorical_features]
    cont_scaled = scaler.transform(cont_df)
    X_scaled = np.hstack([cont_scaled, cat_df.values])
    prob = model.predict_proba(X_scaled)[0, 1]
    return prob


# ==================== Results Section ====================
with col_result:
    st.markdown("### 🎯 Prediction Result")

    if predict_clicked:
        input_data = create_input_data()
        prob = predict_risk(input_data)

        # Calculate death and survival probabilities
        death_prob = prob * 100
        survival_prob = (1 - prob) * 100

        # Determine outcome using 50% threshold
        if prob >= DISPLAY_THRESHOLD:
            outcome_text = "Mortality"
            outcome_class = "outcome-death"
            outcome_icon = "⚠️"
            interpret_class = "interpret-death"
            interpret_title = "⚠️ High mortality risk detected (≥50%)"
            interpret_text = "Consider comprehensive clinical evaluation and intensive management."
        else:
            outcome_text = "Survived"
            outcome_class = "outcome-survive"
            outcome_icon = "✅"
            interpret_class = "interpret-survive"
            interpret_title = "✅ Low mortality risk detected (<50%)"
            interpret_text = "Continue routine monitoring and standard care."

        # ===== Result Card =====
        st.markdown(f"""
        <div class="result-card">
            <div class="result-number">
                {death_prob:.1f}<span class="percent">%</span>
            </div>
            <div class="result-label">3-Year Mortality Probability</div>
            <div class="result-outcome {outcome_class}">
                {outcome_icon} {outcome_text}
            </div>
            <div class="result-prob-detail">
                Death: {death_prob:.1f}% &nbsp;|&nbsp; Survival: {survival_prob:.1f}%
            </div>
            <div class="probability-bar">
                <div class="death-bar" style="width: {death_prob}%;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: #6c757d; margin-top: 0.2rem;">
                <span>0%</span>
                <span>50% (Threshold)</span>
                <span>100%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ===== Threshold Note =====
        st.markdown(f"""
        <div class="threshold-note">
            Classification Threshold: <strong>{DISPLAY_THRESHOLD:.2f}</strong> 
            (≥ {DISPLAY_THRESHOLD*100:.0f}% = Mortality)
        </div>
        """, unsafe_allow_html=True)

        # ===== Interpretation =====
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📖 Interpretation")

        st.markdown(f"""
        <div class="interpret-box {interpret_class}">
            <p class="interpret-title" style="color: {'#721c24' if prob >= DISPLAY_THRESHOLD else '#155724'};">
                {interpret_title}
            </p>
            <p class="interpret-text" style="color: {'#721c24' if prob >= DISPLAY_THRESHOLD else '#155724'};">
                {interpret_text}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ===== Note =====
        st.markdown("""
        <div class="info-note">
            ⚠️ This tool is for clinical research reference only, not for final diagnosis.
        </div>
        """, unsafe_allow_html=True)

    else:
        # ===== Placeholder =====
        st.markdown("""
        <div class="placeholder">
            <p class="placeholder-icon">🔬</p>
            <p class="placeholder-text">
                Enter patient characteristics<br>and click <strong>"Predict"</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ==================== Global Feature Importance ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 📈 Global Feature Importance")

try:
    importance = model.feature_importances_
    feature_names = continuous_features + categorical_features

    imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors_imp = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Global Feature Importance')
    st.pyplot(fig)

    with st.expander("📋 View Feature Importance Details"):
        st.dataframe(imp_df.sort_values('Importance', ascending=False), width='stretch')

except Exception as e:
    st.warning(f"⚠️ Unable to display feature importance: {e}")

# ==================== User Guide ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 📖 User Guide")

col_help1, col_help2, col_help3 = st.columns(3)

with col_help1:
    st.markdown("""
    **📝 Input Steps**
    1. Enter patient characteristics
    2. Select categorical variables
    3. Input continuous variables
    4. Click "Predict"
    """)

with col_help2:
    st.markdown("""
    **📊 Result Interpretation**
    - **Mortality Probability**: 3-year death risk
    - **Survival Probability**: 3-year survival rate
    - **Outcome**: Mortality (≥50%) / Survived (<50%)
    - **Visual Bar**: Probability distribution
    """)

with col_help3:
    st.markdown(f"""
    **💡 Model Information**
    - Algorithm: Random Forest
    - Features: 13 clinical variables
    - AUC: {model_info.get('auc', 0.835):.3f}
    - Balanced Accuracy: {model_info.get('balanced_accuracy', 0.766):.3f}
    - Classification Threshold: 0.50
    - For research reference only
    """)

# ==================== Footer ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    "⚠️ This tool is for clinical research reference only, not for final diagnosis | Model Version: v1.0 | Threshold: 0.50"
)