import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# ==================== Page Configuration ====================
st.set_page_config(
    page_title="CKM 3-Year Mortality Risk Prediction",
    page_icon="🏥",
    layout="wide"
)

# ==================== Custom CSS ====================
st.markdown("""
<style>
    /* ===== Global ===== */
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

    /* ===== Metric Boxes ===== */
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

    /* ===== Input Cards ===== */
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
    .input-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f1f3f5;
    }
    .input-row:last-child {
        border-bottom: none;
    }
    .input-label {
        font-size: 0.85rem;
        color: #495057;
        font-weight: 500;
    }
    .input-value {
        font-size: 0.85rem;
        font-weight: 600;
        color: #1a1a2e;
        background-color: #f8f9fa;
        padding: 0.1rem 0.8rem;
        border-radius: 4px;
        min-width: 60px;
        text-align: center;
    }
    .input-unit {
        font-size: 0.7rem;
        color: #6c757d;
        font-weight: 400;
        margin-left: 4px;
    }

    /* ===== Risk Cards ===== */
    .risk-card {
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        color: white;
        margin-bottom: 0.8rem;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #dc3545, #b02a37);
    }
    .risk-card-moderate {
        background: linear-gradient(135deg, #ffc107, #d39e00);
        color: #1a1a2e;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #28a745, #1a7a34);
    }
    .risk-number {
        font-size: 2.8rem;
        font-weight: 700;
    }
    .risk-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .risk-text {
        font-size: 1.0rem;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* ===== SHAP ===== */
    .shap-container {
        background-color: white;
        border-radius: 8px;
        padding: 0.5rem;
        overflow-x: auto;
        border: 1px solid #e9ecef;
    }

    /* ===== Feature Row ===== */
    .feature-row {
        display: flex;
        justify-content: space-between;
        padding: 0.3rem 0;
        border-bottom: 1px solid #f1f3f5;
    }
    .feature-row:last-child {
        border-bottom: none;
    }
    .feature-name {
        font-size: 0.82rem;
        color: #495057;
    }
    .feature-shap-value {
        font-size: 0.82rem;
        font-weight: 500;
    }
    .shap-positive {
        color: #dc3545;
    }
    .shap-negative {
        color: #007bff;
    }

    /* ===== Buttons ===== */
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

    /* ===== Placeholder ===== */
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

    /* ===== Interpretation ===== */
    .interpret-box {
        border-radius: 8px;
        padding: 0.7rem 1rem;
        border-left: 4px solid;
    }
    .interpret-high {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .interpret-low {
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
threshold = artifacts.get('threshold', 0.5)
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
        <div class="value">{threshold:.4f}</div>
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

    # Row 1: Age, MCV, RDW
    col_age, col_mcv, col_rdw = st.columns(3)
    with col_age:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=65, step=1)
    with col_mcv:
        mcv = st.number_input("MCV (fL)", min_value=50.0, max_value=120.0, value=90.0, step=0.5)
    with col_rdw:
        rdw = st.number_input("RDW (%)", min_value=10.0, max_value=25.0, value=13.5, step=0.1)

    # Row 2: PLT, Albumin, Globulin
    col_plt, col_alb, col_glb = st.columns(3)
    with col_plt:
        plt_val = st.number_input("PLT (×10⁹/L)", min_value=10, max_value=800, value=200, step=5)
    with col_alb:
        alb = st.number_input("Albumin (g/L)", min_value=15.0, max_value=55.0, value=40.0, step=0.5)
    with col_glb:
        glb = st.number_input("Globulin (g/L)", min_value=10.0, max_value=50.0, value=25.0, step=0.5)

    # Row 3: AST, CRP
    col_ast, col_crp = st.columns(2)
    with col_ast:
        ast = st.number_input("AST (U/L)", min_value=1, max_value=200, value=25, step=1)
    with col_crp:
        crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)

    # Row 4: ABSI, SII
    col_absi, col_sii = st.columns(2)
    with col_absi:
        absi = st.number_input("ABSI Index", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
    with col_sii:
        sii = st.number_input("SII (×10⁹/L)", min_value=0, max_value=5000, value=500, step=50)

    st.markdown('</div>', unsafe_allow_html=True)

    # ===== Predict Button =====
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
    prediction = 1 if prob >= threshold else 0
    return prob, prediction, X_scaled


# ==================== SHAP Force Plot ====================
def display_shap_force_plot(expected_value, shap_values, X_scaled, feature_names):
    force_displayed = False

    if X_scaled.ndim == 1:
        X_scaled = X_scaled.reshape(1, -1)
    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(1, -1)

    if hasattr(shap, 'force_plot'):
        try:
            force_plot = shap.force_plot(
                expected_value, shap_values, X_scaled,
                feature_names=feature_names, show=False
            )
            if force_plot:
                shap_html = force_plot.html() if hasattr(force_plot, 'html') else str(force_plot)
                if not shap_html.startswith('<') and hasattr(force_plot, '_repr_html_'):
                    shap_html = force_plot._repr_html_()
                components.html(f'<div class="shap-container">{shap_html}</div>', height=150, scrolling=True)
                force_displayed = True
                return True
        except Exception as e:
            pass

    if not force_displayed and hasattr(shap, 'plots') and hasattr(shap.plots, 'force'):
        try:
            explanation = shap.Explanation(
                values=shap_values, base_values=expected_value,
                data=X_scaled, feature_names=feature_names
            )
            force_plot = shap.plots.force(explanation, show=False)
            if force_plot:
                shap_html = force_plot.html() if hasattr(force_plot, 'html') else str(force_plot)
                components.html(f'<div class="shap-container">{shap_html}</div>', height=150, scrolling=True)
                force_displayed = True
                return True
        except Exception as e:
            pass

    return False


# ==================== Results Section ====================
with col_result:
    st.markdown("### 🎯 Prediction Result")

    if predict_clicked:
        input_data = create_input_data()
        prob, pred, X_scaled = predict_risk(input_data)

        # ===== Risk Card =====
        if prob >= threshold:
            risk_class = "risk-card-high"
            risk_text = "HIGH RISK"
        elif prob >= 0.2:
            risk_class = "risk-card-moderate"
            risk_text = "MODERATE RISK"
        else:
            risk_class = "risk-card-low"
            risk_text = "LOW RISK"

        st.markdown(f"""
        <div class="risk-card {risk_class}">
            <div class="risk-number">{prob * 100:.1f}%</div>
            <div class="risk-label">3-Year Mortality Risk</div>
            <div class="risk-text">{risk_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ===== Predicted Class =====
        st.markdown(f"""
        <div style="text-align: center; color: #6c757d; font-size: 0.82rem; margin-bottom: 0.8rem;">
            Predicted Class: <strong>{pred}</strong> (0. Survived, 1. Mortality)
            <br><span style="font-size: 0.7rem;">Threshold: {threshold:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

        # ===== SHAP Explanation =====
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔍 Model Explanation")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            feature_names = continuous_features + categorical_features
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]

            # ===== Force Plot =====
            force_success = display_shap_force_plot(
                expected_value, shap_values, X_scaled, feature_names
            )

            if not force_success:
                st.info("💡 Showing SHAP bar chart")
                fig, ax = plt.subplots(figsize=(7, 3.5))
                shap_df_alt = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values[0]
                }).sort_values('SHAP Value', ascending=True)
                colors_alt = ['#dc3545' if x > 0 else '#007bff' for x in shap_df_alt['SHAP Value']]
                ax.barh(shap_df_alt['Feature'], shap_df_alt['SHAP Value'], color=colors_alt)
                ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
                ax.set_xlabel('SHAP Value')
                ax.set_title('Feature Impact on Mortality')
                from matplotlib.patches import Patch

                ax.legend(handles=[
                    Patch(facecolor='#dc3545', label='⬆ Increases Risk'),
                    Patch(facecolor='#007bff', label='⬇ Decreases Risk')
                ], loc='lower right')
                plt.tight_layout()
                st.pyplot(fig)

            # ===== Feature Contributions =====
            st.markdown("#### Feature Contributions")

            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': X_scaled[0],
                'SHAP Value': shap_values[0]
            }).sort_values('SHAP Value', ascending=False)

            for _, row in shap_df.iterrows():
                color_class = "shap-positive" if row['SHAP Value'] > 0 else "shap-negative"
                arrow = "⬆" if row['SHAP Value'] > 0 else "⬇"
                st.markdown(f"""
                <div class="feature-row">
                    <span class="feature-name">{row['Feature']}</span>
                    <span class="feature-shap-value {color_class}">{row['SHAP Value']:.3f} {arrow}</span>
                </div>
                """, unsafe_allow_html=True)

            # ===== SHAP Bar Chart =====
            st.markdown("#### SHAP Value Summary")

            fig, ax = plt.subplots(figsize=(7, 3.5))
            shap_sorted = shap_df.sort_values('SHAP Value', ascending=True)
            colors = ['#dc3545' if x > 0 else '#007bff' for x in shap_sorted['SHAP Value']]
            ax.barh(shap_sorted['Feature'], shap_sorted['SHAP Value'], color=colors)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('SHAP Value')
            ax.set_title('Feature Impact on Mortality')
            from matplotlib.patches import Patch

            ax.legend(handles=[
                Patch(facecolor='#dc3545', label='⬆ Increases Risk'),
                Patch(facecolor='#007bff', label='⬇ Decreases Risk')
            ], loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)

            # ===== Detailed Table =====
            with st.expander("📋 View Detailed SHAP Values"):
                shap_detail = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values[0],
                    'Direction': ['⬆ Increases Risk' if x > 0 else '⬇ Decreases Risk' for x in shap_values[0]],
                    '|SHAP Value|': np.abs(shap_values[0])
                }).sort_values('|SHAP Value|', ascending=False)
                st.dataframe(shap_detail, width='stretch')

        except Exception as e:
            st.warning(f"⚠️ SHAP explanation failed: {e}")

        # ===== Interpretation =====
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📖 Interpretation")

        if prob >= threshold:
            st.markdown("""
            <div class="interpret-box interpret-high">
                <p class="interpret-title" style="color:#721c24;">⚠️ Elevated 3-year mortality risk detected</p>
                <p class="interpret-text" style="color:#721c24;">
                    Consider comprehensive clinical evaluation and intensive management.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="interpret-box interpret-low">
                <p class="interpret-title" style="color:#155724;">✅ Low 3-year mortality risk predicted</p>
                <p class="interpret-text" style="color:#155724;">
                    Continue routine monitoring and standard care.
                </p>
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
    - **Risk Probability**: 3-year mortality risk
    - **Risk Level**: Low/Moderate/High
    - **SHAP Values**: Feature contributions
    - **Red**: Increases risk
    - **Blue**: Decreases risk
    """)

with col_help3:
    st.markdown(f"""
    **💡 Model Information**
    - Algorithm: Random Forest
    - Features: 13 clinical variables
    - AUC: {model_info.get('auc', 0.835):.3f}
    - Balanced Accuracy: {model_info.get('balanced_accuracy', 0.766):.3f}
    - For research reference only
    """)

# ==================== Footer ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    "⚠️ This tool is for clinical research reference only, not for final diagnosis | Model Version: v1.0"
)