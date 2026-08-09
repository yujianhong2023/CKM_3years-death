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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6c757d;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-box .label {
        font-size: 0.7rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }
    .metric-box .value {
        font-size: 1.2rem;
        font-weight: 600;
        color: #1a1a2e;
    }
    .risk-card-high {
        background: linear-gradient(135deg, #dc3545, #b02a37);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-card-low {
        background: linear-gradient(135deg, #28a745, #1a7a34);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-card-moderate {
        background: linear-gradient(135deg, #ffc107, #d39e00);
        color: #1a1a2e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .risk-number {
        font-size: 3.5rem;
        font-weight: 700;
    }
    .risk-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .feature-row {
        display: flex;
        justify-content: space-between;
        padding: 0.4rem 0;
        border-bottom: 1px solid #f1f3f5;
    }
    .feature-name {
        font-size: 0.85rem;
        color: #495057;
    }
    .feature-value {
        font-size: 0.85rem;
        font-weight: 500;
        color: #1a1a2e;
    }
    .shap-positive {
        color: #dc3545;
    }
    .shap-negative {
        color: #007bff;
    }
    .divider {
        border-top: 1px solid #e9ecef;
        margin: 1.5rem 0;
    }
    .input-section {
        background-color: #fafbfc;
        border-radius: 12px;
        padding: 1.2rem;
        border: 1px solid #e9ecef;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #4a6cf7, #6a3de8);
        color: white;
        font-weight: 600;
        border: none;
        padding: 0.7rem;
        border-radius: 8px;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(74, 108, 247, 0.4);
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

# Model metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Model Type</div>
        <div class="value">{model_info.get('type', 'Random Forest')}</div>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">AUC</div>
        <div class="value">{model_info.get('auc', 0.835):.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Balanced Accuracy</div>
        <div class="value">{model_info.get('balanced_accuracy', 0.766):.3f}</div>
    </div>
    """, unsafe_allow_html=True)
with col4:
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
col_input, col_result = st.columns([1.2, 1], gap="large")

with col_input:
    st.markdown("### 📋 Patient Characteristics")

    with st.container():
        st.markdown('<div class="input-section">', unsafe_allow_html=True)

        # Categorical Variables
        col_gender, col_cancer, col_ckm = st.columns(3)
        with col_gender:
            gender = st.selectbox("Gender", options=["Male", "Female"])
        with col_cancer:
            cancer = st.selectbox("Cancer History", options=["No", "Yes"])
        with col_ckm:
            ckm = st.selectbox("CKM Stage", options=["Stage 1", "Stage 2", "Stage 3", "Stage 4"])

        st.markdown("#### Continuous Variables")

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

        # Row 3: AST, CRP, ABSI, SII
        col_ast, col_crp, col_absi, col_sii = st.columns(4)
        with col_ast:
            ast = st.number_input("AST (U/L)", min_value=1, max_value=200, value=25, step=1)
        with col_crp:
            crp = st.number_input("CRP (mg/L)", min_value=0.0, max_value=100.0, value=5.0, step=0.5)
        with col_absi:
            absi = st.number_input("ABSI Index", min_value=0.0, max_value=2.0, value=0.5, step=0.01)
        with col_sii:
            sii = st.number_input("SII (×10⁹/L)", min_value=0, max_value=5000, value=500, step=50)

        st.markdown('</div>', unsafe_allow_html=True)

        # Predict button
        predict_clicked = st.button("🔍 Predict", type="primary", use_container_width=True)


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


# ==================== Results ====================
with col_result:
    st.markdown("### 🎯 Prediction Result")

    if predict_clicked:
        input_data = create_input_data()
        prob, pred, X_scaled = predict_risk(input_data)

        # Risk card
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
        <div class="{risk_class}">
            <div class="risk-number">{prob * 100:.1f}%</div>
            <div class="risk-label">3-Year Mortality Risk</div>
            <div style="margin-top: 0.5rem; font-size: 1.1rem; font-weight: 600;">{risk_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # Prediction class
        st.markdown(f"""
        <div style="margin-top: 0.5rem; text-align: center; color: #6c757d; font-size: 0.85rem;">
            Predicted Class: <strong>{pred}</strong> (0. Survived, 1. Mortality)
            <br><span style="font-size: 0.75rem;">Threshold: {threshold:.3f}</span>
        </div>
        """, unsafe_allow_html=True)

        # ==================== SHAP Force Plot ====================
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 🔍 Model Explanation (SHAP)")

        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_scaled)

            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            feature_names = continuous_features + categorical_features
            expected_value = explainer.expected_value
            if isinstance(expected_value, list):
                expected_value = expected_value[1]

            # ===== SHAP Force Plot =====
            st.markdown("#### Force Plot")

            # Generate force plot
            force_plot_html = shap.force_plot(
                expected_value,
                shap_values,
                X_scaled,
                feature_names=feature_names,
                matplotlib=False,
                show=False
            )

            # Display force plot
            shap_html_str = f"""
            <div style="background-color: white; border-radius: 8px; padding: 0.5rem; overflow-x: auto; border: 1px solid #e9ecef;">
                {force_plot_html.html()}
            </div>
            """
            components.html(shap_html_str, height=150, scrolling=True)

            # ===== Feature Contribution Table =====
            st.markdown("#### Feature Contributions")

            shap_df = pd.DataFrame({
                'Feature': feature_names,
                'Value': X_scaled[0],
                'SHAP Value': shap_values[0]
            }).sort_values('SHAP Value', ascending=False)

            # Display top contributing features
            for _, row in shap_df.iterrows():
                color_class = "shap-positive" if row['SHAP Value'] > 0 else "shap-negative"
                arrow = "⬆" if row['SHAP Value'] > 0 else "⬇"
                impact = "Increases Risk" if row['SHAP Value'] > 0 else "Decreases Risk"
                st.markdown(f"""
                <div class="feature-row">
                    <span class="feature-name">{row['Feature']}</span>
                    <span>
                        <span class="feature-value">{row['Value']:.2f}</span>
                        <span class="feature-shap {color_class}">→ {row['SHAP Value']:.3f} {arrow}</span>
                    </span>
                </div>
                """, unsafe_allow_html=True)

            # ===== SHAP Bar Chart =====
            st.markdown("#### SHAP Value Summary")

            fig, ax = plt.subplots(figsize=(8, 4))
            shap_sorted = shap_df.sort_values('SHAP Value', ascending=True)
            colors = ['#dc3545' if x > 0 else '#007bff' for x in shap_sorted['SHAP Value']]
            ax.barh(shap_sorted['Feature'], shap_sorted['SHAP Value'], color=colors)
            ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
            ax.set_xlabel('SHAP Value')
            ax.set_title('Feature Impact on Mortality Prediction')
            from matplotlib.patches import Patch

            legend_elements = [
                Patch(facecolor='#dc3545', label='⬆ Increases Mortality Risk'),
                Patch(facecolor='#007bff', label='⬇ Decreases Mortality Risk')
            ]
            ax.legend(handles=legend_elements, loc='lower right')
            plt.tight_layout()
            st.pyplot(fig)

            # ===== Detailed SHAP Table =====
            with st.expander("📋 View Detailed SHAP Values"):
                shap_detail = pd.DataFrame({
                    'Feature': feature_names,
                    'SHAP Value': shap_values[0],
                    'Direction': ['⬆ Increases Risk' if x > 0 else '⬇ Decreases Risk' for x in shap_values[0]],
                    '|SHAP Value|': np.abs(shap_values[0])
                }).sort_values('|SHAP Value|', ascending=False)
                st.dataframe(shap_detail, use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ SHAP explanation generation failed: {e}")

        # ==================== Clinical Interpretation ====================
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📖 Interpretation")

        if prob >= threshold:
            st.markdown("""
            <div style="background-color: #f8d7da; border-radius: 8px; padding: 0.8rem 1rem; border-left: 4px solid #dc3545;">
                <p style="font-weight: 600; color: #721c24; margin: 0;">⚠️ Elevated 3-year mortality risk detected</p>
                <p style="color: #721c24; font-size: 0.85rem; margin: 0.3rem 0 0 0;">
                    Consider comprehensive clinical evaluation and intensive management.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background-color: #d4edda; border-radius: 8px; padding: 0.8rem 1rem; border-left: 4px solid #28a745;">
                <p style="font-weight: 600; color: #155724; margin: 0;">✅ Low 3-year mortality risk predicted</p>
                <p style="color: #155724; font-size: 0.85rem; margin: 0.3rem 0 0 0;">
                    Continue routine monitoring and standard care.
                </p>
            </div>
            """, unsafe_allow_html=True)

    else:
        # Placeholder
        st.markdown("""
        <div style="background-color: #f8f9fa; border-radius: 12px; padding: 3rem 2rem; text-align: center; border: 2px dashed #dee2e6;">
            <p style="font-size: 3rem; margin: 0;">🔬</p>
            <p style="font-size: 1rem; color: #6c757d; margin-top: 0.5rem;">
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

    fig, ax = plt.subplots(figsize=(10, 5))
    colors_imp = plt.cm.Blues(np.linspace(0.3, 0.9, len(imp_df)))[::-1]
    ax.barh(imp_df['Feature'], imp_df['Importance'], color=colors_imp)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Random Forest Global Feature Importance')
    st.pyplot(fig)

    with st.expander("📋 View Feature Importance Details"):
        st.dataframe(imp_df.sort_values('Importance', ascending=False), use_container_width=True)

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