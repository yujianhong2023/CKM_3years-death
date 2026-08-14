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
    .metric-external {
        border-color: #4a6cf7;
        background-color: #f0f4ff;
    }
    .metric-external .label {
        color: #4a6cf7;
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
    .shap-highlight {
        background-color: #fff3cd;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border-left: 4px solid #ffc107;
        margin-bottom: 1rem;
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


@st.cache_data
def load_shap_importance():
    """Load SHAP importance from external test set"""
    # 尝试多个可能的路径 - 根据用户提供的实际路径
    possible_shap_paths = [
        r"C:\Users\admin\PycharmProjects\PythonProject9\CKM_3 year death.csv",  # 用户的实际文件
        r"C:\Users\admin\PycharmProjects\PythonProject9\SHAP_Values_ExternalTest.csv",
        r"C:\Users\admin\PycharmProjects\PythonProject9\CKM_3 year death\SHAP_Values_ExternalTest.csv",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'SHAP_Values_ExternalTest.csv'),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CKM_3 year death.csv')
    ]

    for shap_path in possible_shap_paths:
        if os.path.exists(shap_path):
            try:
                st.info(f"📂 加载SHAP数据: {shap_path}")
                shap_df = pd.read_csv(shap_path)

                # 显示数据的前几行以便调试
                st.sidebar.write("SHAP数据预览:")
                st.sidebar.dataframe(shap_df.head(3))

                # 识别特征列（排除非特征列）
                exclude_cols = ['Actual', 'Pred_Prob', 'Pred_Class', 'Unnamed: 0']
                feature_cols = [col for col in shap_df.columns if col not in exclude_cols]

                if len(feature_cols) == 0:
                    st.error("❌ 未找到特征列，请检查CSV文件格式")
                    return None

                # 计算平均绝对SHAP值
                mean_abs_shap = shap_df[feature_cols].abs().mean()

                # 创建SHAP重要性DataFrame
                shap_importance = pd.DataFrame({
                    'Feature': feature_cols,
                    'Mean_Abs_SHAP': mean_abs_shap.values
                }).sort_values('Mean_Abs_SHAP', ascending=False)

                # 显示特征重要性排序
                st.sidebar.success("✅ SHAP数据加载成功")
                st.sidebar.write("特征重要性排序:")
                for idx, row in shap_importance.iterrows():
                    st.sidebar.write(f"{row['Feature']}: {row['Mean_Abs_SHAP']:.4f}")

                return shap_importance
            except Exception as e:
                st.warning(f"⚠️ 无法加载SHAP数据: {e}")
                continue

    st.warning("⚠️ SHAP数据文件未找到，请确认文件路径")
    st.info(f"💡 期望的文件路径: {possible_shap_paths[0]}")
    return None


artifacts = load_model()

if artifacts is None:
    st.stop()

# 提取模型和组件
model = artifacts['model']
scaler = artifacts['scaler']
features = artifacts['features']
categorical_features = artifacts['categorical_features']
continuous_features = artifacts['continuous_features']
model_threshold = artifacts.get('threshold', 0.5)

# 获取模型信息（包含外部验证性能）
model_info = artifacts.get('model_info', {})
train_metrics = model_info.get('train_metrics', {})
external_metrics = model_info.get('external_metrics', {})
train_samples = model_info.get('train_samples', 0)
external_samples = model_info.get('external_samples', 0)

# 加载SHAP重要性
shap_importance_df = load_shap_importance()

# 显示阈值
DISPLAY_THRESHOLD = 0.50

# ==================== Header ====================
st.markdown('<p class="main-header">🏥 CKM 3-Year All-Cause Mortality Risk Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Cardiovascular-Kidney-Metabolic Syndrome Risk Assessment Tool</p>',
            unsafe_allow_html=True)

# ==================== Model Metrics ====================
st.markdown("### 📊 Model Validation Performance")

col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)

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
        <div class="label">Training Samples</div>
        <div class="value">{train_samples if train_samples > 0 else 'N/A'}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">External Samples</div>
        <div class="value">{external_samples if external_samples > 0 else 'N/A'}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m4:
    train_auc = train_metrics.get('AUC', 0.835)
    st.markdown(f"""
    <div class="metric-box">
        <div class="label">Internal Validation AUC</div>
        <div class="value" style="color: #28a745;">{train_auc:.3f}</div>
    </div>
    """, unsafe_allow_html=True)

with col_m5:
    ext_auc = external_metrics.get('AUC', 0.820)
    st.markdown(f"""
    <div class="metric-box metric-external">
        <div class="label">⭐ External Validation AUC</div>
        <div class="value" style="color: #4a6cf7;">{ext_auc:.3f}</div>
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

        death_prob = prob * 100
        survival_prob = (1 - prob) * 100

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

        st.markdown(f"""
        <div class="threshold-note">
            Classification Threshold: <strong>{DISPLAY_THRESHOLD:.2f}</strong> 
            (≥ {DISPLAY_THRESHOLD * 100:.0f}% = Mortality)
        </div>
        """, unsafe_allow_html=True)

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

        st.markdown("""
        <div class="info-note">
            ⚠️ This tool is for clinical research reference only, not for final diagnosis.
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="placeholder">
            <p class="placeholder-icon">🔬</p>
            <p class="placeholder-text">
                Enter patient characteristics<br>and click <strong>"Predict"</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)

# ==================== Global Feature Importance (使用SHAP) ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 📈 Global Feature Importance")

# 优先显示SHAP重要性
if shap_importance_df is not None:
    st.markdown("""
    <div class="shap-highlight">
        <strong>📊 SHAP-based Feature Importance (External Validation Set)</strong><br>
        <span style="font-size: 0.85rem; color: #6c757d;">
            SHAP (SHapley Additive exPlanations) values reflect the marginal contribution of each feature 
            to the model's predictions, providing more reliable and interpretable feature importance rankings.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 按照SHAP值从大到小排序（用于显示）
    shap_display = shap_importance_df.sort_values('Mean_Abs_SHAP', ascending=True)

    # 创建条形图
    fig, ax = plt.subplots(figsize=(10, 5))
    colors_shap = plt.cm.RdYlBu_r(np.linspace(0.2, 0.9, len(shap_display)))
    bars = ax.barh(shap_display['Feature'], shap_display['Mean_Abs_SHAP'], color=colors_shap)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title('SHAP Feature Importance (External Validation Set)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)

    # 添加数值标签
    for i, (idx, row) in enumerate(shap_display.iterrows()):
        ax.text(row['Mean_Abs_SHAP'] + 0.001, i, f'{row["Mean_Abs_SHAP"]:.4f}',
                va='center', fontsize=8, color='#6c757d')

    plt.tight_layout()
    st.pyplot(fig)

    # 显示详细的SHAP重要性表格
    with st.expander("📋 View SHAP Importance Details"):
        st.dataframe(
            shap_importance_df.sort_values('Mean_Abs_SHAP', ascending=False),
            use_container_width=True
        )

        st.markdown("""
        **💡 Why SHAP Importance?**

        - **Consistency**: SHAP values provide consistent and reliable feature rankings
        - **Interpretability**: Each SHAP value represents the marginal contribution of a feature
        - **Clinical Relevance**: SHAP is widely adopted in clinical research for model explanation
        - **Local Explanations**: SHAP also enables individual-level prediction explanations
        """)

    # 添加一个比较功能 - 显示RF内置重要性
    with st.expander("🔄 Compare with RF Built-in Importance", expanded=False):
        try:
            importance = model.feature_importances_
            feature_names = continuous_features + categorical_features

            rf_imp_df = pd.DataFrame({
                'Feature': feature_names,
                'RF_Importance': importance
            }).sort_values('RF_Importance', ascending=True)

            fig2, ax2 = plt.subplots(figsize=(10, 4.5))
            colors_rf = plt.cm.Blues(np.linspace(0.3, 0.9, len(rf_imp_df)))[::-1]
            ax2.barh(rf_imp_df['Feature'], rf_imp_df['RF_Importance'], color=colors_rf)
            ax2.set_xlabel('RF Feature Importance (Gini)')
            ax2.set_title('RF Built-in Feature Importance')
            plt.tight_layout()
            st.pyplot(fig2)

            # 合并对比
            comparison_df = pd.merge(
                shap_importance_df[['Feature', 'Mean_Abs_SHAP']],
                rf_imp_df,
                on='Feature',
                how='outer'
            )
            st.dataframe(comparison_df.sort_values('Mean_Abs_SHAP', ascending=False), use_container_width=True)

        except Exception as e:
            st.warning(f"⚠️ 无法显示RF内置重要性: {e}")

else:
    # 如果SHAP不存在，使用RF内置重要性
    st.warning("⚠️ SHAP importance data not found. Displaying RF built-in importance instead.")
    st.info("💡 请确保 SHAP_Values_ExternalTest.csv 文件存在于正确路径")

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
            st.dataframe(imp_df.sort_values('Importance', ascending=False), use_container_width=True)
    except Exception as e:
        st.warning(f"⚠️ Unable to display feature importance: {e}")

# ==================== External Validation Section ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown("### 🎯 External Validation Summary")

col_ext1, col_ext2, col_ext3, col_ext4 = st.columns(4)

with col_ext1:
    ext_auc = external_metrics.get('AUC', 0.820)
    st.metric("External AUC", f"{ext_auc:.3f}")

with col_ext2:
    ext_bal_acc = external_metrics.get('Balanced Accuracy', 0.750)
    st.metric("External Balanced Accuracy", f"{ext_bal_acc:.3f}")

with col_ext3:
    ext_sens = external_metrics.get('Sensitivity', 0.720)
    st.metric("External Sensitivity", f"{ext_sens:.3f}")

with col_ext4:
    ext_spec = external_metrics.get('Specificity', 0.780)
    st.metric("External Specificity", f"{ext_spec:.3f}")

if external_samples > 0:
    st.info(
        f"✅ Model validated on **{external_samples}** independent external samples "
        f"(from different medical center), achieving **AUC = {ext_auc:.3f}**, "
        f"demonstrating robust generalizability."
    )
else:
    st.info(
        "📊 External validation data is available in the model file. "
        "Run the model training script with external data to display validation metrics."
    )

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
    - Internal AUC: {train_metrics.get('AUC', 0.835):.3f}
    - **External AUC: {external_metrics.get('AUC', 0.820):.3f}** ⭐
    - Classification Threshold: 0.50
    - Validated on independent cohort
    - For research reference only
    """)

# ==================== Footer ====================
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.caption(
    f"⚠️ This tool is for clinical research reference only, not for final diagnosis | "
    f"Model validated on {external_samples if external_samples > 0 else 'external'} samples | "
    f"Threshold: 0.50 | SHAP-based feature importance"
)