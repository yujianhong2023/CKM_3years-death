import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt


# ==================== 加载SHAP重要性 ====================
@st.cache_resource
def load_shap_importance():
    """加载SHAP重要性数据"""
    shap_path = r"C:\Users\admin\PycharmProjects\PythonProject9\SHAP_Values_ExternalTest.csv"

    if os.path.exists(shap_path):
        try:
            shap_df = pd.read_csv(shap_path)
            # 计算平均绝对SHAP值
            feature_cols = [col for col in shap_df.columns if col not in ['Actual', 'Pred_Prob', 'Pred_Class']]
            mean_abs_shap = shap_df[feature_cols].abs().mean()

            shap_importance = pd.DataFrame({
                'Feature': feature_cols,
                'Mean_Abs_SHAP': mean_abs_shap.values
            }).sort_values('Mean_Abs_SHAP', ascending=False)

            return shap_importance
        except Exception as e:
            st.warning(f"⚠️ 无法加载SHAP重要性: {e}")
            return None
    return None


# ==================== 在Web中显示 ====================
# 原来的特征重要性代码
st.markdown("### 📈 Global Feature Importance")

# 尝试加载SHAP重要性
shap_importance = load_shap_importance()

if shap_importance is not None:
    # 显示SHAP重要性（推荐）
    st.info("📊 **SHAP-based Feature Importance** (External Validation Set)")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    colors_shap = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(shap_importance)))
    ax.barh(shap_importance['Feature'], shap_importance['Mean_Abs_SHAP'], color=colors_shap)
    ax.set_xlabel('Mean |SHAP Value|')
    ax.set_title('SHAP Feature Importance (External Validation Set)')
    st.pyplot(fig)

    with st.expander("📋 View SHAP Importance Details"):
        st.dataframe(shap_importance, width='stretch')

    # 添加RF内置重要性作为对比（可选）
    with st.expander("🔄 Compare with RF Built-in Importance"):
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
            st.pyplot(fig2)
        except Exception as e:
            st.warning(f"⚠️ 无法显示RF内置重要性: {e}")
else:
    # 如果SHAP不存在，使用RF内置重要性
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