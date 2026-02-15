"""
Multi-Model Classification Framework - Web Interface
Author: Abhi
BITS Pilani - ML Assignment 2

A professional web application for comparing multiple ML classification algorithms
on any dataset with comprehensive metrics and visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns

# Page configuration
st.set_page_config(
    page_title="Multi-Model Classification Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4788;
        font-weight: bold;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e5c8a;
        font-weight: bold;
        margin-top: 1.5rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4788;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Paths
MODELS_DIR = Path("resources/models")
DATA_DIR = Path("resources/data")

# Model mapping
MODEL_FILES = {
    'Logistic Regression': 'logistic_regression_model.pkl',
    'Decision Tree': 'decision_tree_model.pkl',
    'kNN': 'knn_model.pkl',
    'Naive Bayes': 'naive_bayes_model.pkl',
    'Random Forest': 'random_forest_model.pkl',
    'XGBoost': 'xgboost_model.pkl'
}

MODEL_DESCRIPTIONS = {
    'Logistic Regression': 'Linear classifier - Fast, interpretable, probabilistic',
    'Decision Tree': 'Tree-based - Non-linear, interpretable decisions',
    'kNN': 'Instance-based - No training phase, flexible',
    'Naive Bayes': 'Probabilistic - Fast, works well with small data',
    'Random Forest': 'Ensemble - Robust, reduces overfitting',
    'XGBoost': 'Gradient Boosting - State-of-the-art performance'
}

# Cache functions
@st.cache_resource
def load_model(model_name):
    """Load trained model from pickle file"""
    model_path = MODELS_DIR / MODEL_FILES[model_name]
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model {model_name}: {e}")
        return None

@st.cache_resource
def load_scaler():
    """Load fitted scaler"""
    scaler_path = MODELS_DIR / "scaler.pkl"
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        return scaler
    except Exception as e:
        st.error(f"Error loading scaler: {e}")
        return None

@st.cache_data
def load_results():
    """Load model comparison results"""
    results_path = MODELS_DIR / "model_results.csv"
    try:
        return pd.read_csv(results_path)
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

# Main title
st.markdown('<p class="main-header">🤖 Multi-Model Classification Framework</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    <strong>BITS Pilani M.Tech (AIML/DSE) - Machine Learning Assignment 2</strong><br>
    <i>Compare 6 ML Algorithms | Upload Any Dataset | Get Instant Predictions & Metrics</i>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")

    # Model selection
    st.markdown("#### Select Algorithm")
    selected_model = st.selectbox(
        "Choose classification model",
        list(MODEL_FILES.keys()),
        label_visibility="collapsed",
        help="Select from 6 pre-trained classification algorithms"
    )

    # Show model description
    st.info(f"**{selected_model}**\n\n{MODEL_DESCRIPTIONS[selected_model]}")

    st.markdown("---")

    # File upload
    st.markdown("### 📁 Upload Your Data")
    st.markdown("Upload a CSV file with numeric features")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        label_visibility="collapsed",
        help="CSV with numeric features. Optional: include 'target' column for evaluation"
    )

    if uploaded_file is None:
        st.markdown("""
        <div class='info-box'>
        <strong>💡 No file uploaded</strong><br>
        Upload your dataset to get predictions!<br><br>
        <small>Sample data available in:<br>
        <code>resources/data/test_data.csv</code></small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Framework info
    st.markdown("### 📊 Framework Features")
    st.markdown("""
    ✅ 6 ML Algorithms  
    ✅ 6 Evaluation Metrics  
    ✅ Any Dataset Support  
    ✅ Real-time Predictions  
    ✅ Visual Analytics  
    ✅ Export Results  
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Student ID:** 2025aa05325")
    st.markdown("**BITS Pilani** | Feb 2026")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f'<p class="sub-header">🎯 Model: {selected_model}</p>', unsafe_allow_html=True)

    # Load model and scaler
    model = load_model(selected_model)
    scaler = load_scaler()

    if model is None or scaler is None:
        st.error("⚠️ Failed to load model or scaler. Please ensure models are trained.")
        st.info("💡 Run the training script first: `python -m com.abhi.ml.src.main`")
    elif uploaded_file is not None:
        try:
            # Load uploaded data
            test_data = pd.read_csv(uploaded_file)

            st.success(f"✅ Successfully loaded **{len(test_data)} samples** from your dataset")

            # Show data preview
            with st.expander("📋 Preview Uploaded Data (First 10 rows)"):
                st.dataframe(test_data.head(10), use_container_width=True)

            # Check for target column
            has_labels = 'target' in test_data.columns

            if has_labels:
                X_test = test_data.drop('target', axis=1)
                y_test = test_data['target']
                st.info(f"📊 Dataset includes **target labels** - Full evaluation metrics will be shown")
            else:
                X_test = test_data
                st.warning("⚠️ No 'target' column detected - **Prediction-only mode**")

            # Show feature info
            st.markdown(f"""
            <div class='info-box'>
            <strong>Dataset Information:</strong><br>
            • Samples: {len(X_test)}<br>
            • Features: {X_test.shape[1]}<br>
            • Mode: {'Evaluation (with labels)' if has_labels else 'Prediction only'}<br>
            </div>
            """, unsafe_allow_html=True)

            # Prediction button
            if st.button("🚀 Run Predictions", type="primary", use_container_width=True):
                with st.spinner(f"Running {selected_model} predictions..."):
                    try:
                        # Make predictions (data is already scaled)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                        # Display predictions
                        st.markdown('<p class="sub-header">📊 Prediction Results</p>', unsafe_allow_html=True)

                        # Create predictions dataframe
                        pred_df = pd.DataFrame({
                            'Sample': range(1, len(y_pred) + 1),
                            'Predicted Class': y_pred,
                            'Confidence': [f"{max(p, 1-p)*100:.2f}%" for p in y_pred_proba],
                            'Class Probability': [f"{p:.4f}" for p in y_pred_proba]
                        })

                        if has_labels:
                            pred_df['Actual Class'] = y_test.values
                            pred_df['Result'] = ['✅ Correct' if pred_df['Predicted Class'][i] == pred_df['Actual Class'][i]
                                                else '❌ Wrong' for i in range(len(pred_df))]

                        st.dataframe(pred_df, use_container_width=True, height=400)

                        # Summary statistics
                        if has_labels:
                            correct = (y_pred == y_test).sum()
                            total = len(y_test)
                            st.markdown(f"""
                            <div class='metric-card'>
                            <h3>Quick Summary</h3>
                            ✅ Correct Predictions: <strong>{correct} / {total}</strong> ({correct/total*100:.2f}%)<br>
                            ❌ Incorrect Predictions: <strong>{total - correct}</strong>
                            </div>
                            """, unsafe_allow_html=True)

                        # Metrics section (only if labels available)
                        if has_labels:
                            st.markdown('<p class="sub-header">📈 Performance Metrics</p>', unsafe_allow_html=True)

                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            auc = roc_auc_score(y_test, y_pred_proba)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            mcc = matthews_corrcoef(y_test, y_pred)

                            # Display metrics in columns
                            metric_cols = st.columns(3)

                            with metric_cols[0]:
                                st.metric("Accuracy", f"{accuracy:.4f}",
                                         delta=f"{(accuracy-0.5)*100:.1f}% vs random")
                                st.metric("Precision", f"{precision:.4f}",
                                         help="Positive Predictive Value")

                            with metric_cols[1]:
                                st.metric("AUC-ROC", f"{auc:.4f}",
                                         help="Area Under ROC Curve")
                                st.metric("Recall", f"{recall:.4f}",
                                         help="Sensitivity / True Positive Rate")

                            with metric_cols[2]:
                                st.metric("F1-Score", f"{f1:.4f}",
                                         help="Harmonic Mean of Precision & Recall")
                                st.metric("MCC", f"{mcc:.4f}",
                                         help="Matthews Correlation Coefficient")

                            # Confusion Matrix
                            st.markdown('<p class="sub-header">🎯 Confusion Matrix</p>', unsafe_allow_html=True)

                            cm = confusion_matrix(y_test, y_pred)

                            # Determine class labels
                            unique_classes = sorted(y_test.unique())
                            class_labels = [f'Class {i}' for i in unique_classes]

                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                       xticklabels=class_labels,
                                       yticklabels=class_labels,
                                       ax=ax, cbar_kws={'label': 'Count'})
                            ax.set_ylabel('Actual Class', fontsize=12, fontweight='bold')
                            ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
                            ax.set_title(f'Confusion Matrix - {selected_model}',
                                        fontsize=14, fontweight='bold', pad=20)

                            # Add percentage annotations
                            total = cm.sum()
                            for i in range(len(unique_classes)):
                                for j in range(len(unique_classes)):
                                    percentage = (cm[i, j] / total) * 100
                                    ax.text(j+0.5, i+0.7, f'({percentage:.1f}%)',
                                           ha='center', va='center', fontsize=10, color='gray')

                            st.pyplot(fig)

                            # Classification Report
                            st.markdown('<p class="sub-header">📋 Detailed Classification Report</p>', unsafe_allow_html=True)

                            report = classification_report(y_test, y_pred,
                                                          target_names=class_labels,
                                                          output_dict=True,
                                                          zero_division=0)
                            report_df = pd.DataFrame(report).transpose()

                            # Style the dataframe
                            st.dataframe(
                                report_df.style.format("{:.4f}")
                                .background_gradient(cmap='RdYlGn',
                                                    subset=['precision', 'recall', 'f1-score']),
                                use_container_width=True
                            )

                        # Download predictions
                        st.markdown("---")
                        st.markdown("### 💾 Export Results")
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"❌ Error during prediction: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.info("💡 Make sure your CSV file contains numeric features only (except optional 'target' column)")
            st.exception(e)
    else:
        # Instructions when no file uploaded
        st.markdown("""
        <div class='info-box'>
        <h3>👈 Getting Started</h3>
        <ol>
        <li><strong>Upload your dataset</strong> using the sidebar file uploader</li>
        <li><strong>Select a model</strong> from the dropdown menu</li>
        <li><strong>Click "Run Predictions"</strong> to get results</li>
        <li><strong>Download results</strong> as CSV for further analysis</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📝 Dataset Requirements")
        st.markdown("""
        Your CSV file should contain:
        
        ✅ **Numeric features only** (no text or categorical data)  
        ✅ **Scaled/preprocessed data** (if using pre-trained models)  
        ✅ **Optional 'target' column** for evaluation metrics  
        
        **Example CSV format:**
```csv
        feature1,feature2,feature3,...,featureN,target
        1.2,3.4,5.6,...,9.8,0
        2.1,4.3,6.5,...,8.7,1
        ...
```
        
        **Note:** If your data is not pre-scaled, results may vary. For best results, 
        ensure your data is preprocessed similarly to the training data.
        """)

        st.markdown("### 🎯 Available Models")

        model_info = []
        for model, desc in MODEL_DESCRIPTIONS.items():
            model_info.append({"Algorithm": model, "Description": desc})

        st.table(pd.DataFrame(model_info))

with col2:
    st.markdown('<p class="sub-header">📊 Model Comparison</p>', unsafe_allow_html=True)

    # Load results
    results_df = load_results()

    if results_df is not None:
        # Display comparison table
        st.markdown("#### Performance on Training Dataset")
        st.dataframe(
            results_df.style.highlight_max(
                axis=0,
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='lightgreen'
            ).format({
                'Accuracy': '{:.4f}',
                'AUC': '{:.4f}',
                'Precision': '{:.4f}',
                'Recall': '{:.4f}',
                'F1': '{:.4f}',
                'MCC': '{:.4f}'
            }),
            use_container_width=True,
            height=280
        )

        # Best model info
        best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = results_df['Accuracy'].max()

        st.success(f"🏆 **Top Performer:** {best_model} ({best_accuracy:.2%})")

        # Metrics comparison chart
        st.markdown("#### Visual Comparison")

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        x = np.arange(len(results_df))
        width = 0.12

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot)/2) * width + width/2
            ax.bar(x + offset, results_df[metric], width,
                  label=metric, color=colors[i], alpha=0.8)

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Algorithm Performance Comparison', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0.8, 1.0])

        plt.tight_layout()
        st.pyplot(fig)

        # Framework info
        st.markdown("#### 🎓 Framework Info")
        st.markdown("""
        **Algorithms:** 6 models  
        **Metrics:** 6 comprehensive  
        **Flexibility:** Any dataset  
        **Output:** Predictions + Metrics  
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Multi-Model Classification Framework</strong> | BITS Pilani M.Tech (AIML/DSE)</p>
    <p>A professional ML framework for comparing classification algorithms on any dataset</p>
    <p style='font-size: 0.9rem;'>Student ID: 2025aa05325 | February 2026</p>
</div>
""", unsafe_allow_html=True)