"""
Streamlit Web Application
Author: Abhinav
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
    page_title="Breast Cancer Classification",
    page_icon="🏥",
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
        margin-bottom: 2rem;
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
st.markdown('<p class="main-header">🏥 Breast Cancer Classification System</p>', unsafe_allow_html=True)
st.markdown("""
<div style='text-align: center; color: #666; margin-bottom: 2rem;'>
    Machine Learning Assignment 2 - BITS Pilani M.Tech (AIML/DSE)<br>
    <i>Multi-Model Classification for Cancer Diagnosis</i>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=150)
    st.markdown("---")

    st.markdown("### 🔧 Configuration")

    # Model selection
    st.markdown("#### Select Model")
    selected_model = st.selectbox(
        "Choose classification model",
        list(MODEL_FILES.keys()),
        label_visibility="collapsed"
    )

    st.markdown("---")

    # File upload
    st.markdown("### 📁 Upload Test Data")
    st.markdown("Upload CSV file with 30 features")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        label_visibility="collapsed"
    )

    if uploaded_file is None:
        st.info("💡 Use the sample test data provided in the dataset")

    st.markdown("---")

    # Dataset info
    st.markdown("### 📊 Dataset Info")
    st.markdown("""
    - **Samples:** 569
    - **Features:** 30
    - **Classes:** 2 (Binary)
    - **Train:** 455 samples
    - **Test:** 114 samples
    """)

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Abhi**")
    st.markdown("BITS Pilani 2026")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown(f'<p class="sub-header">Selected Model: {selected_model}</p>', unsafe_allow_html=True)

    # Load model and scaler
    model = load_model(selected_model)
    scaler = load_scaler()

    if model is None or scaler is None:
        st.error("⚠️ Failed to load model or scaler. Please check the model files.")
    elif uploaded_file is not None:
        try:
            # Load uploaded data
            test_data = pd.read_csv(uploaded_file)

            st.success(f"✅ Loaded {len(test_data)} test samples")

            # Show data preview
            with st.expander("📋 View Data Preview"):
                st.dataframe(test_data.head(10), use_container_width=True)

            # Check for target column
            has_labels = 'target' in test_data.columns

            if has_labels:
                X_test = test_data.drop('target', axis=1)
                y_test = test_data['target']
            else:
                X_test = test_data
                st.warning("⚠️ No 'target' column found. Predictions only mode.")

            # Prediction button
            if st.button("🔍 Run Predictions", type="primary", use_container_width=True):
                with st.spinner("Making predictions..."):
                    try:
                        # Make predictions (data is already scaled)
                        y_pred = model.predict(X_test)
                        y_pred_proba = model.predict_proba(X_test)[:, 1]

                        # Display predictions
                        st.markdown('<p class="sub-header">Prediction Results</p>', unsafe_allow_html=True)

                        # Create predictions dataframe
                        pred_df = pd.DataFrame({
                            'Sample #': range(1, len(y_pred) + 1),
                            'Prediction': ['Benign' if p == 1 else 'Malignant' for p in y_pred],
                            'Confidence': [f"{max(p, 1 - p) * 100:.2f}%" for p in y_pred_proba],
                            'Probability (Benign)': [f"{p:.4f}" for p in y_pred_proba]
                        })

                        if has_labels:
                            pred_df['Actual'] = ['Benign' if p == 1 else 'Malignant' for p in y_test]
                            pred_df['Status'] = ['✅ Correct' if pred_df['Prediction'][i] == pred_df['Actual'][i]
                                                 else '❌ Wrong' for i in range(len(pred_df))]

                        st.dataframe(pred_df, use_container_width=True, height=400)

                        # Metrics section (only if labels available)
                        if has_labels:
                            st.markdown('<p class="sub-header">📊 Performance Metrics</p>', unsafe_allow_html=True)

                            # Calculate metrics
                            accuracy = accuracy_score(y_test, y_pred)
                            auc = roc_auc_score(y_test, y_pred_proba)
                            precision = precision_score(y_test, y_pred)
                            recall = recall_score(y_test, y_pred)
                            f1 = f1_score(y_test, y_pred)
                            mcc = matthews_corrcoef(y_test, y_pred)

                            # Display metrics in columns
                            metric_cols = st.columns(3)

                            with metric_cols[0]:
                                st.metric("Accuracy", f"{accuracy:.4f}",
                                          delta=f"{(accuracy - 0.5) * 100:.1f}% vs random")
                                st.metric("Precision", f"{precision:.4f}")

                            with metric_cols[1]:
                                st.metric("AUC Score", f"{auc:.4f}")
                                st.metric("Recall", f"{recall:.4f}")

                            with metric_cols[2]:
                                st.metric("F1 Score", f"{f1:.4f}")
                                st.metric("MCC Score", f"{mcc:.4f}")

                            # Confusion Matrix
                            st.markdown('<p class="sub-header">🎯 Confusion Matrix</p>', unsafe_allow_html=True)

                            cm = confusion_matrix(y_test, y_pred)

                            fig, ax = plt.subplots(figsize=(8, 6))
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                                        xticklabels=['Malignant', 'Benign'],
                                        yticklabels=['Malignant', 'Benign'],
                                        ax=ax, cbar_kws={'label': 'Count'})
                            ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
                            ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
                            ax.set_title(f'Confusion Matrix - {selected_model}', fontsize=14, fontweight='bold', pad=20)

                            # Add annotations
                            total = cm.sum()
                            for i in range(2):
                                for j in range(2):
                                    percentage = (cm[i, j] / total) * 100
                                    ax.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)',
                                            ha='center', va='center', fontsize=10, color='gray')

                            st.pyplot(fig)

                            # Classification Report
                            st.markdown('<p class="sub-header">📋 Classification Report</p>', unsafe_allow_html=True)

                            report = classification_report(y_test, y_pred,
                                                           target_names=['Malignant', 'Benign'],
                                                           output_dict=True)
                            report_df = pd.DataFrame(report).transpose()

                            # Style the dataframe
                            st.dataframe(
                                report_df.style.format("{:.4f}").background_gradient(cmap='RdYlGn',
                                                                                     subset=['precision', 'recall',
                                                                                             'f1-score']),
                                use_container_width=True
                            )

                        # Download predictions
                        st.markdown("---")
                        csv = pred_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions as CSV",
                            data=csv,
                            file_name=f"{selected_model.lower().replace(' ', '_')}_predictions.csv",
                            mime="text/csv"
                        )

                    except Exception as e:
                        st.error(f"❌ Error making predictions: {str(e)}")
                        st.exception(e)

        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
            st.exception(e)
    else:
        # Instructions when no file uploaded
        st.info("👈 Please upload a CSV file from the sidebar to begin predictions")

        st.markdown("### 📝 Expected Data Format")
        st.markdown("""
        Your CSV file should contain **30 features** (all numeric):

        **Mean values (10 features):**
        - mean radius, mean texture, mean perimeter, mean area, mean smoothness
        - mean compactness, mean concavity, mean concave points, mean symmetry, mean fractal dimension

        **Standard error values (10 features):**
        - radius error, texture error, perimeter error, area error, smoothness error
        - compactness error, concavity error, concave points error, symmetry error, fractal dimension error

        **Worst values (10 features):**
        - worst radius, worst texture, worst perimeter, worst area, worst smoothness
        - worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension

        **Optional:**
        - Include a `target` column (0=Malignant, 1=Benign) for performance evaluation
        """)

        st.markdown("### 🔬 About the Models")
        st.markdown(f"""
        **{selected_model}** is currently selected.

        - **Logistic Regression:** Linear classifier, best overall performance (98.25%)
        - **Decision Tree:** Interpretable tree-based model
        - **kNN:** Distance-based classification
        - **Naive Bayes:** Probabilistic classifier
        - **Random Forest:** Ensemble of decision trees
        - **XGBoost:** Gradient boosting ensemble (highest recall)
        """)

with col2:
    st.markdown('<p class="sub-header">📈 Model Comparison</p>', unsafe_allow_html=True)

    # Load results
    results_df = load_results()

    if results_df is not None:
        # Display comparison table
        st.dataframe(
            results_df.style.highlight_max(
                axis=0,
                subset=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'],
                color='lightgreen'
            ),
            use_container_width=True,
            height=280
        )

        # Best model info
        best_model = results_df.loc[results_df['Accuracy'].idxmax(), 'Model']
        best_accuracy = results_df['Accuracy'].max()

        st.success(f"🏆 **Best Model:** {best_model} ({best_accuracy:.4f})")

        # Metrics comparison chart
        st.markdown("#### Metrics Comparison")

        fig, ax = plt.subplots(figsize=(10, 6))

        metrics_to_plot = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
        x = np.arange(len(results_df))
        width = 0.12

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, metric in enumerate(metrics_to_plot):
            offset = (i - len(metrics_to_plot) / 2) * width + width / 2
            ax.bar(x + offset, results_df[metric], width, label=metric, color=colors[i], alpha=0.8)

        ax.set_xlabel('Models', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(results_df['Model'], rotation=45, ha='right', fontsize=9)
        ax.legend(loc='lower right', fontsize=8)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim([0.8, 1.0])

        plt.tight_layout()
        st.pyplot(fig)

        # Key insights
        st.markdown("#### 🔑 Key Insights")
        st.markdown(f"""
        - **Highest Accuracy:** {best_model} ({best_accuracy:.4f})
        - **Best AUC:** {results_df.loc[results_df['AUC'].idxmax(), 'Model']} ({results_df['AUC'].max():.4f})
        - **Best Recall:** {results_df.loc[results_df['Recall'].idxmax(), 'Model']} ({results_df['Recall'].max():.4f})
        - All models achieve >91% accuracy
        - Ensemble methods show robust performance
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong>Machine Learning Assignment 2</strong> | BITS Pilani M.Tech (AIML/DSE)</p>
    <p>Breast Cancer Wisconsin Dataset | 569 samples, 30 features, Binary Classification</p>
    <p style='font-size: 0.9rem;'>Developed by Abhi | February 2026</p>
</div>
""", unsafe_allow_html=True)