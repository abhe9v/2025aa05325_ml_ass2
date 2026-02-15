# Multi-Model Classification Framework

**Machine Learning Assignment 2**  
**Student ID:** 2025aa05325
**Name:** Gaikwad Abhinav Rajaram
**Date:** February 15, 2026

---

## 📊 Project Overview

A professional machine learning framework that implements and compares **6 different classification algorithms** on any binary/multi-class dataset. The system provides automated model training, comprehensive evaluation metrics, and an interactive web interface for real-time predictions.

### 🎯 Key Features

- **Multi-Model Comparison** - Train and compare 6 ML algorithms simultaneously
- **Comprehensive Metrics** - Evaluate models using 6 industry-standard metrics
- **Interactive Web UI** - Upload any dataset and get instant predictions with visualizations
- **Production-Ready Code** - Professional package structure with proper separation of concerns
- **Automated Pipeline** - End-to-end workflow from data loading to model deployment
- **Flexible & Extensible** - Easily add new models or metrics

---

## 🤖 Implemented Algorithms

The framework includes 6 popular classification algorithms covering different ML paradigms:

| Algorithm | Type | Strengths |
|-----------|------|-----------|
| **Logistic Regression** | Linear | Fast, interpretable, probabilistic |
| **Decision Tree** | Tree-based | Non-linear, interpretable |
| **K-Nearest Neighbors** | Instance-based | No training phase, flexible |
| **Naive Bayes** | Probabilistic | Fast, works with small data |
| **Random Forest** | Ensemble | Robust, handles overfitting |
| **XGBoost** | Gradient Boosting | State-of-art, high performance |

---

## 📈 Evaluation Metrics

Each model is evaluated using **6 comprehensive metrics**:

1. **Accuracy** - Overall correctness of predictions
2. **AUC-ROC** - Area Under the ROC Curve (ranking quality)
3. **Precision** - Positive predictive value (TP / TP+FP)
4. **Recall** - Sensitivity, true positive rate (TP / TP+FN)
5. **F1-Score** - Harmonic mean of precision and recall
6. **MCC** - Matthews Correlation Coefficient (balanced measure)

---

## 🎨 Interactive Web Application

### Features

- **📁 Upload Any Dataset** - CSV files with features and optional labels
- **🎯 Model Selection** - Choose from 6 trained algorithms
- **📊 Real-Time Predictions** - Get instant classification results
- **📉 Visual Analytics** - Confusion matrix, performance charts
- **📋 Detailed Reports** - Classification report with per-class metrics
- **💾 Export Results** - Download predictions as CSV

### Screenshots



---

## 🏗️ Architecture & Design

### Professional Package Structure
```
breast-cancer-classification/
├── com/abhi/ml/src/           # Main application package
│   ├── config/                # Configuration management
│   │   └── settings.py        # Centralized settings & constants
│   ├── data/                  # Data handling layer
│   │   ├── loader.py          # Dataset loading utilities
│   │   └── preprocessor.py   # Feature scaling & preprocessing
│   ├── models/                # ML models layer
│   │   ├── base_model.py      # Abstract base class
│   │   ├── logistic_model.py  # Individual model implementations
│   │   ├── decision_tree_model.py
│   │   ├── knn_model.py
│   │   ├── naive_bayes_model.py
│   │   ├── random_forest_model.py
│   │   └── xgboost_model.py
│   ├── evaluation/            # Model evaluation
│   │   └── metrics.py         # Metrics calculation & reporting
│   ├── utils/                 # Shared utilities
│   │   ├── file_handler.py    # File I/O operations
│   │   └── logger.py          # Logging utilities
│   └── main.py               # Training pipeline orchestrator
├── resources/
│   ├── data/                 # Generated datasets
│   └── models/               # Trained model artifacts (.pkl)
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                # Documentation
```

### Design Principles

- **Separation of Concerns** - Clear layer separation (data, models, evaluation, utils)
- **Abstract Base Class Pattern** - Consistent interface for all models
- **Dependency Injection** - Flexible configuration management
- **Single Responsibility** - Each module has one clear purpose
- **DRY Principle** - Reusable components throughout

---

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/abhe9v/2025aa05325_ml_ass2.git
cd 2025aa05325_ml_ass2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Train Models
```bash
# Run complete training pipeline
python -m com.abhi.ml.src.main
```

**Output:**
- Trains all 6 models
- Generates performance metrics
- Saves models to `resources/models/`
- Creates test dataset in `resources/data/`

### Launch Web Application
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## 💡 How It Works

### 1. **Data Pipeline**
```python
DataLoader → Preprocessing → Train/Test Split → Feature Scaling
```

### 2. **Model Training**
```python
For each of 6 models:
    ├── Initialize with config
    ├── Train on scaled data
    ├── Generate predictions
    ├── Calculate 6 metrics
    └── Save model artifact
```

### 3. **Web Interface**
```python
User uploads CSV → Model loads from .pkl → Predictions → Visualizations
```

---

## 📊 Usage Example

### Using Your Own Dataset

**Requirements:**
- CSV format with numeric features
- Optional: Include `target` column for evaluation
- Recommended: ≥500 samples, ≥12 features

**Steps:**

1. **Prepare Your Data**
   - Ensure your CSV has appropriate feature `columns` and `target` column for labels

2. **Upload to Web App**
   - Select model from dropdown
   - Upload CSV file
   - Click "Run Predictions"

3. **View Results**
   - Performance metrics (if labels included)
   - Confusion matrix visualization
   - Prediction confidence scores
   - Download results as CSV

---

## 🔧 Customization

### Add a New Model
Add a new model by creating a class that inherits from `BaseMLModel` and implementing the `build_model` method.
Example:
```python
# Create: com/abhi/ml/src/models/your_model.py
from com.abhi.ml.src.models.base_model import BaseMLModel
from sklearn.svm import SVC

class SVMModel(BaseMLModel):
    def __init__(self):
        super().__init__(model_name="SVM")
    
    def build_model(self):
        return SVC(kernel='rbf', probability=True)
```

### Add a New Metric
Add a new metric by importing it from `sklearn.metrics` and including it in the `calculate_metrics` function.
Example:
```python
# Edit: com/abhi/ml/src/evaluation/metrics.py
from sklearn.metrics import your_metric

def calculate_metrics(...):
    metrics = {
        # ... existing metrics
        'YourMetric': your_metric(y_true, y_pred)
    }
```

---

## 📦 Dependencies
```txt
streamlit>=1.28.0          # Web UI framework
scikit-learn>=1.3.0        # ML algorithms & metrics
numpy>=1.24.0,<2.0.0       # Numerical computing
pandas>=2.0.0              # Data manipulation
matplotlib>=3.9.0          # Visualization
seaborn>=0.12.0            # Statistical visualization
scipy>=1.10.0              # Scientific computing
```

---

## 📈 Performance Optimization

The framework includes:
- **Efficient Preprocessing** - One-time scaling, reusable scaler
- **Model Persistence** - Trained models saved as .pkl files
- **Batch Predictions** - Handle multiple samples efficiently
- **Caching** - Streamlit caching for faster UI response
- **Logging** - Track training progress and errors

---

## 🔬 Technical Highlights

### Code Quality
- ✅ Professional package structure
- ✅ Type hints for better IDE support
- ✅ Comprehensive logging
- ✅ Error handling throughout
- ✅ Modular and maintainable

### Best Practices
- ✅ Separation of concerns
- ✅ DRY (Don't Repeat Yourself)
- ✅ SOLID principles
- ✅ Configuration management
- ✅ Consistent coding style

---

## 🎓 Assignment Compliance

**Requirements Met:**

✅ **6 Classification Algorithms** - Logistic Regression, Decision Tree, kNN, Naive Bayes, Random Forest, XGBoost  
✅ **6 Evaluation Metrics** - Accuracy, AUC, Precision, Recall, F1-Score, MCC  
✅ **Dataset Requirements** - Tested on 569 samples with 30 features  
✅ **GitHub Repository** - Complete source code with professional structure  
✅ **Streamlit Application** - Interactive web UI for predictions  
✅ **Comprehensive Documentation** - README with architecture and usage guide  


---



## 📄 License

This project is submitted as part of academic coursework at BITS Pilani.

---

## 🙏 Acknowledgments

- BITS Pilani faculty for guidance
- scikit-learn team for excellent ML libraries
- Streamlit team for the intuitive web framework
- Open-source ML community

---