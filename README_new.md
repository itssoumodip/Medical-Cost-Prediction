# Medical Insurance Cost Prediction Using Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comprehensive machine learning project for predicting individual medical insurance costs using demographic and health-related features. This project implements multiple regression algorithms with systematic feature engineering and provides detailed comparative analysis.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Healthcare cost prediction is crucial for insurance companies to set appropriate premiums and assess risk profiles. This project develops and compares five machine learning models to predict medical insurance charges based on:

- **Demographic factors**: Age, gender, region
- **Health indicators**: BMI, smoking status
- **Family factors**: Number of children

### Key Features
- **Comprehensive EDA**: Statistical analysis and visualization
- **Feature Engineering**: Domain-specific interaction features
- **Model Comparison**: 5 regression algorithms with hyperparameter tuning
- **Robust Evaluation**: Cross-validation and multiple metrics
- **Production Ready**: Modular code with preprocessing pipelines

## 📊 Dataset

The project uses a medical insurance dataset containing **1,338 records** with the following features:

| Feature | Type | Description | Range/Values |
|---------|------|-------------|--------------|
| `age` | Numerical | Age of beneficiary | 18-64 years |
| `sex` | Categorical | Gender | male, female |
| `bmi` | Numerical | Body Mass Index | 15.96-53.13 |
| `children` | Numerical | Number of dependents | 0-5 |
| `smoker` | Categorical | Smoking status | yes, no |
| `region` | Categorical | Beneficiary's region | northeast, northwest, southeast, southwest |
| `charges` | Numerical | **Target variable** - Medical costs | $1,121.87 - $63,770.43 |

### Dataset Statistics
- **Mean charges**: $13,270.42
- **Median charges**: $9,382.03
- **Smokers**: 274 (20.5%)
- **Gender distribution**: Nearly balanced (50.5% male, 49.5% female)
- **No missing values**: Complete dataset

## ✨ Features

### 🔍 Exploratory Data Analysis
- Comprehensive statistical analysis
- Distribution visualizations
- Correlation analysis
- Categorical feature analysis with box plots
- Smoking impact analysis

### 🛠️ Feature Engineering
- **Age-BMI Interaction**: `age × bmi` - Combined health risk
- **Age-Smoker Interaction**: Age for smokers, 0 for non-smokers
- **BMI-Smoker Interaction**: BMI for smokers, 0 for non-smokers

### 🤖 Machine Learning Models
1. **Linear Regression** - Baseline interpretable model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization with feature selection
4. **Random Forest** - Ensemble with bagging
5. **Gradient Boosting** - Sequential ensemble learning

### 📈 Model Evaluation
- **Metrics**: R-squared, MAE, RMSE
- **Validation**: 5-fold cross-validation
- **Diagnostics**: Residual plots, actual vs predicted
- **Feature Importance**: For tree-based models

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/itssoumodip/Medical-Cost-Prediction.git
cd Medical-Cost-Prediction
```

2. **Create virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
```

## 💻 Usage

### Basic Usage

1. **Run the complete analysis**
```bash
python predict_costs.py
```

2. **Custom prediction example**
```python
from predict_costs import prepare_prediction_data, best_model_pipeline

# Create new data point
new_data = prepare_prediction_data(
    age=35,
    sex='male',
    bmi=28.5,
    children=2,
    smoker='no',
    region='northeast'
)

# Make prediction
predicted_cost = best_model_pipeline.predict(new_data)
print(f"Predicted cost: ${predicted_cost[0]:.2f}")
```

### Advanced Usage

#### Model Training with Custom Parameters
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# Custom parameter grid
param_grid = {
    'regressor__n_estimators': [100, 200, 300],
    'regressor__max_depth': [10, 20, None],
    'regressor__min_samples_split': [2, 5, 10]
}

# Train with custom parameters
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid_search.fit(X_train, y_train)
```

## 🔬 Methodology

### 1. Data Preprocessing
- **Missing Value Handling**: No missing values detected
- **Feature Scaling**: StandardScaler for numerical features
- **Encoding**: OneHotEncoder for categorical features
- **Train-Test Split**: 80-20 stratified split

### 2. Feature Engineering
```python
# Interaction features based on domain knowledge
data['age_bmi'] = data['age'] * data['bmi']
data['age_smoker'] = data['age'].where(data['smoker'] == 'yes', 0)
data['bmi_smoker'] = data['bmi'].where(data['smoker'] == 'yes', 0)
```

### 3. Model Selection & Optimization
- **GridSearchCV**: Exhaustive hyperparameter search
- **Cross-Validation**: 5-fold CV for robust evaluation
- **Scoring Metric**: R-squared for model comparison

### 4. Evaluation Framework
- **Performance Metrics**: R², MAE, RMSE
- **Statistical Validation**: Cross-validation scores
- **Visual Diagnostics**: Residual analysis, prediction plots

## 📊 Results

### Model Performance Comparison

| Model | R-squared | MAE | RMSE | CV Score (mean ± std) |
|-------|-----------|-----|------|----------------------|
| **Gradient Boosting** | **0.876** | **$2,847** | **$4,286** | **0.863 ± 0.024** |
| **Random Forest** | **0.869** | **$2,912** | **$4,399** | **0.856 ± 0.028** |
| Ridge Regression | 0.751 | $4,124 | $6,072 | 0.748 ± 0.031 |
| Lasso Regression | 0.748 | $4,157 | $6,106 | 0.745 ± 0.033 |
| Linear Regression | 0.745 | $4,167 | $6,143 | 0.742 ± 0.035 |

### Key Insights

1. **Best Model**: Gradient Boosting achieves **87.6% accuracy** (R²)
2. **Feature Importance**: 
   - Smoking status: 42.3%
   - Age: 18.7%
   - BMI: 15.2%
   - Interaction features: 19.7%
3. **Smoking Impact**: 4x cost increase for smokers vs non-smokers
4. **Model Reliability**: Consistent performance across CV folds

### Prediction Examples
- **Non-smoker** (35, male, BMI 28.5, 2 children): **$5,847**
- **Smoker** (same profile): **$23,420** *(+300% increase)*

## 📁 Project Structure

```
Medical-Cost-Prediction/
│
├── 📄 predict_costs.py          # Main analysis script
├── 📊 insurance.csv             # Dataset
├── 📝 README.md                 # Project documentation
├── 📄 paper.tex                 # IEEE format academic paper
├── 📄 requirements.txt          # Python dependencies
│
├── 📁 models/                   # Saved model artifacts (generated)
│   ├── best_model.joblib
│   └── preprocessor.joblib
│
├── 📁 visualizations/           # Generated plots (created during run)
│   ├── distribution_plots.png
│   ├── correlation_matrix.png
│   └── feature_importance.png
│
└── 📁 results/                  # Model evaluation results
    ├── model_comparison.csv
    └── cv_results.json
```

### File Descriptions

- **`predict_costs.py`**: Complete ML pipeline with EDA, preprocessing, training, and evaluation
- **`insurance.csv`**: Medical insurance dataset (1,338 records, 7 features)
- **`paper.tex`**: IEEE format academic paper ready for Overleaf
- **`requirements.txt`**: Python package dependencies

## 🚀 Deployment Options

### 1. Streamlit Web App
```python
import streamlit as st
import joblib

# Load saved model
model = joblib.load('models/best_model.joblib')

# Create web interface
st.title('Medical Cost Predictor')
age = st.slider('Age', 18, 64, 35)
# ... other inputs
prediction = model.predict(input_data)
st.write(f'Predicted Cost: ${prediction[0]:.2f}')
```

### 2. FastAPI REST API
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    age: int
    sex: str
    bmi: float
    # ... other fields

@app.post("/predict")
def predict_cost(request: PredictionRequest):
    # Process and predict
    return {"predicted_cost": prediction}
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Commit changes: `git commit -am 'Add feature'`
5. Push to branch: `git push origin feature-name`
6. Submit a Pull Request

### Areas for Contribution
- Additional feature engineering techniques
- Deep learning model implementations
- Web interface development
- Performance optimizations
- Documentation improvements

## 📚 Academic Paper

This project includes a complete IEEE format academic paper (`paper.tex`) ready for submission to conferences or journals. The paper covers:

- Literature review
- Methodology details
- Experimental results
- Statistical analysis
- Future work directions

**To use in Overleaf**: Copy the contents of `paper.tex` directly into your Overleaf project.

## 🔮 Future Enhancements

- [ ] **Deep Learning Models**: Neural networks for complex pattern recognition
- [ ] **Time Series Analysis**: Cost trend prediction over time
- [ ] **Additional Features**: Chronic conditions, family history
- [ ] **Interpretable AI**: SHAP values for model explainability
- [ ] **Production Deployment**: Docker containerization
- [ ] **Real-time Predictions**: Streaming data processing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Kaggle Medical Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- Scikit-learn community for excellent ML tools
- Open source contributors

## 📞 Contact

**Soumodip Das**
- GitHub: [@itssoumodip](https://github.com/itssoumodip)
- Email: soumodip.das@example.com
- LinkedIn: [Soumodip Das](https://linkedin.com/in/soumodip-das)

---

**⭐ If you found this project helpful, please consider giving it a star!**
