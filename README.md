# 📊 Data Science & ML Pipeline

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange.svg)](https://scikit-learn.org/)

> End-to-end Data Science and Machine Learning Pipeline with automated feature engineering, model training, and deployment

## 🌟 Overview

A production-grade machine learning pipeline that automates the entire data science workflow from data ingestion to model deployment. Built with best practices for scalability, reproducibility, and maintainability.

## ✨ Key Features

- **🔄 Automated Pipeline**: End-to-end workflow automation from raw data to predictions
- **🛠️ Feature Engineering**: Automated feature selection, transformation, and encoding
- **🧪 Model Training**: Multiple algorithms with hyperparameter tuning
- **📊 Experiment Tracking**: MLflow integration for experiment management
- **🚀 Model Deployment**: REST API with FastAPI for real-time predictions
- **📈 Monitoring**: Data drift detection and model performance tracking

## 🏗️ Architecture

```
Data Ingestion
     ↓
Data Validation
     ↓
Data Preprocessing
     ↓
Feature Engineering
     ↓
Model Training
     │
     ├─── RandomForest
     ├─── XGBoost
     ├─── LightGBM
     └─── Neural Networks
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
Model Selection
     ↓
Model Deployment
     ↓
Monitoring & Logging
```

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
Docker (optional)
MLflow server (for experiment tracking)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/sharandeepreddy/Data-Science-ML-Pipeline.git
cd Data-Science-ML-Pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
```

### Quick Start

```python
from pipeline import MLPipeline
from config import Config

# Initialize pipeline
config = Config()
pipeline = MLPipeline(config)

# Train model
pipeline.train(
    data_path="data/train.csv",
    target_column="target"
)

# Make predictions
predictions = pipeline.predict("data/test.csv")
```

## 💻 Usage Examples

### 1. Data Preprocessing

```python
from pipeline.preprocessing import DataPreprocessor

preprocessor = DataPreprocessor()
X_processed = preprocessor.fit_transform(X_train)
```

### 2. Feature Engineering

```python
from pipeline.features import FeatureEngineer

fe = FeatureEngineer()
X_features = fe.create_features(X_processed)
```

### 3. Model Training with Hyperparameter Tuning

```python
from pipeline.models import ModelTrainer

trainer = ModelTrainer(model_type="xgboost")
trainer.train_with_tuning(X_train, y_train)
```

### 4. Model Deployment

```bash
# Start API server
python api/serve.py

# Make prediction request
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [1.2, 3.4, 5.6]}'
```

## 📊 Pipeline Stages

### 1. Data Ingestion
- Multiple data source support (CSV, SQL, APIs)
- Data validation and quality checks
- Automated data versioning

### 2. Feature Engineering
- Numerical feature scaling and transformation
- Categorical encoding (One-Hot, Label, Target)
- Feature selection (RFE, SelectKBest, RFECV)
- Feature interaction generation
- Time-based feature extraction

### 3. Model Training
- Algorithm selection
- Cross-validation
- Hyperparameter optimization (GridSearch, RandomSearch, Optuna)
- Ensemble methods

### 4. Model Evaluation
- Classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC)
- Regression metrics (MAE, MSE, RMSE, R²)
- Confusion matrix visualization
- Feature importance analysis

### 5. Deployment
- FastAPI REST API
- Docker containerization
- Model versioning
- A/B testing support

## 🛠️ Tech Stack

**Core Libraries:**
- **Data Processing**: pandas, numpy, polars
- **ML Frameworks**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Deep Learning**: TensorFlow, PyTorch (optional)
- **Feature Engineering**: category_encoders, feature-engine

**MLOps & Deployment:**
- **Experiment Tracking**: MLflow, Weights & Biases
- **API Framework**: FastAPI
- **Containerization**: Docker, Docker Compose
- **Monitoring**: Evidently AI, Great Expectations

**Visualization:**
- matplotlib, seaborn, plotly

## 📁 Project Structure

```
Data-Science-ML-Pipeline/
├── data/
│   ├── raw/
│   ├── processed/
│   └── features/
├── pipeline/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── deployment.py
├── models/
│   ├── trained/
│   └── checkpoints/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_model_training.ipynb
├── api/
│   ├── serve.py
│   └── schemas.py
├── tests/
├── config/
│   └── config.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pipeline --cov-report=html

# Run specific test module
pytest tests/test_preprocessing.py
```

## 🐳 Docker Deployment

```bash
# Build image
docker build -t ml-pipeline .

# Run container
docker run -p 8000:8000 ml-pipeline

# Or use docker-compose
docker-compose up
```

## 📈 Model Performance

### Classification Example (Binary)
- **Accuracy**: 94.2%
- **Precision**: 92.8%
- **Recall**: 95.1%
- **F1-Score**: 93.9%
- **ROC-AUC**: 0.97

### Regression Example
- **MAE**: 0.23
- **RMSE**: 0.31
- **R² Score**: 0.89

## 📚 Documentation

For detailed documentation, please visit:
- [Pipeline Documentation](docs/pipeline.md)
- [API Documentation](docs/api.md)
- [Deployment Guide](docs/deployment.md)

## 🔧 Configuration

```yaml
# config.yaml
data:
  train_path: "data/train.csv"
  test_path: "data/test.csv"
  target_column: "target"

preprocessing:
  handle_missing: "mean"
  scale_features: true
  encode_categorical: "onehot"

model:
  algorithm: "xgboost"
  cv_folds: 5
  optimize: true

deployment:
  api_host: "0.0.0.0"
  api_port: 8000
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Resources

- [scikit-learn Documentation](https://scikit-learn.org/)
- [MLflow Documentation](https://mlflow.org/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📧 Contact

**Sharandeep Reddy** - [@sharandeepreddy](https://github.com/sharandeepreddy)

Project Link: [https://github.com/sharandeepreddy/Data-Science-ML-Pipeline](https://github.com/sharandeepreddy/Data-Science-ML-Pipeline)

---

⭐ Star this repository if you find it helpful!
