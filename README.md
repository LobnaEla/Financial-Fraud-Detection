# 💳 Financial Fraud Detection

This project aims to detect fraudulent online financial transactions using advanced machine learning models. Built on the IEEE-CIS Fraud Detection dataset, the solution is designed to identify anomalies with high precision and usability.

---

## 🎯 Project Goals

- Detect fraud in financial transactions with minimal false positives/negatives
- Handle imbalanced datasets using robust evaluation metrics (F1-score, AUC)
- Build a modular and explainable feature pipeline
- Deploy the model through a web server and user-friendly interface

---

## 🧠 Project Highlights

- **Data preprocessing**: Merging identity and transaction datasets, handling missing values, normalizing variables
- **Feature engineering**:
  - Outlier detection
  - Time-based fraud alerts
  - Device and user profiling
  - Behavioral indicators (e.g., decimal precision, email mismatch)
- **Model Architecture**
  - Base Models: XGBoost, CatBoost
  - Meta-Model: LightGBM (Stacking approach)
  - Validation: 5-fold cross-validation
- **Performance**:
  - F1-score: **0.89**
  - Recall: **0.87**
  - Precision: **0.91**
  - AUC: **0.98**

---

## 📁 Project Folder Overview

- `api/`: Contains Flask server logic and prediction endpoint
- `models/`: Trained models (XGBoost, CatBoost, LightGBM)
- `notebooks/`: Jupyter notebooks for EDA, modeling, and deployment
- `src/`: Preprocessing scripts, encoders, and transformation logic
- `static/` & `templates/`: Frontend interface (HTML/CSS/JS)
- `build_pipeline.py`: Builds the data processing pipeline
- `requirements.txt`: Python dependencies

---

## 📚 Dataset

**IEEE-CIS Fraud Detection Dataset**  
🔗 Available on [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)

