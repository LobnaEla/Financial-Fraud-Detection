# 💳 Financial Fraud Detection using Machine Learning

This project focuses on detecting fraudulent credit card transactions using machine learning. It includes a complete ML pipeline with data preprocessing, advanced feature engineering, model training and evaluation, a RESTful Flask API, and a simple web interface for testing transactions.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Screenshots](#screenshots)
- [License](#license)

---

## 🧠 Project Overview

Credit card fraud is a serious and growing problem in the financial world. The objective of this project is to:

- Detect suspicious credit card transactions using a supervised ML model.
- Apply **feature engineering techniques** to extract meaningful patterns from raw transaction data.
- Provide an interface to interact with the model via a Flask API and a web interface.
- Evaluate model performance using metrics like precision, recall, F1-score, and ROC AUC.

---

## ⚙️ Tech Stack

- **Python**
- **scikit-learn**: for preprocessing and modeling
- **Flask**: backend server to expose the model via an API
- **HTML/CSS + JS**: simple web interface
- **Pandas / NumPy**: data manipulation
- **Matplotlib / Seaborn**: data visualization

---

## 🧰 Feature Engineering

To improve model performance, the project includes several feature engineering steps:

- **Date & Time transformations**: extracting hour of day, weekday, etc.
- **User behavior metrics**: e.g., average transaction amount per user, transaction frequency.
- **Device/browser aggregation features**
- **Transaction pattern analysis**: such as amount relative to user mean, time since last transaction, etc.
- **Encoding & scaling**: categorical encoding and standardization using a custom scikit-learn pipeline.

These engineered features helped enhance the model’s ability to detect subtle fraud patterns.
