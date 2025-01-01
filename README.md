# User Behavior Analysis System

This project implements an advanced machine learning pipeline to analyze user behavior, detect anomalies, and provide personalized insights. The system is designed for real-time analysis and integrates various machine learning models such as clustering (K-Means, DBSCAN) and classification (XGBoost).

---

## Features

- **Behavior Analysis**: Analyzes user interaction patterns (e.g., cursor dynamics, keystrokes, session tracking).
- **User Segmentation**: Clustering users into behavior-based segments using K-Means and DBSCAN.
- **Real-Time Predictions**: Predict user feedback categories and churn likelihood using XGBoost.
- **Personalized Insights**: Provide tailored recommendations and user-specific metrics.
- **Data Processing**: Preprocesses and normalizes raw user interaction data.

---

## Prerequisites

Ensure you have the following installed:

1. **Python**: Version 3.8 or later.
2. **Packages**:
   - pandas
   - numpy
   - scikit-learn
   - xgboost
   - joblib
3. **System Requirements**:
   - At least 4 GB of RAM for processing large datasets.

Install the required Python packages:
```bash
pip install -r requirements.txt
