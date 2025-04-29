import joblib
import pandas as pd
from src.create_features import create_features

# Load model once when server starts
model = joblib.load('models/fraud_detection_model.pkl')  # Adjust filename if needed

def predict_single(input_json):
    # Convert incoming JSON to DataFrame
    df = pd.DataFrame([input_json])
    
    # Apply feature engineering
    df = create_features(df)

    # Predict probability
    preds = model.predict_proba(df)[:, 1]

    return {'fraud_probability': float(preds[0])}
