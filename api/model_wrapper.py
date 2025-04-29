import pandas as pd
import joblib

pipeline = joblib.load('models/full_fraud_pipeline.pkl')

def predict_single(input_json):
    df = pd.DataFrame([input_json])
    preds = pipeline.transform(df)
    return {'fraud_prediction': int(preds[0][0])}
