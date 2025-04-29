import pandas as pd
import joblib
from flask import current_app as app
pipeline = joblib.load('models/full_fraud_pipeline.pkl')
input_json = {...}
def predict_single(input_json):
    df = pd.DataFrame([input_json])
    app.logger.info("This is an info log2")
    preds = pipeline.transform(df)
    return {'fraud_prediction': int(preds[0][0])}