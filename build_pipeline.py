import joblib
from sklearn.pipeline import Pipeline
from src.transaction_transformer import TransactionFeatureEngineer
from src.stacking_predictor import StackingPredictor

# Load your trained models
xgb_model = joblib.load('models/xgb_model.pkl')
catboost_model = joblib.load('models/catboost_model.pkl')
meta_model = joblib.load('models/lgb_model.pkl')

# Build the full pipeline
full_pipeline = Pipeline([
    ('feature_engineering', TransactionFeatureEngineer()),
    ('stacking_predictor', StackingPredictor(xgb_model, catboost_model, meta_model))
])

# Save the pipeline
joblib.dump(full_pipeline, 'models/full_fraud_pipeline.pkl')
