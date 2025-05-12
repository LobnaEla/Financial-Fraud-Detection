from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np


class StackingPredictor(BaseEstimator, TransformerMixin):
    def __init__(self, xgb_model, catboost_model, meta_model):
        self.xgb_model = xgb_model
        self.catboost_model = catboost_model
        self.meta_model = meta_model

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        xgb_pred = self.xgb_model.predict_proba(X)[:, 1]
        cat_pred = self.catboost_model.predict_proba(X)[:, 1]
        meta_features = np.column_stack((xgb_pred, cat_pred))

        # Logging to Flask
        from flask import current_app as app

        app.logger.info(f"xgb_pred: {xgb_pred}")
        app.logger.info(f"cat_pred: {cat_pred}")
        app.logger.info(f"meta_features shape: {meta_features.shape}")

        final_pred = self.meta_model.predict(meta_features)
        return final_pred
