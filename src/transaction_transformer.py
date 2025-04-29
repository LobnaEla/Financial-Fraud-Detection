from sklearn.base import BaseEstimator, TransformerMixin
from src.feature_engineering_functions import *  # import all your feature functions

class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = extract_device_info_from_transaction(X)
        X = log_transform_transaction_amt(X)
        X = outlier(X)
        X = compute_transaction_date_features(X)
        X = add_productcd_day_features(X)
        X = normalize_d_column_times(X)
        X = label_encode_transaction(X)
        X = missing(X)
        X = coding(X)
        X = generate_transaction_specific_features(X)
        X = coding2(X)
        X = add_features(X)
        X = generate_device_hash_for_transaction(X)
        X = generate_additional_transaction_features(X)
        X = alertfeature_transaction(X)
        X = productid(X)
        X = clean_transaction(X)
        return X
