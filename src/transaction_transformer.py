from sklearn.base import BaseEstimator, TransformerMixin
from src.feature_engineering_functions import *  # import all your feature functions
from flask import current_app as app


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        # Utilisation de app.logger pour afficher les messages dans le terminal
        X = extract_device_info_from_transaction(X)
        app.logger.info("[INFO] extract_device_info_from_transaction executed")

        X = log_transform_transaction_amt(X)
        app.logger.info("[INFO] log_transform_transaction_amt executed")
        X = outlier(X)
        app.logger.info("[INFO] outlier executed")

        X = compute_transaction_date_features(X)
        app.logger.info("[INFO] compute_transaction_date_features executed")

        # --------cell26-------
        X = apply_productcd_day_count(X)
        app.logger.info("[INFO] apply_productcd_day_count")

        X = normalize_d_column_times(X)
        app.logger.info("[INFO] normalize_d_column_times executed")

        X = label_encode_transaction(X)
        app.logger.info("[INFO] label_encode_transaction executed")

        X = missing(X)
        app.logger.info("[INFO] missing executed")

        X = coding(X)
        app.logger.info("[INFO] coding executed")

        X = generate_transaction_specific_features(X)
        app.logger.info("[INFO] generate_transaction_specific_features executed")

        X = coding2(X)
        app.logger.info("[INFO] coding2 executed")

        X = add_features(X)
        app.logger.info("[INFO] add_features executed")

        X = generate_device_hash_for_transaction(X)
        app.logger.info("[INFO] generate_device_hash_for_transaction executed")

        X = generate_device_counts_for_transaction(X)
        app.logger.info("[INFO] generate_device_counts_for_transaction executed")

        X = compute_decimal_digit_for_transaction(X)
        app.logger.info("[INFO] compute_decimal_digit_for_transaction executed")

        X = generate_additional_transaction_features(X)
        app.logger.info("[INFO] generate_additional_transaction_features executed")

        X = alertfeature_transaction(X)
        app.logger.info("[INFO] alertfeature_transaction executed")

        X = apply_count_encoding_to_transaction(X)
        app.logger.info("[INFO] apply_count_encoding_to_transaction executed")

        X = add_day_hour_counts(X)
        app.logger.info("[INFO] add_day_hour_counts executed")

        X = process_product_id_for_transaction(X)
        app.logger.info("[INFO] process_product_id_for_transaction executed")

        X = apply_crossover_features_to_transaction(X)
        app.logger.info("[INFO] apply_crossover_features_to_transaction executed")

        X = apply_cross_stats(X)
        app.logger.info("[INFO] apply_cross_stats executed")

        X = clean_transaction(X)
        app.logger.info("[INFO] clean_transaction executed")

        X = apply_common_values_to_transaction(X)
        app.logger.info("[INFO] apply_common_values_to_transaction executed")
        X, diffs = compare_and_clean_columns(X)
        app.logger.info(f"Colonnes manquantes : {diffs['missing_columns']}")
        app.logger.info("done")

        app.logger.info(
            f"removed_unexpected_columns : {diffs['removed_unexpected_columns']}"
        )
        app.logger.info("[INFO] All transformation steps completed.")
        app.logger.info(f"Colonnes finales dans X : {list(X.columns)}")
        return X
