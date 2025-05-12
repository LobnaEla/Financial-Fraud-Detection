import pandas as pd
import numpy as np
import joblib
from flask import current_app as app

pipeline = joblib.load("models/full_fraud_pipeline.pkl")


def predict_single(input_json):
    import pandas as pd
    import numpy as np
    from flask import current_app as app

    # Convertir tous les None en np.nan
    for key, value in input_json.items():
        if value is None:
            input_json[key] = np.nan

    # Créer le DataFrame d'entrée
    df = pd.DataFrame([input_json])

    # Juste pour logguer avant la prédiction
    app.logger.info("[INFO] Raw input JSON converted to DataFrame:")
    app.logger.info(df.head(1).to_json())

    # Appliquer le pipeline
    preds = pipeline.predict(df)

    # Enregistrer les features finales utilisées (si transform a bien été exécuté)
    if hasattr(pipeline.named_steps["stacking_predictor"], "meta_model"):
        X_transformed = pipeline.named_steps["feature_engineering"].transform(df)
        app.logger.info(
            f"[INFO] Final features sent to model ({X_transformed.shape[1]} columns):"
        )
        app.logger.info(X_transformed.head(1).to_json())

    return {"fraud_prediction": int(preds[0])}
