from flask import Blueprint, request, jsonify, current_app as app
import numpy as np

# Importez votre module de prédiction
from .model_wrapper import predict_single

api_blueprint = Blueprint("api", __name__, url_prefix="/api")


@api_blueprint.route("/predict", methods=["POST"])
def predict():
    try:
        # Récupérer les données JSON
        data = request.json

        if not data:
            return jsonify({"error": "Aucune donnée fournie"}), 400

        # Convertir les null en np.nan pour le modèle (déjà géré dans predict_single)
        app.logger.info(f"Données reçues: {data}")

        # Faire la prédiction
        result = predict_single(data)

        # Retourner le résultat
        return jsonify(result)

    except Exception as e:
        app.logger.error(f"Erreur lors de la prédiction: {str(e)}")
        return jsonify({"error": str(e)}), 500
