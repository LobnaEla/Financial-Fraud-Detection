import os
import json
import numpy as np
from flask import Flask, jsonify, render_template
from flask_cors import CORS
import logging
from api.routes import api_blueprint

# Configurer les chemins pour les templates et fichiers statiques
template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates"))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../static"))


# Classe pour gérer la sérialisation des valeurs NumPy et NaN
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isnan(obj):
                return None  # NaN devient null en JSON
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)
CORS(app)  # Activer CORS pour toutes les routes
app.register_blueprint(api_blueprint)

# Configuration du JSONEncoder personnalisé pour l'application
app.json_encoder = NumpyEncoder

# Configuration du logging
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

app.config["DEBUG"] = True


@app.route("/")
def home():
    app.logger.info("Page d'accueil chargée")
    return render_template("index.html")


if __name__ == "__main__":
    # Afficher les chemins pour le débogage
    app.logger.info(f"Chemin des templates: {template_dir}")
    app.logger.info(f"Chemin des fichiers statiques: {static_dir}")
    app.run(debug=True, host="0.0.0.0", port=5000)
