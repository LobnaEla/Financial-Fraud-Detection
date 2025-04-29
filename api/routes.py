from flask import Blueprint, request, jsonify
from api.model_wrapper import predict_single

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        prediction = predict_single(input_data)
        return jsonify(prediction)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
