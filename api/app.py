from flask import Flask, jsonify
import logging
from api.routes import api_blueprint

app = Flask(__name__)
app.register_blueprint(api_blueprint)
app.logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
app.logger.addHandler(handler)

app.config['DEBUG'] = True
@app.route('/')
def home():
    app.logger.info("This is an info log")
    return "Check the terminal for logs"
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
