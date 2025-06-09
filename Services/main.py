# app/main.py
from flask import Flask, request, jsonify
from config import Config
from model import process_html

def app():
    app = Flask(__name__)
    app.config.from_object(Config)
    # Enable CORS, set up logging, etc.
    
    @app.route('/process', methods=['POST'])
    def process():
        raw = request.get_json().get('html', '')
        processed = process_html(raw)
        return jsonify({'html': processed})

    return app

# If you want to run via `flask run`:
# app = app()
