# main.py
import base64
from flask import Flask, request, Response, jsonify
from config import Config
from engine import run_comparison_engine
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    app.config['MAX_CONTENT_LENGTH'] = 200 * 1024 * 1024  # 200 MiB
    CORS(app)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        return jsonify({"error": "Payload too large (over 200 MiB)"}), 413

    @app.route('/process', methods=['POST'])
    def process():
        ct = request.content_type or ''
        if not ct.startswith('application/json'):
            return jsonify({"error": f"Unsupported Content-Type: {ct}"}), 415

        data = request.get_json(force=True)

        # decode the two blobs
        try:
            html_in  = base64.b64decode(data.get('html_b64','')).decode('utf-8')
            # raw_text = base64.b64decode(data.get('text_b64','')).decode('utf-8')
        except Exception as err:
            return jsonify({"error": "Base64 decode failed", "detail": str(err)}), 400

        start_marker = data.get('initial_marker','')
        end_marker   = data.get('final_marker','')
        orig_text    = data.get('original_text','')

        # now pass both into your engine
        output_html = run_comparison_engine(
            html_in,
            start_marker,
            end_marker,
            orig_text
        )

        return Response(output_html, content_type='text/html; charset=utf-8')

    return app

app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=Config.DEBUG)
