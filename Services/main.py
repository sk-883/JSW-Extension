
from flask import Flask, request, jsonify, Response
from config import Config
from engine import process_html
# from model import process_html
from flask_cors import CORS

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)
    CORS(app)

    @app.route('/process', methods=['POST'])
    def process():
        ct = request.content_type or ""
        if ct.startswith('application/json'):
            # old behavior
            html = request.get_json(force=True).get('html', '')
        elif ct.startswith('text/html'):
            # read raw HTML body
            html = request.get_data(as_text=True)
        else:
            return jsonify({"error": f"Unsupported Content-Type: {ct}"}), 415

        processed = process_html(html)

        # If you want to return JSON-wrapped HTML:
        # return jsonify({'html': processed})

        # OR return raw HTML directly:
        return Response(processed, content_type='text/html; charset=utf-8')

    return app

app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=Config.DEBUG)
