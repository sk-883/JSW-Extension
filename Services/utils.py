# app/utils.py

import logging
import re
from html import escape as html_escape
from flask import jsonify  # or from fastapi.responses import JSONResponse

def setup_logging(level: str = None) -> None:
    """
    Configure the root logger to output timestamps and log levels.
    Call this once at application startup.
    """
    lvl = getattr(logging, level or "INFO", logging.INFO)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=lvl
    )

def sanitize_html(raw_html: str) -> str:
    """
    Basic sanitization: escape any <script> tags and 
    remove on* attributes to prevent XSS in returned HTML.
    """
    # Escape script tags
    cleaned = re.sub(r"<\s*script[^>]*>.*?<\s*/\s*script\s*>",
                     "", raw_html, flags=re.IGNORECASE|re.DOTALL)
    # Remove inline event handlers: onclick="...", onload='...'
    cleaned = re.sub(r'\son\w+="[^"]*"', "", cleaned)
    cleaned = re.sub(r"\son\w+='[^']*'", "", cleaned)
    return cleaned

def error_response(message: str, status_code: int = 400):
    """
    Return a JSON error payload with the given HTTP status.
    Works in Flask; for FastAPI you’d return JSONResponse.
    """
    payload = {"error": message}
    return jsonify(payload), status_code

def load_model(path: str):
    """
    Example wrapper for loading your ML model once.
    Adjust to your framework (torch.load, tensorflow.keras.models.load_model, etc.).
    """
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    # from torch import load as torch_load
    # return torch_load(path, map_location="cpu")
    # placeholder:
    return {"model_path": path}
