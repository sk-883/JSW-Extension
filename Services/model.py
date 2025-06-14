# app/model.py
import os
# from torch import load, ...
MODEL_PATH = os.getenv('MODEL_PATH', 'models/default.pt')

# load your model once on import
# model = load(MODEL_PATH)

def process_html(html: str) -> str:
    # 1. parse/extract features
    # 2. run model inference
    # 3. wrap results in HTML
    return f"{html}"
