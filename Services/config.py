# app/config.py
import os

class Config:
    DEBUG = os.getenv('DEBUG', 'false').lower() == 'true'
    MODEL_PATH = os.getenv('MODEL_PATH', '/path/to/model')
    # add other settings (e.g. logging level, timeouts)


# RPS_5P34
# Conversion container