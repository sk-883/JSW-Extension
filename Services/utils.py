# app/utils.py
import logging

def setup_logging(level=None):
    lvl = level or logging.INFO
    logging.basicConfig(
        format='%(asctime)s %(levelname)s:%(message)s',
        level=lvl
    )

def sanitize_html(html: str) -> str:
    # remove dangerous tags, scripts, etc.
    return html.replace('<script', '&lt;script')
