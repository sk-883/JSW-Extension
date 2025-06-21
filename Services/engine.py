# engine.py

# ─── IMPORTS & GLOBALS ───────────────────────────────────────────────────────────
import re
import numpy as np
import diff_match_patch as dmp_module
import html2text
from sentence_transformers import SentenceTransformer, util

# Pre-load heavy objects just once
_EMBED_MODEL = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
_DIFF_ENGINE = dmp_module.diff_match_patch()


# ─── EXISTING UTILITIES ──────────────────────────────────────────────────────────
def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def convert_diff_to_html(diffs):
    html = ""
    for op, data in diffs:
        if op == dmp_module.diff_match_patch.DIFF_INSERT:
            html += f'&nbsp;<span style="color:red;text-decoration:line-through;">{data}</span>'
        elif op == dmp_module.diff_match_patch.DIFF_DELETE:
            html += f'<span style="color:red;">{data}</span>'
        else:
            html += f'<span>{data}</span>'
    return html

def extract_clauses_from_text(text: str):
    lines = text.splitlines()
    clauses, current = [], ""
    pattern = re.compile(r'^(c\d+)\.?\s+(.*)', re.IGNORECASE)
    for l in lines:
        l = l.strip()
        if not l: continue
        m = pattern.match(l)
        if m:
            if current:
                clauses.append(current.strip())
            current = f"{m.group(1).upper()}. {m.group(2)}"
        else:
            current += " " + l
    if current:
        clauses.append(current.strip())
    return clauses

def alpha_end_all_lines(text: str) -> str:
    return "\n".join(line + "α" for line in text.splitlines())

def extract_with_html2text(html: str) -> str:
    h = html2text.HTML2Text()
    h.ignore_links = h.ignore_emphasis = h.ignore_images = True
    h.body_width = 0
    h.single_line_break = True
    h.unicode_snob = True
    return h.handle(html).replace("|","").replace("~","").replace("-","")

def extract_between_markers_from_html(html: str, start: str, end: str) -> str:
    plain = extract_with_html2text(html)
    s, e = plain.find(start), plain.find(end)
    if 0 <= s < e:
        return plain[s+len(start):e]
    if s != -1:
        return plain[s+len(start):]
    if e != -1:
        return plain[:e]
    return plain

def format_clause_html(html_text: str) -> str:
    return html_text.replace("α", "<br>")


# ─── WRAPPER FUNCTION ────────────────────────────────────────────────────────────

def run_comparison_engine(
    your_html_content: str,
    initial_marker:    str,
    final_marker:      str,
    original_text:     str
) -> str:
    """
    1) Marks up original_text with α → split into original_clauses
    2) Extracts the snippet between initial_marker & final_marker from your_html_content
    3) Splits that into revised_clauses
    4) Runs the embed+diff logic to produce HTML snippets
    5) Wraps everything in a single HTML string and returns it
    """
    # Prepare original clauses
    marked_orig = alpha_end_all_lines(original_text)
    orig_clauses = extract_clauses_from_text(marked_orig)

    # Pull out just the relevant part from the incoming HTML
    snippet = extract_between_markers_from_html(
        your_html_content, initial_marker, final_marker
    )
    rev_clauses = extract_clauses_from_text(snippet)

    # Determine matching window
    window = int(abs(len(rev_clauses) - len(orig_clauses)) * 1.5 + 5)

    # Embed & diff sequentially
    model = _EMBED_MODEL
    dmp   = _DIFF_ENGINE

    # Normalize & embed
    orig_norm = [normalize_whitespace(c) for c in orig_clauses]
    rev_norm  = [normalize_whitespace(c) for c in rev_clauses]
    orig_embs = model.encode(orig_norm, batch_size=64, show_progress_bar=False)
    rev_embs  = model.encode(rev_norm,  batch_size=64, show_progress_bar=False)

    used = set()
    results = [""] * max(len(orig_norm), len(rev_norm))

    for i, emb in enumerate(orig_embs):
        start, end = max(0, i-window), min(len(rev_embs), i+window+1)
        sims   = util.cos_sim(emb, rev_embs[start:end])[0]
        best   = int(np.argmax(sims))
        score  = float(sims[best])
        ridx   = start + best

        if score > 0.5 and ridx not in used:
            diffs = dmp.diff_main(orig_norm[i], rev_norm[ridx])
            dmp.diff_cleanupSemantic(diffs)
            results[i] = (
                f"<div style='font-family:Courier; font-size:15px; white-space:pre-wrap;'>"
                f"{convert_diff_to_html(diffs)}</div>"
            )
            used.add(ridx)
        else:
            results[i] = (
                f"<div style='font-family:Courier; font-size:15px; white-space:pre-wrap;"
                f" color:red;'>{orig_norm[i]}</div>"
            )

    # Any unmatched revised clauses → strikethrough
    for idx, clause in enumerate(rev_norm):
        if idx not in used:
            strike = (
                f"<div style='font-family:Courier; font-size:15px; "
                f"white-space:pre-wrap;color:red;text-decoration:line-through;'>"
                f"{clause}</div>"
            )
            if idx < len(results) and not results[idx]:
                results[idx] = strike
            else:
                results.append(strike)

    # Wrap the whole thing in <html>…</html> and return
    body = "".join(format_clause_html(r) for r in results)
    return f"<html><body style='font-family:Courier; font-size:15px;'>{body}</body></html>"
