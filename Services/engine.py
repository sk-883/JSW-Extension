
# ─── IMPORTS ─────────────────────────────────────────────────────────────────────
import re
import numpy as np
import diff_match_patch as dmp_module
import html2text

from html2text import html2text
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.util import cos_sim
from IPython.display import display, HTML



# ─── GLOBAL INITIALIZATION ───────────────────────────────────────────────────────
# Pre-load your embedding model so you only pay the load cost once
_EMBED_MODEL = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
_EMBED_WARM = _EMBED_MODEL.encode("warmup")

# Pre-load your diff engine
_DIFF_ENGINE = dmp_module.diff_match_patch()


# ─── UTILITY FUNCTIONS ───────────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def convert_diff_to_html(diffs: list[tuple[int, str]]) -> str:
    html_output = ""
    for op, data in diffs:
        if op == dmp_module.diff_match_patch.DIFF_INSERT:
            html_output += f'&nbsp;<span style="color:red;text-decoration:line-through;">{data}</span>'
        elif op == dmp_module.diff_match_patch.DIFF_DELETE:
            html_output += f'<span style="color:red;">{data}</span>'
        else:
            html_output += f'<span>{data}</span>'
    return html_output


def extract_clauses_from_text(text: str) -> list[str]:
    lines = text.splitlines()
    clauses = []
    current = ""
    pattern = re.compile(r'^(c\d+)\.?\s+(.*)', re.IGNORECASE)

    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            if current:
                clauses.append(current.strip())
            current = f"{match.group(1).upper()}. {match.group(2)}"
        else:
            current += " " + line

    if current:
        clauses.append(current.strip())
    return clauses


def alpha_end_all_lines(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(line + "α" for line in lines)


def extract_with_html2text(html: str) -> str:
    handler = html2text.HTML2Text()
    handler.ignore_links = True
    handler.ignore_emphasis = True
    handler.ignore_images = True
    handler.body_width = 0
    handler.single_line_break = True
    handler.unicode_snob = True

    text = handler.handle(html)
    return text.replace("|", "").replace("~", "").replace("-", "")


def extract_between_markers_from_html(html: str, start_marker: str, end_marker: str) -> str:
    plain = extract_with_html2text(html)
    s = plain.find(start_marker)
    e = plain.find(end_marker)

    if 0 <= s < e:
        return plain[s + len(start_marker):e]
    if s != -1:
        return plain[s + len(start_marker):]
    if e != -1:
        return plain[:e]
    return plain


def format_clause_html(html_text: str) -> str:
    # Re-interpret our α marker as a line break
    return html_text.replace("α", "<br>")


# ─── CORE LOGIC ─────────────────────────────────────────────────────────────────

def compare_clauses_sequentially(
    original_clauses: list[str],
    revised_clauses: list[str],
    window: int = 3,
    threshold: float = 0.5
) -> list[str]:
    if not original_clauses:
        return []
    if not revised_clauses:
        return [
            f"<div style='color:red;'>{normalize_whitespace(c)}</div>"
            for c in original_clauses
        ]

    # Load a fresh model instance per call (matches original logic)
    model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")
    dmp = dmp_module.diff_match_patch()

    orig_norm = [normalize_whitespace(c) for c in original_clauses]
    rev_norm  = [normalize_whitespace(c) for c in revised_clauses]

    orig_embs = model.encode(orig_norm, batch_size=64, show_progress_bar=True)
    rev_embs  = model.encode(rev_norm,  batch_size=64, show_progress_bar=True)

    used = set()
    output = [""] * max(len(orig_norm), len(rev_norm))

    for i, emb in enumerate(orig_embs):
        start = max(0, i - window)
        end   = min(len(rev_embs), i + window + 1)
        sims  = util.cos_sim(emb, rev_embs[start:end])[0]
        jbest = int(np.argmax(sims))
        score = float(sims[jbest])
        ridx  = start + jbest

        if score > threshold and ridx not in used:
            diffs = dmp.diff_main(orig_norm[i], rev_norm[ridx])
            dmp.diff_cleanupSemantic(diffs)
            output[i] = (
                f"<div style='font-family:Courier; font-size:15px; "
                f"white-space:pre-wrap;'>{convert_diff_to_html(diffs)}</div>"
            )
            used.add(ridx)
        else:
            output[i] = (
                f"<div style='font-family:Courier; font-size:15px; "
                f"white-space:pre-wrap; color:red;'>{orig_norm[i]}</div>"
            )

    # Append any unmatched revised clauses as red-striked
    for idx, clause in enumerate(rev_norm):
        if idx not in used:
            strike = (
                f"<div style='font-family:Courier; font-size:15px; "
                f"white-space:pre-wrap; color:red; text-decoration:line-through;'>"
                f"{clause}</div>"
            )
            if idx < len(output) and not output[idx]:
                output[idx] = strike
            else:
                output.append(strike)

    return output


# ─── EXAMPLE USAGE ──────────────────────────────────────────────────────────────




# original_text--->Clauses--->From the USer manual Input.
# initial_marker, final_marker--->From the USer manual Input.
# your_html_content--->Page ka tml-->utomatic scrapping.


# 1) Prepare original clauses
original_text = """
C1.	ALL NEGOS / EVENTUAL FIXTURE TO BE KEPT PRIVATE AND CONFIDENTIAL.
*****
"""
original_text = alpha_end_all_lines(original_text)


original_clauses = extract_clauses_from_text(original_text)
print(original_clauses)

# 2) Extract relevant snippet from HTML
# inject html here from the original req.

 = """<>"""
initial_marker = "Mon 6/16/2025 11:20 AM"
final_marker   = "*****"
relevant_text  = extract_between_markers_from_html(
    your_html_content, initial_marker, final_marker
)

# 3) Clause extraction
revised_clauses = extract_clauses_from_text(relevant_text)
print(revised_clauses)

# 4) Compute window dynamically and compare
window = int(abs(len(revised_clauses) - len(original_clauses)) * 1.5 + 5)
print("Window size:", window)

comparison_results = compare_clauses_sequentially(
    original_clauses, revised_clauses, window
)

# 5) Display as HTML in a notebook
def display_comparison_results(results: list[str]) -> None:
    full_html = (
        "<html><body style='font-family:Courier; font-size:15px; "
        "white-space:pre-wrap;'>"
        + "".join(format_clause_html(r) for r in results)
        + "</body></html>"
    )
    # display(HTML(full_html))
    # print("Comparison results displayed above.")

display_comparison_results(comparison_results)

