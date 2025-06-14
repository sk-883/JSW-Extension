# import re
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import diff_match_patch as dmp_module
# from originalText import original_text
# from bs4 import BeautifulSoup
# from htmlContent import your_html_content
# from IPython.display import display, HTML
# def process_html(html: str) -> str:
#     # 1. parse/extract features
#     # 2. run model inference
#     # 3. wrap results in HTML
#     return f"{html}"

# def extract_clauses_from_text(text):
#     # Match patterns like "1. ...", "2. ...", all the way until the next such pattern or end of string
#     pattern = re.compile(r'(\d+\..*?)(?=\d+\.\s*|$)', re.DOTALL)
#     clauses = pattern.findall(text)
#     return [clause.strip() for clause in clauses if clause.strip()]

# # Example: If you have the full text of original document, parse it

# original_clauses = extract_clauses_from_text(original_text)
# # print(original_clauses)




# # Your HTML content (replace this with your actual HTML string)


# # Parse the HTML with BeautifulSoup
# soup = BeautifulSoup(your_html_content, 'html.parser')

# # Extract all text, removing tags
# text_content = soup.get_text(separator=' ', strip=True)

# # Print the extracted text
# revised_clauses = extract_clauses_from_text(text_content)
# print(text_content)
# print(revised_clauses)




# def normalize_whitespace(text):
#     return re.sub(r'\s+', ' ', text).strip()

# def convert_diff_to_html(diff):
#     html_output = ""
#     for op, data in diff:
#         if op == dmp_module.diff_match_patch.DIFF_INSERT:
#             html_output += f'&nbsp<span style="color:red;text-decoration:line-through;">{data}</span>'
#         elif op == dmp_module.diff_match_patch.DIFF_DELETE:
#             html_output += f'<span style="color:red;">{data}</span>'
#         else:
#             html_output += f'<span>{data}</span>'
#     return html_output

# def compare_clauses_sequentially(original_clauses, revised_clauses, window=5, threshold=0.05):
#     embedder = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")  # Fast + accurate
#     dmp = dmp_module.diff_match_patch()

#     if not original_clauses or not revised_clauses:
#         return [f"<div style='color:red;'>{normalize_whitespace(c)}</div>" for c in original_clauses]

#     comparison_results = []
#     revised_embeddings = embedder.embed_documents(revised_clauses)
#     used_revised = set()

#     for i, orig_clause in enumerate(original_clauses):
#         orig_emb = embedder.embed_query(orig_clause)

#         # Define matching window
#         start = max(0, i - window)
#         end = min(len(revised_clauses), i + window + 1)
#         candidates = revised_clauses[start:end]
#         candidate_embeddings = revised_embeddings[start:end]

#         if not candidates:
#             comparison_results.append(
#                 f"<div style='font-family:Courier; font-size:15px; white-space:pre-wrap; color:red;'>{normalize_whitespace(orig_clause)}</div>"
#             )
#             continue

#         # Compute cosine similarity
#         sims = cosine_similarity([orig_emb], candidate_embeddings)[0]
#         best_idx = int(np.argmax(sims))
#         best_score = sims[best_idx]
#         actual_rev_idx = start + best_idx
#         rev_clause = revised_clauses[actual_rev_idx]

#         if best_score > threshold and actual_rev_idx not in used_revised:
#             used_revised.add(actual_rev_idx)
#             orig_norm = normalize_whitespace(orig_clause)
#             rev_norm = normalize_whitespace(rev_clause)
#             diffs = dmp.diff_main(orig_norm, rev_norm)
#             dmp.diff_cleanupSemantic(diffs)
#             html_result = convert_diff_to_html(diffs)
#             comparison_results.append(
#                 f"<div style='font-family:Courier; font-size:15px; white-space:pre-wrap;'>{html_result}</div>"
#             )
#         else:
#             orig_norm = normalize_whitespace(orig_clause)
#             comparison_results.append(
#                 f"<div style='font-family:Courier; font-size:15px; white-space:pre-wrap; color:red;'>{orig_norm}</div>"
#             )

#     return comparison_results
# #BAAI/bge-m3 # First and best until now at 0.45 similarity parameter
# #intfloat/e5-large-v2 not that good





# def format_clause_html(html_text):
#     # 1. Handle ## → double line break
#     html_text = html_text.replace('##', '<br><br>')

#     # 2. Handle # → single line break
#     html_text = html_text.replace('#', '<br>')

#     return html_text

# def display_comparison_results(comparison_results):
#     full_html = "<html><body style='font-family:Courier; font-size:15px; white-space:pre-wrap;'>"
#     full_html += "<br>".join(format_clause_html(clause_html) for clause_html in comparison_results)
#     full_html += "</body></html>"

#     display(HTML(full_html))
#     print("Comparison results displayed above.")

# # Example usage (assuming comparison_results is already generated)
# comparison_results = compare_clauses_sequentially(original_clauses, revised_clauses)
# display_comparison_results(comparison_results)


# html_processor.py

import re
import numpy as np
import diff_match_patch as dmp_module
from bs4 import BeautifulSoup
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

# ─── GLOBAL INITIALIZATION ─────────────────────────────────────────────────

# 1) Pre-load your embedder so you don't pay the model-load cost per request
EMBEDDER = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")
# warm up
_EMBED_WARM = EMBEDDER.embed_query("warmup")

# 2) Pre-load your diff engine
DIFF_ENGINE = dmp_module.diff_match_patch()

# ─── UTILITIES ────────────────────────────────────────────────────────────────

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def extract_clauses_from_text(text: str) -> list[str]:
    pattern = re.compile(r'(\d+\..*?)(?=\d+\.\s*|$)', re.DOTALL)
    return [clause.strip() for clause in pattern.findall(text) if clause.strip()]

def convert_diff_to_html(diffs: list[tuple[int,str]]) -> str:
    out = []
    for op, data in diffs:
        if op == dmp_module.diff_match_patch.DIFF_INSERT:
            out.append(f'&nbsp;<span style="color:red;text-decoration:line-through;">{data}</span>')
        elif op == dmp_module.diff_match_patch.DIFF_DELETE:
            out.append(f'<span style="color:red;">{data}</span>')
        else:
            out.append(f'<span>{data}</span>')
    return "".join(out)

def format_clause_html(html_text: str) -> str:
    return html_text.replace('##', '<br><br>').replace('#', '<br>')

# ─── CORE LOGIC ────────────────────────────────────────────────────────────────

def process_html(html: str,
                 window: int = 5,
                 threshold: float = 0.05) -> str:
    """
    1) Parse the raw `html` string, extract text clauses
    2) Embed and compare against original_text clauses
    3) Return a single HTML string showing diffs and highlights
    """
    # --- 1) Extract text and clauses from incoming HTML
    soup = BeautifulSoup(html, 'html.parser')
    text_content = soup.get_text(separator=' ', strip=True)
    revised_clauses = extract_clauses_from_text(text_content)

    # --- 2) Original clauses (imported from your module)
    from originalText import original_text
    original_clauses = extract_clauses_from_text(original_text)

    # early exit
    if not original_clauses:
        return "<div style='color:red;'>No original clauses to compare.</div>"

    # embed all revised clauses once
    revised_embeddings = EMBEDDER.embed_documents(revised_clauses)

    results = []
    used = set()
    for i, orig in enumerate(original_clauses):
        orig_emb = EMBEDDER.embed_query(orig)

        # find candidates in a window
        start = max(0, i - window)
        end   = min(len(revised_clauses), i + window + 1)
        cands = revised_clauses[start:end]
        cand_embs = revised_embeddings[start:end]

        if not cands:
            # no candidate → highlight missing
            html = normalize_whitespace(orig)
            results.append(f"<div style='color:red;'>{html}</div>")
            continue

        sims = cosine_similarity([orig_emb], cand_embs)[0]
        j_best = int(np.argmax(sims))
        score  = sims[j_best]
        rev_i  = start + j_best
        rev    = revised_clauses[rev_i]

        if score > threshold and rev_i not in used:
            used.add(rev_i)
            diffs = DIFF_ENGINE.diff_main(normalize_whitespace(orig),
                                          normalize_whitespace(rev))
            DIFF_ENGINE.diff_cleanupSemantic(diffs)
            diff_html = convert_diff_to_html(diffs)
            results.append(f"<div style='white-space:pre-wrap;'>{diff_html}</div>")
        else:
            # treat as deleted / unmatched
            html = normalize_whitespace(orig)
            results.append(f"<div style='white-space:pre-wrap; color:red;'>{html}</div>")

    # ─── 3) Wrap up into a full HTML page
    body = "<br>".join(format_clause_html(r) for r in results)
    return f"<html><body style='font-family:Courier; font-size:15px;'>{body}</body></html>"
