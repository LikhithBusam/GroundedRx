"""Drug-identity matching: does a query name a specific medication, and does
a retrieved chunk actually mention it? Ported from GroundedRx_Colab.ipynb's
"Safety Improvement: Drug Identity Gate" cell.

Exact/near-exact string matching only (case-insensitive substring + a tight
difflib cutoff for spelling variants) -- deliberately NOT embedding
similarity: drug identity is an exact constraint, not a fuzzy one. Pure
functions, no GPU/model dependency, so this whole module is unit-testable
with plain strings and dicts.
"""

import difflib
import re
from typing import List

# ── Drug registry: generic -> {brand names, spelling variants, Arabic form} ──
# ponytail: SEED LIST, not exhaustive over the full 464-document corpus.
# Extend as new drugs are encountered live.
DRUG_REGISTRY = {
    "lisinopril": {
        "en": ["lisinopril", "linopril"],
        "ar": ["ليزينوبريل", "لينوبريل"],
    },
    "enalapril": {
        "en": ["enalapril"],
        "ar": ["إينالابريل", "انالابريل"],
    },
    "ramipril": {
        "en": ["ramipril"],
        "ar": ["راميبريل"],
    },
    "levonorgestrel_ethinylestradiol": {  # Logynon's active ingredients
        "en": ["logynon"],
        "ar": ["لوجينون"],
    },
    "desogestrel_ethinylestradiol": {  # Marvelon's active ingredients
        "en": ["marvelon"],
        "ar": ["مارفيلون"],
    },
    "batlor": {
        "en": ["batlor"],
        "ar": ["باتلور"],
    },
}

# Flat lookup: any known surface form -> its canonical drug key
_SURFACE_TO_DRUG = {
    name.lower(): drug
    for drug, forms in DRUG_REGISTRY.items()
    for names in forms.values()
    for name in names
}


def extract_drug_identity(query: str, language: str) -> dict:
    """
    Find a known medication name in `query`, if any.

    Returns {"drug": <canonical key> or None, "all_drugs": [...]} --
    "all_drugs" covers the multiple-medications-in-one-query case; "drug"
    is the first match, for the common single-drug case.
    """
    q_lower = query.lower()
    found = []

    # 1. exact substring match against every known surface form
    for surface, drug in _SURFACE_TO_DRUG.items():
        if surface in q_lower or surface in query:  # `query` kept too: Arabic has no case
            found.append(drug)

    # 2. spelling-variant fallback: only if nothing matched exactly, check
    # each token against the registry with a tight similarity cutoff (0.85)
    # -- catches e.g. "lisinoprol" typos without opening the door to
    # unrelated-but-similar-looking drug names.
    if not found:
        tokens = re.findall(r"[^\W\d_]+", query, flags=re.UNICODE)
        surfaces = list(_SURFACE_TO_DRUG.keys())
        for tok in tokens:
            match = difflib.get_close_matches(tok.lower(), surfaces, n=1, cutoff=0.85)
            if match:
                found.append(_SURFACE_TO_DRUG[match[0]])

    found = list(dict.fromkeys(found))  # de-dup, preserve order
    return {"drug": found[0] if found else None, "all_drugs": found}


def chunk_matches_drug(chunk: dict, drug_key: str) -> bool:
    """
    Does this retrieved chunk actually mention the requested drug, by any
    of its known surface forms? Literal text match against the chunk's own
    content -- not the chunk's embedding, not its rerank score.
    """
    forms = DRUG_REGISTRY.get(drug_key, {})
    text = chunk.get("text", "")
    text_lower = text.lower()
    for lang_forms in forms.values():
        for surface in lang_forms:
            if surface.lower() in text_lower or surface in text:
                return True
    return False


def filter_chunks_by_drug(chunks: List[dict], drug_key: str) -> List[dict]:
    """Every chunk in `chunks` that mentions `drug_key`, in order."""
    return [c for c in chunks if chunk_matches_drug(c, drug_key)]
