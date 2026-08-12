"""NLI-based contradiction verification, layered on top of the deterministic
groundedness gate in grounding.py. Ported from GroundedRx_Colab.ipynb's
"Safety Improvement: NLI Verification" cell.

Catches a failure mode the base gate's cosine-similarity check cannot:
negation. "Take this with alcohol" against context "do NOT take this with
alcohol" scores high on topical similarity and passes the base gate; an NLI
model, trained specifically for entailment/contradiction, catches it. This
is a fabrication-filter *addition*, not a faithfulness guarantee -- see
CLAUDE.md for the full documented ceiling.

`classify_fn` is an injectable parameter (defaults to the real, lazy-loaded
mDeBERTa-v3-xnli model via resources.py) specifically so check_grounding_nli
is unit-testable without a GPU or the transformers/torch packages installed
-- mirrors how grounding.check_grounding injects `embedder`. Pass any
callable matching (premise: str, hypothesis: str) -> {"label": str,
"scores": dict}.
"""

import logging
from functools import lru_cache
from typing import Any, Callable, Optional

from . import config
from .grounding import _sentences, check_grounding

logger = logging.getLogger(__name__)

ClassifyFn = Callable[[str, str], dict]


def _real_nli_classify(premise: str, hypothesis: str) -> dict:
    """premise = context sentence (source of truth), hypothesis = answer
    sentence (claim being checked). Loads the real, GPU-resident NLI model
    via resources.py -- deferred imports so this module stays importable
    without torch/transformers installed."""
    import torch

    from .resources import get_nli_model, get_nli_tokenizer

    tokenizer = get_nli_tokenizer()
    model = get_nli_model()
    inputs = tokenizer(
        premise, hypothesis, return_tensors="pt", truncation=True, max_length=256
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]
    # Read label order from the model's own config -- do not hardcode indices.
    labels = {k: v.lower() for k, v in model.config.id2label.items()}
    scores = {labels[i]: float(probs[i]) for i in range(len(probs))}
    label = max(scores, key=scores.get)
    return {"label": label, "scores": scores}


@lru_cache(maxsize=1)
def _nli_available() -> bool:
    """Attempts to load the real NLI model once; caches the result so a
    genuinely unavailable model (missing deps, no GPU) isn't re-attempted
    on every single check_grounding_nli call."""
    try:
        from .resources import get_nli_model, get_nli_tokenizer

        get_nli_tokenizer()
        get_nli_model()
        return True
    except Exception as e:
        logger.error(f"NLI model unavailable, contradiction checking disabled: {e}")
        return False


def check_grounding_nli(
    answer: str,
    context: str,
    language: str,
    embedder: Optional[Any] = None,
    classify_fn: Optional[ClassifyFn] = None,
) -> dict:
    """
    Same contract as grounding.check_grounding, wrapping it with an
    additional NLI contradiction check. Runs the base numeric + semantic
    checks UNCHANGED first; only if NLI is enabled/available and the base
    verdict didn't already fail or return a refusal, checks each answer
    sentence against its best-matching context sentence for entailment
    contradiction. A strong contradiction overrides the verdict to
    grounded=False.

    `classify_fn` defaults to the real model (lazy-loaded, GPU-resident);
    inject a test double matching (premise, hypothesis) -> {"label": ...,
    "scores": {...}} to unit-test this without a GPU.
    """
    base = check_grounding(answer, context, language, embedder=embedder)

    if not config.CONFIG_NLI["enabled"]:
        return base
    if base["reason"] == "refusal" or not base["grounded"]:
        # Already correctly handled -- a refusal has nothing to contradict,
        # and an already-blocked answer doesn't need a second reason.
        return base

    if classify_fn is None:
        if not _nli_available():
            return base
        classify_fn = _real_nli_classify

    try:
        if embedder is None:
            from .resources import get_embed_model

            embedder = get_embed_model()

        ans_sents = _sentences(answer) or ([answer.strip()] if answer.strip() else [])
        ctx_sents = _sentences(context) or ([context.strip()] if context.strip() else [])
        if not ans_sents or not ctx_sents:
            return base

        a_vecs = embedder.encode(ans_sents, normalize_embeddings=True)
        c_vecs = embedder.encode(ctx_sents, normalize_embeddings=True)
        sims = a_vecs @ c_vecs.T
        best_ctx_idx = sims.argmax(axis=1)

        contradictions = []
        for i, sent in enumerate(ans_sents):
            paired_ctx = ctx_sents[best_ctx_idx[i]]
            verdict = classify_fn(paired_ctx, sent)
            if (
                verdict["label"] == "contradiction"
                and verdict["scores"]["contradiction"] >= config.CONFIG_NLI["contradiction_threshold"]
            ):
                contradictions.append((sent, paired_ctx, verdict["scores"]["contradiction"]))

        if contradictions:
            logger.warning(f"NLI contradiction detected: {len(contradictions)} sentence(s)")
            return {
                **base,
                "grounded": False,
                "reason": "contradiction (NLI)",
                "nli_contradictions": contradictions,
            }
        return {**base, "nli_contradictions": []}

    except Exception as e:
        # Fail SOFT here, not closed -- NLI is documented as an additional
        # signal, not the safety baseline. The base gate's verdict (already
        # computed above) still stands; this only affects whether the extra
        # layer ran.
        logger.error(f"NLI check errored, falling back to base gate verdict: {e}")
        return base
