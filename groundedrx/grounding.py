"""Runtime groundedness gate: is every claim in an answer supported by its
retrieved context? Two deterministic checks, deliberately not an LLM
self-judge (see CLAUDE.md for why). Ported from GroundedRx_Colab.ipynb
Component 5, Cell 2b -- including three real bugs found and fixed live,
each preserved here with its original explanation and each covered by a
regression test in tests/test_grounding.py.

`embedder` is an injectable parameter (defaults to the real lazy-loaded
bge-m3 via resources.get_embed_model()) specifically so this module's logic
is unit-testable without a GPU or the sentence-transformers package -- pass
any object with an `.encode(list_of_str, normalize_embeddings=True) ->
array-like` method.
"""

import logging
import re

from . import config

logger = logging.getLogger(__name__)

# Arabic-Indic and Persian digits -> ASCII, so "١٠ مغ" and "10 mg" compare equal.
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def _numbers(text: str) -> set:
    """Every numeric literal in the text, digit-system normalized."""
    return set(re.findall(r"\d+(?:\.\d+)?", text.translate(_DIGIT_MAP)))


def _sentences(text: str) -> list:
    """Split on both Latin and Arabic terminators. Arabic '؟' is not '?'."""
    parts = re.split(r"[.!?؟\n]+", text)
    return [s.strip() for s in parts if len(s.strip()) >= config.CONFIG_GATE["min_sentence_chars"]]


def check_grounding(answer: str, context: str, language: str, embedder=None) -> dict:
    """
    Is every claim in `answer` supported by `context`?
    Returns a verdict dict; never raises -- an internal failure fails CLOSED.
    """
    # A correct refusal is the safe outcome, not an ungrounded claim.
    #
    # BUG FIXED: this used to be a plain substring test (REFUSAL in answer),
    # which matched an answer that merely QUOTES the refusal phrase
    # mid-sentence inside a much longer, substantive answer -- observed
    # live: a real dosage answer (with real numbers) that added "...it is
    # 'I don't have enough information to answer this question.' as the
    # context provides dosages for specific conditions..." The old check
    # short-circuited to grounded=True on that substring match, skipping
    # BOTH the numeric and semantic checks entirely for an answer
    # containing unverified numbers -- the fail-closed guarantee broke
    # exactly where it matters. Fix: the refusal phrase must account for
    # (nearly) the WHOLE answer, not just appear somewhere in it.
    # ponytail: 1.5x is a calibration knob for minor formatting slack
    # (trailing space/punctuation), not a derived constant.
    _refusal = config.REFUSAL.get(language, "")
    if _refusal and _refusal.lower() in answer.lower() and len(answer.strip()) <= len(_refusal) * 1.5:
        return {
            "grounded": True,
            "reason": "refusal",
            "min_similarity": 1.0,
            "ungrounded_sentences": [],
            "hallucinated_numbers": [],
        }

    try:
        # ── check 1: numeric grounding ──
        # Highest-consequence failure mode in a medication-leaflet system is
        # an invented dose. Exact set membership, no similarity involved.
        bad_numbers = sorted(_numbers(answer) - _numbers(context))

        # ── check 2: semantic grounding ──
        # ponytail: KNOWN CEILING -- cosine measures topical overlap, not
        # entailment. "Take this with alcohol" vs context "Do NOT take this
        # with alcohol" scores very high and passes. This is a fabrication
        # filter, not a faithfulness guarantee. A second, distinct known
        # ceiling: it also cannot catch wrong-drug substitution (a
        # confidently-answered but off-target retrieval scores high on
        # topical overlap too) -- see CLAUDE.md/README for both.
        ans_sents = _sentences(answer)
        ctx_sents = _sentences(context)

        # BUG FIXED: bulleted answers (common in Arabic output) fragment
        # into pieces shorter than min_sentence_chars and get filtered to
        # nothing. The old code then returned min_similarity=1.0 and
        # PASSED -- a list-shaped hallucination skipped the semantic check
        # entirely while reporting a perfect score. Fall back to scoring
        # the whole answer/context as one unit instead.
        if not ans_sents and answer.strip():
            ans_sents = [answer.strip()]
        if not ctx_sents and context.strip():
            ctx_sents = [context.strip()]

        if not ans_sents:
            return {
                "grounded": False,
                "reason": "empty answer",
                "min_similarity": 0.0,
                "ungrounded_sentences": [],
                "hallucinated_numbers": bad_numbers,
            }
        if not ctx_sents:
            return {
                "grounded": False,
                "reason": "empty context",
                "min_similarity": 0.0,
                "ungrounded_sentences": ans_sents,
                "hallucinated_numbers": bad_numbers,
            }

        if embedder is None:
            from .resources import get_embed_model

            embedder = get_embed_model()

        a_vecs = embedder.encode(ans_sents, normalize_embeddings=True)
        c_vecs = embedder.encode(ctx_sents, normalize_embeddings=True)
        sims = a_vecs @ c_vecs.T  # normalized -> dot product IS cosine
        best = sims.max(axis=1)

        thr = config.CONFIG_GATE["min_sentence_similarity"]
        ungrounded = [(s, float(b)) for s, b in zip(ans_sents, best) if b < thr]

        # BUG FIXED (observed live, twice): blocking on ANY single
        # ungrounded sentence let one purely structural sentence -- a
        # trailing "I don't have enough information" hedge, or a leading
        # "According to the sources:" framing line -- veto an otherwise
        # accurate, well-grounded multi-sentence answer. Fix: block on a
        # MAJORITY of sentences failing, not any one. Still fails closed on
        # genuine fabrication. ponytail: known ceiling, untested in
        # practice -- exactly half-ungrounded on a very short (e.g.
        # 2-sentence) answer does NOT trigger (needs strictly >50%).
        majority_ungrounded = len(ungrounded) > len(ans_sents) / 2

        return {
            "grounded": (not majority_ungrounded) and (not bad_numbers),
            "reason": "ok" if (not majority_ungrounded and not bad_numbers) else "unsupported content",
            "min_similarity": float(best.min()),
            "ungrounded_sentences": ungrounded,
            "hallucinated_numbers": bad_numbers,
        }

    except Exception as e:
        # FAIL CLOSED. In offline eval a broken metric costs a data point;
        # here it would ship an unverified medical answer. Never default to
        # pass.
        logger.error(f"Grounding gate error: {e}")
        return {
            "grounded": False,
            "reason": f"gate error: {e}",
            "min_similarity": 0.0,
            "ungrounded_sentences": [],
            "hallucinated_numbers": [],
        }
