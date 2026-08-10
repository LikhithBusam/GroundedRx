"""Regression tests for the three real bugs found and fixed in the
groundedness gate (see CLAUDE.md for the full incident history), plus the
core numeric/refusal/fail-closed behavior.

Uses a hand-scripted FakeEmbedder test double instead of real bge-m3 --
this tests the gate's deterministic CONTROL FLOW (bullet-list fallback,
refusal-substring fix, majority-vote fix, fail-closed on error), not real
semantic understanding. Real semantic quality is verified separately, live,
in the notebook's own self-check block against actual bge-m3 embeddings.
No GPU, no sentence-transformers, no network needed for any test here.
"""

import numpy as np

from groundedrx.config import REFUSAL
from groundedrx.grounding import check_grounding

CTX = (
    "The recommended dose is 10 mg once daily. "
    "Store below 25 degrees Celsius. "
    "Common side effects include dizziness and a dry cough."
)


class FakeEmbedder:
    """Deterministic test double: maps known sentence strings to hand-picked
    vectors so cosine similarity is fully controlled by the test, not by a
    real model's judgment."""

    def __init__(self, mapping: dict, default=(0.0, 0.0, 0.0, 1.0)):
        self.mapping = mapping
        self.default = list(default)

    def encode(self, texts, normalize_embeddings=True):
        return np.array([self.mapping.get(t, self.default) for t in texts])


# Context sentences after _sentences() splitting, each given an orthogonal
# vector so cross-similarity between *different* context sentences is 0.
CTX_VECS = {
    "The recommended dose is 10 mg once daily": [1, 0, 0, 0],
    "Store below 25 degrees Celsius": [0, 1, 0, 0],
    "Common side effects include dizziness and a dry cough": [0, 0, 1, 0],
}


def test_grounded_answer_passes():
    answer = "The recommended dose is 10 mg once daily."
    dose_vec = CTX_VECS["The recommended dose is 10 mg once daily"]
    embedder = FakeEmbedder({**CTX_VECS, answer.rstrip("."): dose_vec})
    result = check_grounding(answer, CTX, "en", embedder=embedder)
    assert result["grounded"] is True
    assert result["hallucinated_numbers"] == []


def test_numeric_grounding_catches_invented_dose():
    # Topically identical to the grounded sentence (same vector) -- passes
    # semantically -- but the number itself was never in the context.
    answer = "The recommended dose is 80 mg once daily."
    embedder = FakeEmbedder(
        {**CTX_VECS, answer.rstrip("."): CTX_VECS["The recommended dose is 10 mg once daily"]}
    )
    result = check_grounding(answer, CTX, "en", embedder=embedder)
    assert result["grounded"] is False
    assert "80" in result["hallucinated_numbers"]


def test_correct_refusal_is_not_flagged_as_hallucination():
    result = check_grounding(REFUSAL["en"], CTX, "en", embedder=FakeEmbedder({}))
    assert result["grounded"] is True
    assert result["reason"] == "refusal"
    assert result["min_similarity"] == 1.0


def test_quoted_refusal_does_not_bypass_the_gate():
    """BUG FIX regression: a real dosage answer that merely QUOTES the
    refusal phrase mid-sentence must not short-circuit to grounded=True --
    its numbers still need checking. Old code used a bare substring test."""
    answer = (
        'The maximum dose is 200 mg/day. However, if asked generally, it is '
        '"I don\'t have enough information to answer this question." since the '
        "context covers specific conditions only."
    )
    result = check_grounding(answer, CTX, "en", embedder=FakeEmbedder({}))
    assert result["reason"] != "refusal"
    assert "200" in result["hallucinated_numbers"]


def test_bullet_list_answer_is_scored_not_vacuously_passed():
    """BUG FIX regression: bulleted answers fragment below min_sentence_chars
    and used to get filtered to an empty list, which the old code scored as
    a vacuous min_similarity=1.0 PASS. Must fall back to scoring the whole
    answer as one unit instead."""
    answer = "- headache\n- nausea\n- dizziness"
    # CTX_VECS gives context sentences their real distinguishing vectors;
    # without it they'd fall to the same default as the unmapped answer
    # fallback string below and produce a false 1.0 match against itself.
    embedder = FakeEmbedder({**CTX_VECS, answer: [0, 0, 0, 1]})
    result = check_grounding(answer, CTX, "en", embedder=embedder)
    assert result["min_similarity"] < 1.0


def test_majority_vote_one_weak_sentence_does_not_veto_a_good_answer():
    """BUG FIX regression: a single low-relevance sentence (a hedge, a
    framing line) used to veto an otherwise well-grounded multi-sentence
    answer. Must block only when a MAJORITY of sentences are ungrounded."""
    answer = (
        "This document was reviewed for accuracy purposes only. "
        "The recommended dose is 10 mg once daily. "
        "Store below 25 degrees Celsius."
    )
    embedder = FakeEmbedder(CTX_VECS)  # unmapped framing sentence -> default/unrelated vector
    result = check_grounding(answer, CTX, "en", embedder=embedder)
    assert result["grounded"] is True


def test_majority_vote_still_blocks_a_fully_fabricated_answer():
    """The majority-vote fix must not weaken detection of real
    hallucination -- a single wrong sentence is still 100% ungrounded.

    Uses CTX_VECS (not an empty mapping) so context sentences keep their
    distinguishing vectors -- an empty mapping would collapse both the
    fabricated answer sentence AND every context sentence to the same
    default vector, producing a false perfect match instead of testing
    anything.
    """
    answer = "This medicine cures pancreatic cancer completely and permanently for all patients."
    result = check_grounding(answer, CTX, "en", embedder=FakeEmbedder(CTX_VECS))
    assert result["grounded"] is False


def test_empty_answer_fails_closed():
    result = check_grounding("", CTX, "en", embedder=FakeEmbedder({}))
    assert result["grounded"] is False
    assert result["reason"] == "empty answer"


def test_gate_fails_closed_on_internal_error():
    class BrokenEmbedder:
        def encode(self, texts, normalize_embeddings=True):
            raise RuntimeError("simulated embedder failure")

    result = check_grounding("The dose is 10 mg once daily.", CTX, "en", embedder=BrokenEmbedder())
    assert result["grounded"] is False
    assert "gate error" in result["reason"]
