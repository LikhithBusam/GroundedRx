"""Regression tests for the NLI contradiction-verification wrapper
(groundedrx/nli.py), ported from GroundedRx_Colab.ipynb's "Safety
Improvement: NLI Verification" self-check.

Uses hand-scripted FakeEmbedder/classify_fn test doubles instead of real
bge-m3/mDeBERTa-v3 -- this tests check_grounding_nli's CONTROL FLOW (when
NLI runs vs. short-circuits, how a contradiction overrides the verdict), not
real entailment understanding. No GPU, no torch, no transformers, no
network needed for any test here.
"""

import numpy as np

from groundedrx.config import REFUSAL
from groundedrx.nli import check_grounding_nli

NO_ALCOHOL = "The medicine should NOT be taken with alcohol"
DOSE = "The recommended dose is 10 mg once daily"
CTX = f"{NO_ALCOHOL}. {DOSE}."

CTX_VECS = {NO_ALCOHOL: [1, 0, 0, 0], DOSE: [0, 1, 0, 0]}


class FakeEmbedder:
    """Same role as test_grounding.py's FakeEmbedder: deterministic vectors
    so cosine similarity (and therefore the base gate's verdict) is fully
    controlled by the test."""

    def __init__(self, mapping: dict, default=(0.0, 0.0, 0.0, 1.0)):
        self.mapping = mapping
        self.default = list(default)

    def encode(self, texts, normalize_embeddings=True):
        return np.array([self.mapping.get(t, self.default) for t in texts])


def embedder_for(answer: str, matches: str) -> FakeEmbedder:
    """A FakeEmbedder where `answer` (minus trailing period) gets the same
    vector as context sentence `matches` -- i.e. answer and context "agree"
    on cosine similarity, so the base gate's semantic check passes and any
    block only comes from the NLI layer under test."""
    return FakeEmbedder({**CTX_VECS, answer.rstrip("."): CTX_VECS[matches]})


def always_entailment(premise, hypothesis):
    return {"label": "entailment", "scores": {"contradiction": 0.02, "entailment": 0.95, "neutral": 0.03}}


def always_contradiction(premise, hypothesis):
    return {"label": "contradiction", "scores": {"contradiction": 0.95, "entailment": 0.02, "neutral": 0.03}}


def counting_classify_fn(calls):
    def fn(premise, hypothesis):
        calls.append((premise, hypothesis))
        return always_entailment(premise, hypothesis)

    return fn


def test_contradiction_overrides_a_topically_similar_answer():
    # Same vector as its best-matching context sentence -> passes the base
    # cosine check -- exactly the negation case cosine alone cannot catch.
    answer = "The medicine can be taken with alcohol."
    embedder = embedder_for(answer, NO_ALCOHOL)

    result = check_grounding_nli(answer, CTX, "en", embedder=embedder, classify_fn=always_contradiction)

    assert result["grounded"] is False
    assert result["reason"] == "contradiction (NLI)"
    assert len(result["nli_contradictions"]) == 1


def test_no_contradiction_passes_through_unchanged():
    answer = DOSE + "."
    embedder = embedder_for(answer, DOSE)

    result = check_grounding_nli(answer, CTX, "en", embedder=embedder, classify_fn=always_entailment)

    assert result["grounded"] is True
    assert result["nli_contradictions"] == []


def test_refusal_short_circuits_before_nli_runs():
    calls = []
    result = check_grounding_nli(
        REFUSAL["en"], CTX, "en", embedder=FakeEmbedder({}), classify_fn=counting_classify_fn(calls)
    )
    assert result["reason"] == "refusal"
    assert calls == []


def test_already_failed_base_verdict_short_circuits_before_nli_runs():
    calls = []
    # Numeric hallucination: base check already fails on the number alone.
    answer = "The recommended dose is 999 mg once daily."
    embedder = embedder_for(answer, DOSE)

    result = check_grounding_nli(
        answer, CTX, "en", embedder=embedder, classify_fn=counting_classify_fn(calls)
    )

    assert result["grounded"] is False
    assert "999" in result["hallucinated_numbers"]
    assert calls == []


def test_nli_check_fails_soft_on_error_and_keeps_base_verdict():
    answer = DOSE + "."
    embedder = embedder_for(answer, DOSE)

    def broken_classify_fn(premise, hypothesis):
        raise RuntimeError("simulated NLI model failure")

    result = check_grounding_nli(answer, CTX, "en", embedder=embedder, classify_fn=broken_classify_fn)

    # Base gate already passed this answer; NLI failing must not flip that.
    assert result["grounded"] is True


def test_disabled_config_skips_nli_entirely(monkeypatch):
    import groundedrx.config as config_module

    monkeypatch.setitem(config_module.CONFIG_NLI, "enabled", False)
    calls = []
    answer = "The medicine can be taken with alcohol."
    embedder = embedder_for(answer, NO_ALCOHOL)

    result = check_grounding_nli(
        answer, CTX, "en", embedder=embedder, classify_fn=counting_classify_fn(calls)
    )

    assert calls == []
    assert result["grounded"] is True  # base gate alone has no way to catch this negation


def test_nli_unavailable_falls_back_to_base_verdict(monkeypatch):
    import groundedrx.nli as nli_module

    monkeypatch.setattr(nli_module, "_nli_available", lambda: False)
    answer = "The medicine can be taken with alcohol."
    embedder = embedder_for(answer, NO_ALCOHOL)

    # classify_fn intentionally omitted -- forces the _nli_available() path.
    result = check_grounding_nli(answer, CTX, "en", embedder=embedder)

    assert result["grounded"] is True  # falls back to the base gate's verdict
