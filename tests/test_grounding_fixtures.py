"""Real-data regression tests for the groundedness gate, backed by a frozen
Arabic case with genuinely real bge-m3 embeddings -- not synthetic test
doubles. See tests/fixtures/grounding/ar_refusal_case.json.

Origin (see the fixture's own "description" field for the full story): the
query and the retrieved-context excerpts are real, verbatim from a live
GroundedRx demo run (Component 7) that got blocked by the gate. The
"answer_raw" text is a reconstructed, representative unsupported answer
(the actual raw model output from that run was never saved) built from real
side-effect content documented elsewhere in the corpus. Every embedding
vector is real bge-m3 output, computed once on a real GPU (Kaggle T4) and
frozen here so this test needs zero GPU, zero network, zero model download
-- prompted by https://github.com/LikhithBusam/GroundedRx/issues/1.
"""

import json
from pathlib import Path

import numpy as np

from groundedrx.config import CONFIG_GATE, REFUSAL
from groundedrx.grounding import check_grounding

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "grounding" / "ar_refusal_case.json"


class FrozenEmbedder:
    """Test double backed by real, precomputed bge-m3 vectors, looked up by
    exact sentence text. Raises loudly (not silently) if check_grounding's
    _sentences() splits the frozen text differently than expected -- a
    KeyError here means the fixture's text/vectors have drifted apart, not
    that the gate behaved unexpectedly."""

    def __init__(self, sentence_vectors: dict):
        self.sentence_vectors = sentence_vectors

    def encode(self, texts, normalize_embeddings=True):
        try:
            return np.array([self.sentence_vectors[t] for t in texts])
        except KeyError as e:
            raise KeyError(
                f"No frozen vector for sentence {e} -- _sentences() split the "
                "fixture's context/answer_raw text differently than expected."
            ) from e


def _load_fixture():
    with open(FIXTURE_PATH, encoding="utf-8") as f:
        return json.load(f)


def test_ar_case_is_blocked_with_real_embeddings():
    """The primary case: a real query, real retrieved context, and an
    answer whose content is real (documented elsewhere in the corpus) but
    unsupported by THIS context -- must be blocked, using real bge-m3
    similarity scores, not synthetic ones."""
    fixture = _load_fixture()
    embedder = FrozenEmbedder(fixture["sentence_vectors"])

    verdict = check_grounding(
        fixture["answer_raw"], fixture["context"], fixture["language"], embedder=embedder
    )

    assert verdict["grounded"] is False
    assert verdict["reason"] == "unsupported content"
    assert verdict["min_similarity"] < CONFIG_GATE["min_sentence_similarity"]
    assert verdict["hallucinated_numbers"] == []  # blocked on semantics, not invented numbers


def test_ar_case_delivered_answer_becomes_refusal_and_raw_is_kept():
    """Exercises the same block-and-substitute logic pipeline.rag_answer()
    applies, without needing GPU generation: a blocked verdict must produce
    the Arabic refusal as the delivered answer while the original raw text
    is preserved separately for inspection."""
    fixture = _load_fixture()
    embedder = FrozenEmbedder(fixture["sentence_vectors"])

    verdict = check_grounding(
        fixture["answer_raw"], fixture["context"], fixture["language"], embedder=embedder
    )

    delivered = fixture["answer_raw"]
    if CONFIG_GATE["block_on_fail"] and not verdict["grounded"]:
        delivered = REFUSAL[fixture["language"]]

    assert delivered == REFUSAL["ar"]
    assert delivered != fixture["answer_raw"]  # answer_raw_is_kept: true


def test_ar_case_known_ceiling_exactly_half_ungrounded_does_not_block():
    """Documented, known ceiling (see grounding.py): a MAJORITY of sentences
    must fail to trigger a block, not just "at least one." Using only the
    first two of the three real answer sentences from the same fixture
    (1 of 2 below threshold -- exactly half, not a majority) reproduces
    that documented gap with real data instead of leaving it untested."""
    fixture = _load_fixture()
    embedder = FrozenEmbedder(fixture["sentence_vectors"])

    two_sentences = fixture["answer_raw_two_sentence_subset"]
    verdict = check_grounding(two_sentences, fixture["context"], fixture["language"], embedder=embedder)

    assert verdict["grounded"] is True  # known ceiling: half-ungrounded does not block
