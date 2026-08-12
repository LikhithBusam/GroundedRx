"""FastAPI wiring tests. `rag_answer` is monkeypatched -- no GPU, no model,
no vector store needed; this tests the HTTP <-> Pydantic <-> pipeline-dict
mapping in api.py, not the pipeline itself (that's covered elsewhere)."""

from fastapi.testclient import TestClient

import groundedrx.api as api_module


def _fake_result(**overrides):
    result = {
        "query": "What are the side effects of Linopril?",
        "language": "en",
        "answer": "Common side effects include headache and dizziness.",
        "answer_raw": "Common side effects include headache and dizziness.",
        "grounding": {
            "grounded": True,
            "reason": "ok",
            "min_similarity": 0.82,
            "ungrounded_sentences": [],
            "hallucinated_numbers": [],
        },
        "retrieval_score": 0.71,
        "rewrite_count": 1,
        "reranked_chunks": [],
        "input_tokens": 120,
        "output_tokens": 40,
    }
    result.update(overrides)
    return result


def test_health_does_not_require_a_loaded_model():
    client = TestClient(api_module.app)
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model" in body


def test_answer_maps_pipeline_result_to_response_shape(monkeypatch):
    monkeypatch.setattr(api_module, "rag_answer", lambda query, document_id_filter=None: _fake_result())
    client = TestClient(api_module.app)

    resp = client.post("/answer", json={"query": "What are the side effects of Linopril?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Common side effects include headache and dizziness."
    assert body["grounding"]["grounded"] is True
    assert body["grounding"]["min_similarity"] == 0.82
    assert body["retrieval_score"] == 0.71


def test_answer_surfaces_a_blocked_gate_verdict(monkeypatch):
    """The blocked-vs-answer_raw distinction is the deliberate design
    highlight of this endpoint (see api.py docstring) -- verify it survives
    the HTTP round trip, not just the Python dict."""
    blocked = _fake_result(
        answer="I don't have enough information to answer this question.",
        answer_raw="The dose is 999 mg.",
        grounding={
            "grounded": False,
            "reason": "unsupported content",
            "min_similarity": 0.31,
            "ungrounded_sentences": [],
            "hallucinated_numbers": ["999"],
        },
    )
    monkeypatch.setattr(api_module, "rag_answer", lambda query, document_id_filter=None: blocked)
    client = TestClient(api_module.app)

    resp = client.post("/answer", json={"query": "What is the dose?"})

    body = resp.json()
    assert body["grounding"]["grounded"] is False
    assert body["grounding"]["hallucinated_numbers"] == ["999"]
    assert body["answer"] != body["answer_raw"]


def test_document_id_filter_is_passed_through(monkeypatch):
    seen = {}

    def fake_rag_answer(query, document_id_filter=None):
        seen["document_id_filter"] = document_id_filter
        return _fake_result()

    monkeypatch.setattr(api_module, "rag_answer", fake_rag_answer)
    client = TestClient(api_module.app)

    client.post("/answer", json={"query": "test", "document_id_filter": 502})

    assert seen["document_id_filter"] == 502


def test_answer_omits_nli_contradictions_when_nli_never_ran(monkeypatch):
    """_fake_result()'s grounding dict has no "nli_contradictions" key at
    all -- the real shape whenever NLI is skipped (refusal, disabled config,
    unavailable model, already-failed base verdict). Must not KeyError and
    must surface as an empty list, not a missing field."""
    monkeypatch.setattr(api_module, "rag_answer", lambda query, document_id_filter=None: _fake_result())
    client = TestClient(api_module.app)

    resp = client.post("/answer", json={"query": "What are the side effects of Linopril?"})

    assert resp.status_code == 200
    assert resp.json()["grounding"]["nli_contradictions"] == []


def test_answer_exposes_nli_contradiction_evidence(monkeypatch):
    """When NLI actually blocks an answer, the evidence -- which answer
    sentence contradicted which context sentence, at what score -- must
    survive the HTTP round trip, not just the "contradiction (NLI)" reason
    string."""
    blocked = _fake_result(
        answer="I don't have enough information to answer this question.",
        answer_raw="This medicine is safe to use during pregnancy.",
        grounding={
            "grounded": False,
            "reason": "contradiction (NLI)",
            "min_similarity": 0.71,
            "ungrounded_sentences": [],
            "hallucinated_numbers": [],
            "nli_contradictions": [
                (
                    "This medicine is safe to use during pregnancy.",
                    "This medicine must not be used by pregnant women.",
                    0.93,
                )
            ],
        },
    )
    monkeypatch.setattr(api_module, "rag_answer", lambda query, document_id_filter=None: blocked)
    client = TestClient(api_module.app)

    resp = client.post("/answer", json={"query": "Is this medicine safe during pregnancy?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["grounding"]["reason"] == "contradiction (NLI)"
    contradictions = body["grounding"]["nli_contradictions"]
    assert len(contradictions) == 1
    assert contradictions[0] == {
        "sentence": "This medicine is safe to use during pregnancy.",
        "context_sentence": "This medicine must not be used by pregnant women.",
        "contradiction_score": 0.93,
    }


def test_answer_with_nli_run_but_no_contradictions_returns_empty_list(monkeypatch):
    """NLI actually ran (unlike the "never ran" case above) and found
    nothing -- the real shape is an explicit empty list, not a missing key."""
    result = _fake_result(
        grounding={
            "grounded": True,
            "reason": "ok",
            "min_similarity": 0.82,
            "ungrounded_sentences": [],
            "hallucinated_numbers": [],
            "nli_contradictions": [],
        }
    )
    monkeypatch.setattr(api_module, "rag_answer", lambda query, document_id_filter=None: result)
    client = TestClient(api_module.app)

    resp = client.post("/answer", json={"query": "What are the side effects of Linopril?"})

    assert resp.status_code == 200
    assert resp.json()["grounding"]["nli_contradictions"] == []
