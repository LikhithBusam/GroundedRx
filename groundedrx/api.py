"""FastAPI service over rag_answer() — the on-premises deployment surface.

Two endpoints: GET /health (liveness, does not force a model load) and
POST /answer (the real end-to-end RAG call). No auth, no API keys, no rate
limiting, by design -- this is meant to run inside your own network (see
README "On-Premises Deployment"), not be exposed to the public internet.

Not imported by groundedrx/__init__.py -- fastapi is an optional `api`
dependency extra, not a core one, so `import groundedrx` still needs no
web-framework install either.
"""

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel

from . import __version__, config
from .pipeline import rag_answer

app = FastAPI(
    title="GroundedRx",
    description="Bilingual (Arabic/English) medical RAG with a runtime groundedness gate.",
    version=__version__,
)


class AnswerRequest(BaseModel):
    query: str
    document_id_filter: Optional[int] = None  # eval-only knob; leave unset for real queries


class NLIContradiction(BaseModel):
    """One sentence-level contradiction caught by the NLI verification layer
    (see nli.check_grounding_nli). Field order/meaning matches the 3-tuple
    nli.py itself appends: (answer sentence, its best-matching context
    sentence, the NLI model's contradiction probability)."""

    sentence: str
    context_sentence: str
    contradiction_score: float


class GroundingVerdict(BaseModel):
    grounded: bool
    reason: str
    min_similarity: float
    hallucinated_numbers: List[str]
    # nli.check_grounding_nli only adds this key to the verdict dict when the
    # NLI check actually ran to completion (skipped entirely on a refusal,
    # an already-failed base verdict, disabled config, or an unavailable
    # model) -- defaults to [] so callers always get a list, never a
    # missing field, regardless of which path produced the verdict.
    nli_contradictions: List[NLIContradiction] = []


class AnswerResponse(BaseModel):
    query: str
    language: str
    answer: str
    answer_raw: str
    grounding: GroundingVerdict
    retrieval_score: float


@app.get("/health")
def health() -> dict:
    """Liveness check -- does not load the model or touch the vector store,
    so it stays fast even before the first real request warms anything up."""
    return {"status": "ok", "model": config.MODEL_NAME}


@app.post("/answer", response_model=AnswerResponse)
def answer(request: AnswerRequest) -> AnswerResponse:
    """
    Retrieve -> rerank -> generate -> groundedness gate, end to end.

    The response deliberately surfaces `grounding` and `answer_raw`
    alongside the delivered `answer` -- if the gate blocked the answer,
    `answer` is the refusal and `answer_raw` is what the model actually
    generated before the gate stepped in. Callers can see the gate is
    real and inspect what it caught, not just trust a black box.
    """
    result = rag_answer(request.query, document_id_filter=request.document_id_filter)
    grounding = result["grounding"]
    return AnswerResponse(
        query=result["query"],
        language=result["language"],
        answer=result["answer"],
        answer_raw=result["answer_raw"],
        grounding=GroundingVerdict(
            grounded=grounding["grounded"],
            reason=grounding["reason"],
            min_similarity=grounding["min_similarity"],
            hallucinated_numbers=grounding["hallucinated_numbers"],
            nli_contradictions=[
                NLIContradiction(sentence=s, context_sentence=c, contradiction_score=score)
                for s, c, score in grounding.get("nli_contradictions", [])
            ],
        ),
        retrieval_score=result["retrieval_score"],
    )
