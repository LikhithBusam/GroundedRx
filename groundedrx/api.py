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


class GroundingVerdict(BaseModel):
    grounded: bool
    reason: str
    min_similarity: float
    hallucinated_numbers: List[str]


class AnswerResponse(BaseModel):
    query: str
    language: str
    answer: str
    answer_raw: str
    grounding: GroundingVerdict
    retrieval_score: float


@app.get("/health")
def health():
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
    return AnswerResponse(
        query=result["query"],
        language=result["language"],
        answer=result["answer"],
        answer_raw=result["answer_raw"],
        grounding=GroundingVerdict(**{
            k: result["grounding"][k]
            for k in ("grounded", "reason", "min_similarity", "hallucinated_numbers")
        }),
        retrieval_score=result["retrieval_score"],
    )
