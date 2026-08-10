"""Hybrid retrieval pipeline: language detection -> query rewrite -> embed ->
retrieve (dense+BM25, RRF-fused) -> quality gate (feedback loop) -> rerank ->
context build. Ported from GroundedRx_Colab.ipynb Component 4.

`rrf_fuse` is pulled out as a pure function (plain dicts in, plain dicts out,
no Qdrant/BM25 dependency) specifically so the fusion math is unit-testable
without a live vector store or a GPU.
"""

import logging
import re
from functools import lru_cache
from typing import List, Optional, TypedDict

from . import config

logger = logging.getLogger(__name__)


class RAGState(TypedDict):
    query: str
    language: str
    rewritten_query: str
    rewrite_count: int
    query_vector: List[float]
    retrieved_chunks: List[dict]
    reranked_chunks: List[dict]
    retrieval_score: float
    context: str
    needs_rewrite: bool
    document_id_filter: Optional[int]  # eval-only: pin retrieval to one known document


def _tokenize(text: str) -> List[str]:
    # ponytail: unicode word split, no stemming or Arabic morphological
    # analysis. Verified to tokenize Arabic script correctly. Add a light
    # Arabic stemmer only if recall on inflected forms measurably suffers.
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def rrf_fuse(dense_hits: List[dict], sparse_hits: List[dict], rrf_k: int) -> List[dict]:
    """
    Reciprocal Rank Fusion over two already rank-ordered chunk lists.

    dense_hits: rank-ordered list of dicts with at least {"id", "score", ...},
                "score" is the dense cosine similarity.
    sparse_hits: rank-ordered list of dicts with at least {"id", "bm25_score", ...}.

    Fuses on RANK, not raw score: cosine is bounded 0-1 while BM25 is
    unbounded and its scale shifts per query, so the two aren't directly
    comparable and a score-weighted blend would need per-query
    normalization. RRF sidesteps that entirely.

    Pure function -- no Qdrant/BM25/network dependency, fully unit-testable
    with hand-built dicts.
    """
    fused: dict = {}
    meta: dict = {}

    for rank, hit in enumerate(dense_hits):
        pid = hit["id"]
        fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
        meta[pid] = {**hit, "bm25_score": hit.get("bm25_score", 0.0)}

    for rank, hit in enumerate(sparse_hits):
        pid = hit["id"]
        fused[pid] = fused.get(pid, 0.0) + 1.0 / (rrf_k + rank + 1)
        if pid not in meta:
            # sparse-only hit: no cosine was ever computed for it
            meta[pid] = {**hit, "score": hit.get("score", 0.0)}
        else:
            meta[pid]["bm25_score"] = hit.get("bm25_score", 0.0)

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [{**meta[pid], "rrf_score": s} for pid, s in ranked]


# ── Node 1: Language Detection ──
def detect_language(state: RAGState) -> RAGState:
    from langdetect import detect

    try:
        lang = detect(state["query"])
        language = "ar" if lang == "ar" else "en"
    except Exception:
        language = "en"
    logger.info(f"Language: {language.upper()} | Query: {state['query'][:60]}")
    return {**state, "language": language}


# ── Node 2: Query Rewriting ──
def rewrite_query(state: RAGState) -> RAGState:
    """
    Pass 0 embeds the query as-is; retry passes expand with medical domain
    terms. Expansion is a RETRY lever, not a default transform -- measured to
    roughly halve cross-lingual retrieval for Arabic queries (60% -> 33% of
    top-20 crossing the language boundary) by pulling the embedding back
    toward the query's own language. Raw queries already clear the 0.5
    quality gate (cosine 0.63-0.73), so expansion only earns its cost once
    the gate has actually failed.
    """
    query = state["query"]
    language = state["language"]
    rewrite_count = state.get("rewrite_count", 0)

    if rewrite_count == 0:
        logger.info(f"Pass 0 [{language.upper()}]: raw query, no expansion")
        return {**state, "rewritten_query": query, "rewrite_count": 1}

    if language == "ar":
        rewritten = (
            f"{query} "
            f"معلومات دوائية آثار جانبية جرعة تحذيرات "
            f"نشرة المريض تخزين الدواء"
        )
    else:
        rewritten = (
            f"{query} "
            f"medication information side effects dosage "
            f"warnings patient leaflet storage instructions"
        )

    logger.info(f"Rewritten [{language.upper()}]: {rewritten[:100]}")
    return {**state, "rewritten_query": rewritten, "rewrite_count": rewrite_count + 1}


# ── Node 3: Query Embedding ──
def embed_query(state: RAGState) -> RAGState:
    from .resources import get_embed_model

    query = state.get("rewritten_query", state["query"])
    vector = get_embed_model().encode(query, normalize_embeddings=True).tolist()
    logger.info(f"Query embedded | dim: {len(vector)}")
    return {**state, "query_vector": vector}


# ── Node 4: Retrieval ──
def retrieve_chunks(state: RAGState) -> RAGState:
    """Hybrid retrieval: dense (bge-m3 / Qdrant cosine) + sparse (BM25),
    combined via `rrf_fuse`."""
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    from .resources import get_bm25_docs, get_bm25_index, get_client

    doc_filter = state.get("document_id_filter")
    query_filter = (
        Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=doc_filter))])
        if doc_filter is not None
        else None
    )

    k = config.CONFIG_C4["top_k_retrieve"]
    query = state.get("rewritten_query", state["query"])

    # ── dense half ──
    dense_points = get_client().query_points(
        collection_name=config.CONFIG_C4["collection_name"],
        query=state["query_vector"],
        limit=k,
        query_filter=query_filter,
    ).points
    dense_hits = [
        {
            "id": p.id,
            "text": p.payload.get("chunk_text", ""),
            "language": p.payload.get("language", ""),
            "category": p.payload.get("category", ""),
            "document_id": p.payload.get("document_id", ""),
            "chunk_id": p.payload.get("chunk_id", ""),
            "file_name": p.payload.get("file_name", ""),
            "score": p.score,
        }
        for p in dense_points
    ]

    # ── sparse half ──
    bm25_docs = get_bm25_docs()
    bm_scores = get_bm25_index().get_scores(_tokenize(query))
    cand_idx = range(len(bm25_docs))
    if doc_filter is not None:
        cand_idx = [i for i in cand_idx if bm25_docs[i]["document_id"] == doc_filter]
    sparse_idx = sorted(cand_idx, key=lambda i: bm_scores[i], reverse=True)[:k]
    sparse_hits = [
        {**bm25_docs[i], "bm25_score": float(bm_scores[i])} for i in sparse_idx
    ]

    chunks = rrf_fuse(dense_hits, sparse_hits, config.CONFIG_C4["rrf_k"])[:k]

    # retrieval_score stays the best DENSE cosine, deliberately -- the
    # quality gate below is calibrated against cosine (threshold 0.5); an
    # RRF score (max ~0.016) or raw BM25 score would make the gate fire on
    # every query or on none. Fusion changes what we retrieve, not how we
    # judge it.
    best_score = dense_hits[0]["score"] if dense_hits else 0.0
    n_sparse_only = sum(1 for ch in chunks if ch["score"] == 0.0)
    logger.info(
        f"Hybrid retrieved {len(chunks)} chunks "
        f"({n_sparse_only} sparse-only) | Best dense: {best_score:.4f}"
    )

    return {**state, "retrieved_chunks": chunks, "retrieval_score": best_score}


# ── Node 5: Quality Check ──
def check_retrieval_quality(state: RAGState) -> RAGState:
    score = state["retrieval_score"]
    rewrite_count = state.get("rewrite_count", 0)
    needs_rewrite = (
        score < config.CONFIG_C4["score_threshold"]
        and rewrite_count < config.CONFIG_C4["max_rewrites"]
    )

    if needs_rewrite:
        logger.warning(
            f"Low score: {score:.4f} | Rewrite {rewrite_count}/{config.CONFIG_C4['max_rewrites']}"
        )
    else:
        logger.info(f"Quality OK: {score:.4f}")

    return {**state, "needs_rewrite": needs_rewrite}


# ── Node 6: Reranking ──
def rerank_chunks(state: RAGState) -> RAGState:
    from .resources import get_reranker

    query = state.get("rewritten_query", state["query"])
    chunks = state["retrieved_chunks"]

    if not chunks:
        return {**state, "reranked_chunks": []}

    pairs = [(query, c["text"]) for c in chunks]
    scores = get_reranker().predict(pairs)

    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    top_chunks = [
        {**chunk, "rerank_score": float(score)}
        for score, chunk in ranked[: config.CONFIG_C4["top_k_rerank"]]
    ]

    logger.info(
        f"Reranked {len(chunks)} → {len(top_chunks)} | Top: {top_chunks[0]['rerank_score']:.4f}"
    )
    return {**state, "reranked_chunks": top_chunks}


# ── Node 7: Context Builder ──
def build_context(state: RAGState) -> RAGState:
    chunks = state["reranked_chunks"]
    language = state["language"]

    if not chunks:
        context = (
            "No relevant information found." if language == "en" else "لم يتم العثور على معلومات ذات صلة."
        )
        return {**state, "context": context}

    parts = []
    for i, chunk in enumerate(chunks, 1):
        header = f"[Source {i}]" if language == "en" else f"[المصدر {i}]"
        parts.append(
            f"{header}\n"
            f"Category : {chunk.get('category', 'N/A')}\n"
            f"Language : {chunk.get('language', 'N/A')}\n"
            f"Text     : {chunk['text']}\n"
        )

    context = "\n---\n".join(parts)
    logger.info(f"Context built | {len(chunks)} chunks | {len(context):,} chars")
    return {**state, "context": context}


def route_after_quality_check(state: RAGState) -> str:
    if state.get("needs_rewrite", False):
        return "rewrite_query"
    return "rerank_chunks"


@lru_cache(maxsize=1)
def get_pipeline():
    """Compile the LangGraph pipeline once, lazily -- not at import time."""
    from langgraph.graph import END, StateGraph

    workflow = StateGraph(RAGState)
    workflow.add_node("detect_language", detect_language)
    workflow.add_node("rewrite_query", rewrite_query)
    workflow.add_node("embed_query", embed_query)
    workflow.add_node("retrieve_chunks", retrieve_chunks)
    workflow.add_node("check_retrieval_quality", check_retrieval_quality)
    workflow.add_node("rerank_chunks", rerank_chunks)
    workflow.add_node("build_context", build_context)

    workflow.set_entry_point("detect_language")
    workflow.add_edge("detect_language", "rewrite_query")
    workflow.add_edge("rewrite_query", "embed_query")
    workflow.add_edge("embed_query", "retrieve_chunks")
    workflow.add_edge("retrieve_chunks", "check_retrieval_quality")
    workflow.add_conditional_edges(
        "check_retrieval_quality",
        route_after_quality_check,
        {"rewrite_query": "rewrite_query", "rerank_chunks": "rerank_chunks"},
    )
    workflow.add_edge("rerank_chunks", "build_context")
    workflow.add_edge("build_context", END)

    return workflow.compile()


def run_pipeline(query: str, document_id_filter: Optional[int] = None) -> dict:
    initial_state = RAGState(
        query=query,
        language="",
        rewritten_query="",
        rewrite_count=0,
        query_vector=[],
        retrieved_chunks=[],
        reranked_chunks=[],
        retrieval_score=0.0,
        context="",
        needs_rewrite=False,
        document_id_filter=document_id_filter,
    )
    return get_pipeline().invoke(initial_state)
