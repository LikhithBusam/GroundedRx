"""End-to-end glue: retrieve -> rerank -> generate -> groundedness gate.
Ported from GroundedRx_Colab.ipynb Component 5, Cell 3. Uses the
NLI-augmented gate (nli.check_grounding_nli), which wraps grounding.
check_grounding and fails soft to it if the NLI model is unavailable.
"""

import logging
from typing import Optional

from . import config
from .generation import generate_answer
from .nli import check_grounding_nli
from .retrieval import run_pipeline

logger = logging.getLogger(__name__)


def rag_answer(query: str, document_id_filter: Optional[int] = None) -> dict:
    """End-to-end RAG: retrieve → rerank → generate → groundedness gate."""
    pipeline_result = run_pipeline(query, document_id_filter=document_id_filter)
    context = pipeline_result["context"]
    language = pipeline_result["language"]
    generation = generate_answer(query, context, language)

    # Gate judges against generation["context_used"] -- the truncated text
    # the model actually saw. Judging against the full context would credit
    # the model for content it was never shown.
    verdict = check_grounding_nli(generation["answer"], generation["context_used"], language)

    answer = generation["answer"]
    if config.CONFIG_GATE["block_on_fail"] and not verdict["grounded"]:
        logger.warning(
            f"Gate BLOCKED answer: {verdict['reason']} | bad numbers={verdict['hallucinated_numbers']}"
        )
        answer = config.REFUSAL.get(language, config.REFUSAL["en"])

    return {
        "query": query,
        "language": language,
        "answer": answer,
        "answer_raw": generation["answer"],  # pre-gate, kept for inspection
        "grounding": verdict,
        "retrieval_score": pipeline_result["retrieval_score"],
        "rewrite_count": pipeline_result["rewrite_count"],
        "reranked_chunks": pipeline_result["reranked_chunks"],
        "input_tokens": generation["input_tokens"],
        "output_tokens": generation["output_tokens"],
    }


def print_result(result: dict) -> None:
    """Pretty-print a rag_answer() result -- used by the notebook's smoke
    tests, not by the API (which returns the dict as JSON directly)."""
    print(f"Query          : {result['query']}")
    print(f"Language       : {result['language'].upper()}")
    print(f"Retrieval score: {result['retrieval_score']:.4f}")
    print(f"Input tokens   : {result['input_tokens']}")
    print(f"Output tokens  : {result['output_tokens']}")
    g = result["grounding"]
    flag = "PASS" if g["grounded"] else "BLOCKED"
    print(f"Groundedness   : {flag} | min sentence sim {g['min_similarity']:.3f} | {g['reason']}")
    if g["hallucinated_numbers"]:
        print(f"Hallucinated numbers: {g['hallucinated_numbers']}")
    print(f"Answer         : {result['answer']}")
