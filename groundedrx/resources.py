"""Lazy-loaded, process-wide singletons for every GPU/disk-heavy resource.

Every getter here is `functools.lru_cache(maxsize=1)` — the underlying model
or index is loaded on first call and reused after that, matching the
notebook's original "load once in Setup, use everywhere" behavior, but
without paying that cost just from `import groundedrx`.

This is what makes the rest of the package (retrieval.py's pure functions,
grounding.py with an injected embedder, paths.py) importable and testable on
a CPU-only machine with none of torch/transformers/sentence-transformers
installed: nothing in this module runs at import time, only when a getter is
actually called. Heavy imports are deliberately deferred inside each
function for the same reason.
"""

from functools import lru_cache
from typing import List

from . import config
from .paths import resolve_store_path


@lru_cache(maxsize=1)
def get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(path=resolve_store_path())


@lru_cache(maxsize=1)
def get_embed_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("BAAI/bge-m3", device="cuda")


@lru_cache(maxsize=1)
def get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.CONFIG_C4["reranker_model"], device="cuda")


@lru_cache(maxsize=1)
def get_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(config.MODEL_NAME)


@lru_cache(maxsize=1)
def get_model():
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    # bf16 has more dynamic range than fp16 under 4-bit quant -- a safe
    # general default. Falls back to fp16 only on GPUs without native bf16
    # (e.g. T4).
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    bnb_config = (
        BitsAndBytesConfig(load_in_8bit=True)
        if config.USE_8BIT
        else BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    )
    # bitsandbytes' 8-bit kernel only supports fp16 inputs internally --
    # loading in bf16 makes it silently cast bf16->fp16 on every matmul.
    # 4-bit NF4 has no such restriction, so this only changes the 8-bit path.
    model_dtype = torch.float16 if config.USE_8BIT else compute_dtype

    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=model_dtype,
    )
    model.eval()
    return model


@lru_cache(maxsize=1)
def get_bm25_docs() -> List[dict]:
    """Every chunk's payload, pulled once via a full collection scroll."""
    points, _ = get_client().scroll(
        collection_name=config.CONFIG_C4["collection_name"],
        limit=10000,
        with_payload=True,
        with_vectors=False,
    )
    return [
        {
            "id": p.id,
            "text": p.payload.get("chunk_text", ""),
            "language": p.payload.get("language", ""),
            "category": p.payload.get("category", ""),
            "document_id": p.payload.get("document_id", ""),
            "chunk_id": p.payload.get("chunk_id", ""),
            "file_name": p.payload.get("file_name", ""),
        }
        for p in points
    ]


@lru_cache(maxsize=1)
def get_bm25_index():
    from rank_bm25 import BM25Okapi

    from .retrieval import _tokenize

    return BM25Okapi([_tokenize(d["text"]) for d in get_bm25_docs()])
