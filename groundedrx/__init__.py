"""GroundedRx — bilingual (Arabic/English) medical RAG with a runtime
groundedness gate. See the repo README for the full picture; this package is
the extracted, importable, testable form of GroundedRx_Colab.ipynb's
pipeline (Components 4 and 5).

Importing this package is CPU-safe and does not load any model or open the
vector store -- every GPU/disk-heavy resource in `resources.py` is loaded
lazily on first use. Actually calling `rag_answer()` requires the `gpu`
dependency extra installed and a CUDA GPU available.
"""

from .config import CONFIG_C4, CONFIG_GATE, MODEL_NAME
from .grounding import check_grounding
from .pipeline import print_result, rag_answer
from .retrieval import run_pipeline

__version__ = "0.1.0"

__all__ = [
    "rag_answer",
    "print_result",
    "run_pipeline",
    "check_grounding",
    "CONFIG_C4",
    "CONFIG_GATE",
    "MODEL_NAME",
]
