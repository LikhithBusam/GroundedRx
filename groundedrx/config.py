"""All tunable configuration in one place — the single source for retrieval,
rerank, rewrite, fusion, generation, and groundedness-gate behavior.

Extracted verbatim from GroundedRx_Colab.ipynb (Components 4 and 5). Every
value and every comment here reflects a real, measured decision documented in
CLAUDE.md — none of these are placeholder defaults.
"""

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

# ponytail: 8-bit was tested (Falcon-H1) and REJECTED as a fix for Arabic
# generation quality -- fixed one symptom (CJK corruption) but worsened other
# Arabic signals and unexpectedly regressed English. Kept as a toggle for a
# possible future A/B on the current model, not because it's recommended.
# See CLAUDE.md "Arabic generation quality investigation".
USE_8BIT = False

CONFIG_C4 = {
    "collection_name": "peach_healthcare_multilingual",
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "top_k_retrieve": 20,
    "top_k_rerank": 5,
    "score_threshold": 0.5,
    "max_rewrites": 2,
    "rrf_k": 60,  # Reciprocal Rank Fusion damping (standard default)
}

CONFIG_GATE = {
    # ponytail: CALIBRATION KNOB, not a derived constant. Observed grounded
    # answers score 0.601-0.883 against real bge-m3 embeddings; this is a
    # starting guess, not fit against a labeled dataset.
    "min_sentence_similarity": 0.50,
    "min_sentence_chars": 25,  # shorter fragments are punctuation noise
    "block_on_fail": True,  # ungrounded answer -> replaced by refusal
}

REFUSAL = {
    "en": "I don't have enough information to answer this question.",
    "ar": "لا أملك معلومات كافية للإجابة على هذا السؤال.",
}

PROMPT_EN = """You are a bilingual medical assistant specializing in patient information leaflets.
Answer the question using ONLY the information provided in the context below.
If the answer is not found in the context, say exactly: "I don't have enough information to answer this question."
Do NOT add any medical information not present in the context.

Context:
{context}

Question: {query}

Answer:"""

PROMPT_AR = """أنت مساعد طبي متخصص في نشرات معلومات المرضى.
أجب على السؤال باستخدام المعلومات الواردة في السياق أدناه فقط.
إذا لم تكن الإجابة موجودة في السياق، قل بالضبط: "لا أملك معلومات كافية للإجابة على هذا السؤال."
لا تضف أي معلومات طبية غير موجودة في السياق.

السياق:
{context}

السؤال: {query}

الإجابة:"""
