# GroundedRx

**A bilingual (Arabic/English) medical RAG system with a runtime groundedness gate that
blocks unsupported answers before they reach the user.**

Answers patient questions about medications from a corpus of patient information leaflets,
in either Arabic or English, using hybrid (dense + BM25) retrieval and a locally-hosted
Qwen2.5-7B-Instruct model. Every generated answer is checked against its retrieved source
text before being shown — if the check fails, the user gets a refusal, not a guess.

> Was named "FalconMed AI" — renamed after a model swap (Falcon-H1 → Qwen2.5) left the old
> name referring to an architecture no longer in use.

**Not medical advice. A portfolio/demo project, not a certified clinical tool.**

---

## Why this project is interesting

- **Fully self-hosted, on-premises-capable.** No call to OpenAI, Anthropic, or any external
  LLM API. The generation model, the embedding model, the reranker, and the vector store are
  all local. For a medical-data system, that means patient-leaflet content and queries never
  leave the deployment environment. (Docker packaging for an actual on-prem deployment is
  in progress — see [Status](#status--what-actually-exists-today) below.)
- **True cross-lingual retrieval, measured, not assumed.** An Arabic query can surface English
  source chunks and vice versa without translating first — verified directly (see
  [Results](#results-real-numbers)), not just claimed because the embedding model is
  "multilingual."
- **A groundedness gate that has actually caught a real fabrication.** Not a theoretical
  safety feature — on the 12-question evaluation set it blocked an answer that invented a
  contraindication number absent from the source context, and it fails closed (any internal
  error → refusal, never a silent pass).
- **Two precisely diagnosed failure modes, found through live testing, documented rather than
  hidden.** See [Known Limitations](#known-limitations).
- **A disciplined "reject when it doesn't work" experiment history**, not just a working
  pipeline: greedy decoding and 8-bit quantization were both tried against a real quality
  problem (Arabic generation corruption) and both explicitly rejected with numbers, before
  a model swap was attempted. See `PROJECT_REPORT.md`.

## Architecture

```
detect_language → rewrite_query → embed_query → retrieve_chunks → check_retrieval_quality
                        ^                                                    |
                        └──────────────── (score < 0.5, retries < 2) ────────┤
                                                                              v
                                                            rerank_chunks → build_context
                                                                              |
                                                                              v
                                                    generate_answer → check_grounding → answer
```

- **Retrieval:** dense (BAAI/bge-m3 embeddings → Qdrant, cosine) fused with sparse (BM25) via
  Reciprocal Rank Fusion — dense alone fuzzes exact terms like dosage numbers, BM25 alone is
  useless on generic phrasing like "side effects"; fusion covers both.
- **Reranking:** BAAI/bge-reranker-v2-m3 cross-encoder, top-20 → top-5.
- **Generation:** Qwen2.5-7B-Instruct, 4-bit NF4 quantized, greedy decoding, context-only
  prompting with an explicit refusal instruction.
- **Groundedness gate:** two deterministic checks (not an LLM self-judge, deliberately —
  see `CLAUDE.md`) — every number in the answer must appear in the retrieved context, and
  every answer sentence must have a high-cosine match against a context sentence. Majority-vote
  blocking, fails closed on any error.

Full design rationale, every rejected alternative, and the reasoning behind each choice is in
`CLAUDE.md`.

## Results (real numbers)

All numbers from actual runs over a 12-question (7 EN + 5 AR) evaluation set, each question
grounded in one specific real document (not generic phrasing). Full methodology and every
number in `PROJECT_REPORT.md`.

| Metric | English | Arabic |
|---|---|---|
| BERTScore (judge-independent) | 0.899 | 0.637 |
| Cross-lingual retrieval share (raw query, unfiltered) | — | ~60% of top-20 crossed the language boundary |
| Groundedness gate | caught 1/12 real fabrications (invented contraindication number) | fails closed on internal error |

**Judge-based metrics (DeepEval, LLM-as-judge) are reported separately and with a caveat, not
folded into the table above.** After the Qwen2.5-7B model swap, judge-routed scores dropped
sharply (e.g. DeepEval Answer Relevancy 0.733 → 0.394) while BERTScore — the one
judge-independent signal — stayed flat-to-improved. Because the same model both generates
*and* judges in this design, a model swap changes both sides of the comparison at once. The
most defensible reading is a **self-judging confound** (a more capable model scoring its own
output more strictly), not a real quality regression — but this is disclosed as an open
question, not resolved. See `PROJECT_REPORT.md` §8 for the full reasoning and what would be
needed to resolve it (an independent third judge, or a human rating pass — neither done yet).

## Known Limitations

The groundedness gate is a **fabrication filter, not a faithfulness guarantee.** Two specific,
evidenced ceilings, both found through live testing rather than theorized:

1. **Cannot catch negation.** Cosine similarity measures topical overlap, not logical
   entailment — *"take this with alcohol"* against context *"do NOT take this with alcohol"*
   scores high and passes.
2. **Cannot catch wrong-drug substitution.** Asking about a drug not in the corpus (e.g.
   ibuprofen) can retrieve an unrelated drug's leaflet whose dosing section is phrased
   similarly enough to rank top-1, and the model will answer with that drug's real numbers —
   correctly named, but not what was asked about. The gate scores this "grounded" because
   every claim genuinely is supported by the retrieved text; it has no concept of
   query-vs-document identity. Root cause is one layer upstream, in retrieval, which matches
   on phrasing structure, not drug identity.

Both are documented in detail, with the exact reproducing query, in `CLAUDE.md` and
`PROJECT_REPORT.md`.

## Status — what actually exists today

This is being built incrementally in the open. Current state:

- ✅ Full RAG pipeline, working and evaluated (retrieval, reranking, generation, groundedness
  gate)
- ✅ Evaluation harness (BERTScore, DeepEval, LLM-as-judge)
- ✅ A Gradio demo UI (`GroundedRx_Colab.ipynb`, Component 7) — session-based public link,
  not a permanent deployment
- ⬜ Installable Python package (currently notebook-only)
- ⬜ Automated tests / CI
- ⬜ FastAPI service + on-premises Dockerfile
- ⬜ Permanent hosted demo

## Running it

Requires a CUDA GPU (the model is served via `transformers` + 4-bit `bitsandbytes`) — this
does not run on a CPU-only machine. Developed against Google Colab's free T4 tier, verified
portable to Kaggle.

1. Open `GroundedRx_Colab.ipynb` in Colab or Kaggle.
2. Upload `qdrant_db_archive.zip` (Colab: to `/content/`; Kaggle: as an attached Dataset —
   see `CLAUDE.md` for the platform differences).
3. Run cells top to bottom: Setup → Component 4 (retrieval) → Component 5 (generation) →
   Component 7 (optional Gradio demo UI).

Full instructions, including the Colab/Kaggle differences and every cell's purpose, are in
`CLAUDE.md`.

## Repository contents

- `GroundedRx_Colab.ipynb` — the notebook to run.
- `CLAUDE.md` — full technical documentation: every design decision, every rejected
  alternative, every bug found and fixed, with reasoning.
- `PROJECT_REPORT.md` — the evaluation report: every experiment run, every number measured.
- `qdrant_db_archive/` / `qdrant_db_archive.zip` — the pre-built vector store (2,365 chunks,
  464 documents).

## License

MIT — see `LICENSE`.
