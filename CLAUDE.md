# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GroundedRx** — a bilingual (Arabic/English) medical RAG system built over the PEACH RAG
Dataset (patient information leaflets). Was named "FalconMed AI"; renamed after the
Falcon-H1 → Qwen2.5 model swap left the old name referring to an architecture no longer in
use — see "Model swap" below. Originally scoped toward an ArabicNLP Workshop submission
(ACL/EMNLP); that framing is superseded — current direction is a production-grade,
portfolio-ready system (packaged library, tests, API, on-premises Docker deployment), not a
paper. The core RAG pipeline and evaluation harness described below are unaffected by that
change; only the intended audience/output changed. The entire pipeline currently lives in a
Google Colab notebook (also runs on Kaggle — see "Running the Notebook" below) — there is no
package structure, build system, or test suite yet. Requires a CUDA GPU
(the generation model is served via `transformers` + 4-bit `bitsandbytes`, which is
CUDA-only), so this only runs in Colab/Kaggle, not on a local CPU-only machine. Generation
model is `Qwen/Qwen2.5-7B-Instruct` (`MODEL_NAME` in Setup) — was
`tiiuae/Falcon-H1-1.5B-Deep-Instruct`, swapped after two decoding/quantization experiments
failed to fix Arabic quality (see "Arabic generation quality investigation" below).

**Note:** an SGLang-based generation backend was tried and reverted (see Architecture below,
"Backend history") — it does not work on Colab free's default image, whose preinstalled
RAPIDS/cuDF/cuML/numba/dask-cuda stack is tightly pinned to CUDA 12.x and conflicts with the
torch/CUDA-13/numpy≥2.5 combination `sglang[all]` wants.

## Package (`groundedrx/`)

The retrieval/generation/grounding pipeline also exists as an installable package under
`groundedrx/`, extracted from the notebook (Components 4 and 5) as faithfully as possible —
same logic, same comments, same documented bug fixes, just split into real modules
(`config.py`, `paths.py`, `resources.py`, `retrieval.py`, `generation.py`, `grounding.py`,
`pipeline.py`) with `functools.lru_cache`-based lazy loading so `import groundedrx` needs no
GPU and no torch/transformers/sentence-transformers install. `rrf_fuse()` (retrieval.py) is a
pure function taking plain dicts, and `check_grounding()` (grounding.py) takes an injectable
`embedder` parameter — both specifically so they're unit-testable on a CPU-only machine
(`tests/`, 20 tests, all passing, no GPU/model downloads, <1s to run).

**Deliberate, current limitation: the notebook does NOT import from this package.** It keeps
its own copy of the same logic. Rewriting the notebook to `pip install -e .` and import from
`groundedrx/` instead would eliminate the duplication, but doing that blind — without a live
Kaggle GPU run to verify nothing broke — is a real risk with no local GPU available to catch
it first. Until that verification pass happens, the notebook remains the trusted,
independently-tested-on-real-GPU artifact, and the package is a separate, tested foundation
for what comes next (e.g. the planned FastAPI service). If you change pipeline behavior, both
copies currently need the same fix — check both `GroundedRx_Colab.ipynb` and `groundedrx/`
until they're unified.

## CI (`.github/workflows/ci.yml`)

Runs on every push/PR to `main`, Python 3.10 and 3.11, all CPU-only — no GPU, no model
downloads, no vector store needed: `ruff check` (lint, default E/F/I rules only — see
"tooling philosophy" note below), the full `pytest tests/` suite (20 tests), and
`scripts/check_notebook.py` (verifies every notebook code cell still parses as valid Python,
the same `ast.parse()` check used by hand throughout this project's development, now
automatic on every push instead of relying on someone remembering to run it). Deliberately
does **not** run the notebook itself — Setup/Component 4/5/6 all require a CUDA GPU CI
runners don't have.

Tooling is deliberately minimal: ruff only, no black/isort/mypy/pre-commit stack. `E501`
(line-too-long) is disabled specifically for `groundedrx/config.py`, since its prompt
templates are triple-quoted strings sent verbatim to the LLM — a `# noqa` comment can't be
added inside a string literal without corrupting the actual prompt text, so the whole file is
exempted rather than reflowing prompt content to satisfy a linter.

## Repository Contents

- `GroundedRx_Colab.ipynb` — **the notebook to run.** A cleaned, linearly-ordered rebuild of
  `Untitled1_(1).ipynb`: title/instructions cell → Setup → Component 4 (retrieval) →
  Component 5 (generation) → Component 6 (evaluation). Each cell is the verbatim
  best-working version pulled from the original notebook (see below), with only
  Colab-install-line fixes applied.
- `Untitled1_(1).ipynb` — the original prototype, kept as historical reference. Cells are
  **not in a clean linear order** — later cells are reworked/patched versions of earlier
  ones (e.g. cell 15 is a leaner rewrite of cells 4+5 combined, cells 7–9 are abandoned
  dependency-fix attempts for cell 6's RAGAS evaluation, cells 1/3 are superseded by cell
  14's self-contained setup). Prefer `GroundedRx_Colab.ipynb` over editing this file directly.
- `qdrant_db_archive/` — an on-disk Qdrant vector store (local-mode, file-based, not a
  server) containing the pre-built collection `peach_healthcare_multilingual`.
  - `meta.json` — collection config: 1024-dim vectors, Cosine distance.
  - `collection/peach_healthcare_multilingual/storage.sqlite` — the actual vector/payload data.
- `qdrant_db_archive.zip` — zipped copy of the above, ready to upload to Colab's `/content/`.

## Running the Notebook

There is no local runner — this requires a CUDA GPU (developed against Colab, since verified
portable to Kaggle). To execute:

**On Colab:**
1. Open `GroundedRx_Colab.ipynb` in Colab. Runtime → Change runtime type → GPU.
2. Upload `qdrant_db_archive.zip` to `/content/` (Files panel).
3. Run cells top to bottom (see table below).

**On Kaggle** (motivated by Colab free's opaque, session-based GPU quota vs. Kaggle's
published ~30 hrs/week — same single T4 per session, not more powerful, just more
predictable for sustained work):
1. Notebook Settings → Internet → **On** (off by default; needed for `pip install` and
   model downloads).
2. Add Data → Upload → `qdrant_db_archive.zip`, attached as a Kaggle Dataset. **Kaggle
   auto-extracts any `.zip` uploaded as a Dataset** — unlike Colab, there is never a raw zip
   to unzip; what's mounted is the already-extracted store (`meta.json`, `collection/`,
   `.lock`) directly. `/kaggle/input/` is also **read-only**, but Qdrant's local client needs
   to write a `.lock` file into the store directory to open it — so Setup searches for
   `meta.json` (the store's own marker file, not a zip) under `/kaggle/input/`, then
   **copies** the whole containing folder into the writable `/kaggle/working/`, skipping the
   copy if it already exists (idempotent across re-runs in the same session).
3. Run cells top to bottom — Setup auto-detects Kaggle via `os.path.exists("/kaggle/input")`.
   No manual path edits needed; the Colab path (raw zip → `!unzip`) is unchanged and still
   fully supported — this is a portability addition, not a migration.
   - **Unverified:** Kaggle's preinstalled package/CUDA stack has not been tested against
     this notebook's `pip install` list. If something conflicts, diagnose the actual
     conflict — don't guess-fix it (see "Backend history" below for what guess-fixing a
     stack conflict cost on Colab with SGLang).

Cells, either platform:
   - **Setup** — installs deps, unzips the Qdrant archive, loads `client`, `embed_model`,
     `reranker`, `tokenizer`, and `model` (`Qwen/Qwen2.5-7B-Instruct` via `MODEL_NAME`,
     4-bit NF4 via `bitsandbytes` by default). `USE_8BIT` (default `False` — reverted after
     the 8-bit experiment on Falcon-H1 was rejected; toggle preserved for a future A/B on the
     new model if needed) switches to 8-bit — see "Arabic generation quality investigation"
     below for why this exists. **The `causal-conv1d`/`mamba-ssm` fallback line that used to
     live here has been removed entirely**, not just commented — it was a fix for Falcon-H1's
     Mamba2 layers specifically, and Qwen2.5 (standard attention transformer) has no SSM
     layers for it to apply to.
   - **Component 4** — builds `rag_pipeline` / `run_pipeline` (LangGraph retrieval graph),
     runs 5 smoke-test queries. Tests 1–4 are generic phrasing; **test 5 is an exact-term
     query** (`"What is the lisinopril 10 mg dose?"`) and reports how many chunks BM25
     contributed that dense retrieval missed. Without test 5 hybrid search is unfalsifiable
     from the smoke tests — tests 1–4 are expected to look identical with or without fusion,
     since `retrieval_score` is the dense cosine by design and generic phrasing is exactly
     where BM25 adds nothing.
   - **Component 4b** — cross-lingual retrieval verification (see below). Retrieval stack
     only, no Falcon — runs in seconds and is safe to re-run while iterating.
   - **Component 5** — defines `generate_answer` / `rag_answer`, runs 5 smoke-test queries.
   - **Component 6** — runs the full BERTScore + DeepEval + LLM-as-judge evaluation over
     `EVAL_QA` (12 AR/EN questions), saves `evaluation_results.csv` +
     `evaluation_summary.json`. Per-metric failures are caught individually (see below) so
     one bad DeepEval parse on one question doesn't abort the whole eval run.
4. Call `rag_answer("<question>")` directly for one-off end-to-end retrieve → rerank → generate.
   - **Component 7** (optional) — a Gradio UI over `rag_answer`, run after Components 4/5.
     `demo.launch(share=True)` tunnels through Gradio's own servers, so it works unmodified on
     both Colab and Kaggle with no extra deployment step. Shows the answer, the grounding
     verdict (pass/blocked + reason + min sentence similarity), and the retrieved source
     chunks (`file_name`, `category`, rerank score). The link only lasts for the session —
     it's a live demo of the running notebook, not a hosted deployment (that's separate,
     later work: `groundedrx/` package + FastAPI + Docker, not yet started).

There are no lint/format/test commands — this is exploratory notebook code, not a package.

## Architecture

The pipeline is a LangGraph `StateGraph` (`RAGState` TypedDict) with a feedback loop, defined
in Component 4 and consumed by Component 5:

```
detect_language → rewrite_query → embed_query → retrieve_chunks → check_retrieval_quality
                        ^                                                    |
                        └──────────────── (score < 0.5, retries < 2) ────────┤
                                                                              v
                                                            rerank_chunks → build_context → END
```

- **detect_language** — `langdetect`, maps to `"ar"` / `"en"` (default `"en"` on failure);
  drives which prompt template and rewrite-terms are used downstream.
- **rewrite_query** — **pass 0 returns the query unchanged**; only retry passes append static
  medical-domain keyword expansions (different for AR/EN). Not an LLM rewrite. It used to
  expand on every pass, which Component 4b measured as cutting cross-lingual retrieval for
  *Arabic* queries roughly in half (60% → 33% of top-20 crossing the language boundary) by
  pulling the embedding back toward the query's own language — English queries were
  unaffected, making the damage asymmetric and invisible to EN-only testing. Raw queries
  already clear the 0.5 quality gate (cosine 0.63–0.73), so expansion is only worth its cost
  after the gate fails. This also revived the feedback loop: the expansion is built from
  `state["query"]` (always raw), so both passes previously produced a byte-identical string
  and the retry re-ran identical retrieval — it could never change its own outcome.
- **embed_query** — `BAAI/bge-m3` SentenceTransformer, `normalize_embeddings=True` (must match
  the model/normalization used when the Qdrant collection was originally indexed).
- **retrieve_chunks** — **hybrid**: dense (`client.query_points` against
  `peach_healthcare_multilingual`, top-20, cosine over 1024-dim vectors) fused with sparse
  (BM25 over `chunk_text`) via Reciprocal Rank Fusion. Payload fields: `chunk_text`,
  `language` ("English"/"Arabic"), `category`, `document_id`, `chunk_id`, `file_name`.
  - The BM25 index (`bm25_index` / `BM25_DOCS`, built once in Component 4 from a full
    `client.scroll`) exists because dense embeddings fuzz exact tokens. Measured on this
    corpus: BM25 ranks document 502 — the actual lisinopril leaflet — top for
    `"lisinopril dose 10 mg"`, while bge-m3 alone spreads that across dosage text from
    unrelated drugs. Conversely BM25 is useless on generic phrasing (`"side effects"` scores
    ~7.9 and scatters), which is dense retrieval's strength. Complementary, hence fusion.
  - **Fusion is on rank, not score.** Cosine is bounded 0–1 while BM25 is unbounded and its
    scale shifts per query, so a score-weighted blend would need per-query normalization;
    RRF (`rrf_k` = 60 in `CONFIG_C4`) sidesteps that entirely. The join key is the Qdrant
    **point id** — verified unique across all 2,365 chunks.
  - **`retrieval_score` deliberately stays the best _dense cosine_**, not the RRF score. The
    `check_retrieval_quality` gate is calibrated against cosine (threshold 0.5); an RRF score
    (max ~0.016) would make it fire on every query and a raw BM25 score on none. Fusion
    changes *what is retrieved*, not *how retrieval is judged*.
  - Chunks carry `score` (dense cosine, `0.0` if the chunk was found by BM25 only),
    `bm25_score`, and `rrf_score` for inspection.
  - `document_id_filter` constrains **both** halves — the sparse candidate list is filtered
    before ranking, so the eval-only pinning can't leak through the BM25 path.
  - Known limitation: BM25 is lexical, so it cannot match a Latin-script drug name against an
    Arabic leaflet spelling it in Arabic script (`"Logynon"` scores 0.00 — the token is absent
    from the index). Cross-language matching remains the dense half's job; because RRF is
    additive this costs nothing that wasn't already the case.
- **check_retrieval_quality** — feedback-loop gate: if best score < `score_threshold` (0.5)
  and `rewrite_count < max_rewrites` (2), routes back to `rewrite_query`; otherwise proceeds
  to reranking. Config lives in `CONFIG_C4`.
- **rerank_chunks** — `BAAI/bge-reranker-v2-m3` CrossEncoder, top-20 → top-5.
- **build_context** — formats top-5 reranked chunks into a numbered, labeled context block
  (language-aware headers, e.g. `[Source N]` vs `[المصدر N]`).
- **generate_answer** (Component 5) — **`Qwen/Qwen2.5-7B-Instruct`, loaded 4-bit NF4**
  (`MODEL_NAME` in Setup; was `tiiuae/Falcon-H1-1.5B-Deep-Instruct`). **Model swap** — see
  "Arabic generation quality investigation" below for the full chain of evidence: two
  experiments (greedy decoding, 8-bit quantization) each failed to fix Falcon-H1's Arabic
  quality problems and converged on model capacity as the limitation, not a decoding or
  quantization setting. Qwen2.5-7B chosen over an Arabic-specialized model (e.g. Jais) as the
  first test: standard attention-only transformer (sidesteps the Mamba2-specific issues
  below entirely), mainstream `transformers`/`bitsandbytes` support, strong documented
  multilingual/Arabic benchmarks. `trust_remote_code` no longer needed (was required for
  Falcon-H1's custom architecture code; Qwen2.5 has been in mainline `transformers` for a
  while). **Evaluated against Component 6 — result is genuinely ambiguous, not a clean win
  or loss.** See "Arabic generation quality investigation" below for the full breakdown:
  BERTScore (judge-independent) says quality is flat-to-improved, but every judge-based
  metric collapsed — most likely a self-judging confound (the model changed on both the
  generation and the judging side at once), not a real quality regression. Do not cite the
  judge-based numbers as "Qwen generates worse answers" without that caveat.

  Uses separate AR/EN prompt templates that instruct the model to answer **only** from the
  provided context and to emit an exact "I don't have enough information" refusal string
  (AR/EN) otherwise — context-only prompting to suppress hallucination, model-agnostic.
  Context is truncated to 2000 chars to stay within the input budget. **Must go through
  `tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, ...)`**,
  not a raw `tokenizer(prompt)` call — this remains true for any instruct-tuned chat model,
  Qwen2.5 included, not just the Falcon-H1-specific case that originally surfaced it (skipping
  it made Falcon-H1 hallucinate literal `assistant` role-turn text and repeat its own answer;
  same raw-completion pattern was also present in Component 6's LLM-judge scoring loop and
  DeepEval wrapper, fixed the same way there).

  **Historical, Falcon-H1-specific — kept for context, does not describe the current model:**
  the `bnb_4bit_compute_dtype` bf16-preferred-else-fp16 logic (`torch.cuda.is_bf16_supported()`)
  was originally motivated by fp16 overflowing to NaN/Inf mid-generation *on Falcon-H1's
  hybrid architecture specifically* (first as garbled mixed-language tokens, then a hard
  `AcceleratorError: device-side assert triggered` inside `torch.multinomial`, which poisons
  the CUDA context for the rest of the runtime — restart the session, don't just re-run the
  cell); bf16-when-available is kept as a safe general default under the new model, but this
  specific overflow bug has no documented history on a standard attention transformer.

  **Backend history (transformers → SGLang → transformers), Falcon-H1-specific:** Component 6's
  DeepEval/LLM-judge calls kept hitting CUDA OOM even after the bf16 fix above: the actual
  cause was `transformers` falling back to a **naive, non-optimized SSM scan** for Falcon-H1's
  Mamba2 layers (no `causal-conv1d`/`mamba-ssm` kernels installed), which scales memory
  roughly `O(seq_len²)` — visible directly in the OOM traceback as a
  `(batch, chunks, seq_len, seq_len, heads, state_dim)` intermediate tensor. SGLang shipped
  native optimized Falcon-H1 hybrid-attention/Mamba2 kernels in October 2025 and looked like
  the fix, so the whole generation backend was switched to SGLang's offline `Engine` API —
  **this was tried and reverted.** `pip install "sglang[all]"` pulled a torch/CUDA-13/numpy≥2.5
  combination that conflicts with Colab free's preinstalled RAPIDS/cuDF/cuML/numba/dask-cuda
  stack (all pinned to CUDA 12.x). Each attempted fix (reinstall torchvision, then force-reinstall
  torch+torchvision) broke a *different* part of the environment — CUDA major-version mismatch,
  then `transformers` losing its (apparently hard, not optional) torchvision dependency, then
  numpy getting bumped to 2.5.1 and breaking cuDF/cuML/numba/dask-cuda/langchain-openai/moviepy
  across the whole Colab image, while landing torch on a version SGLang itself didn't support
  (`sglang 0.5.16 requires torch==2.11.0`). Three fixes, three different failure sites — an
  architectural incompatibility, not a version pin to chase further, so the backend was
  reverted to `transformers` + `bitsandbytes`. **Qwen2.5 has no Mamba2/SSM layers**, so this
  entire OOM mechanism doesn't apply — kept here because the revert decision (avoid another
  SGLang-style dependency cascade) directly informed the "standard architecture" criterion
  used to pick Qwen2.5 over other candidates.

  **Current OOM mitigations** (all still active in the reverted version):
  `os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"` (Setup, reduces
  fragmentation from many small `generate()` calls); both BERTScore calls forced to
  `device="cpu"` (Component 6, frees ~2GB that isn't needed on GPU); `torch.cuda.empty_cache()`
  after each question in the DeepEval loop; `max_length`/`max_new_tokens` capped tighter for
  DeepEval's judge calls (1024/128) and the LLM-judge loop (1024/50) than for `generate_answer`
  itself (2048/300), since the naive SSM scan's `O(seq_len²)` cost makes DeepEval's
  multi-chunk-context calls the most expensive by far. If Component 6 still OOMs on a genuinely
  clean session after all of this, the next lever is the commented-out
  `causal-conv1d`/`mamba-ssm` install in the Setup cell — the actual fix for the root cause
  (near-linear memory scaling instead of quadratic), lower blast-radius than SGLang since it
  compiles against the existing torch install rather than requiring a specific torch/CUDA
  version, but has a documented history of wheel-build failures on Colab specifically; a failed
  install here is self-contained (clear pip error, nothing else breaks) unlike the SGLang
  cascade.
- **check_grounding** (Component 5, "CELL 2b") — **runtime groundedness gate**, applied to
  every live answer inside `rag_answer`. Deliberately *not* an LLM self-judge: DeepEval's
  `FaithfulnessMetric` failed on every question with this same 1.5B model regardless of token
  budget, a second Falcon generation per query is expensive on the naive SSM path, and any
  LLM judge reintroduces a parse-failure mode. Two deterministic checks instead:
  - **Numeric grounding** — every numeric literal in the answer must appear in the context.
    Targets the highest-consequence failure in a leaflet system, an invented dose. Arabic-Indic
    and Persian digits are normalized to ASCII first, so `١٠ مغ` and `10 mg` compare equal.
  - **Semantic grounding** — answer split into sentences (on Latin *and* Arabic terminators;
    `؟` is not `?`), each scored by max cosine against context sentences via the already-loaded
    `bge-m3`. No new model, no new dependency.
  - **Fails closed.** Empty context, no context sentences, or any exception ⇒ `grounded: False`.
    In offline eval a broken metric costs a data point; at runtime it would ship an unverified
    medical answer, so it must never default to pass.
  - A correct **refusal short-circuits to pass** — otherwise the model's own honesty would be
    flagged as a hallucination.
  - Judges against `generation["context_used"]` (the 2000-char *truncated* context the model
    actually saw), not the full context it never received.
  - `CONFIG_GATE["block_on_fail"]` (default `True`) replaces a failed answer with the
    language-matched refusal; the original is always preserved as `answer_raw`.
  - **`min_sentence_similarity` (0.50) is an uncalibrated guess, not a derived constant.**
    Observed on the Component 5 smoke tests: grounded answers scored 0.601–0.883, and the
    ordering tracked subjective answer quality (most precise answer 0.883, weakest 0.601).
    Component 6 prints a calibration block over the 12 `EVAL_QA` questions.
  - **Measured: the numeric check is the half that works.** On the 12-question `EVAL_QA` run
    it blocked 1/12 — *"What are the contraindications of Linopril?"*, which asserted a `4`
    absent from the context — at `sim=0.637`, i.e. **the semantic half had already passed it**.
    Across both the smoke tests and the eval run the semantic half has never once fired on
    real output. Treat numeric grounding as the load-bearing check and the cosine half as
    unproven.
  - **Fixed bug (was silently disabling the semantic check):** bulleted answers — which this
    model produces constantly in Arabic — split into fragments below `min_sentence_chars`,
    were filtered to an empty list, and the old code then returned `min_similarity: 1.0` and
    passed. A list-shaped hallucination bypassed the semantic check entirely while reporting
    a perfect score; this is why the first eval run showed a median `min_sim` of exactly
    1.000. The gate now falls back to scoring the whole answer as one unit, and a genuinely
    empty answer fails closed instead of passing vacuously.
  - **Fixed bug (refusal check was a bare substring test):** observed live — a real dosage
    answer (containing real, unverified numbers) added an aside quoting the refusal phrase
    verbatim ("...it is 'I don't have enough information...' since the context covers
    specific conditions only"). The old check (`REFUSAL in answer`) matched that quote and
    short-circuited to `grounded: True`, skipping **both** the numeric and semantic checks
    for an answer with real numbers in it — the fail-closed guarantee broke exactly where it
    matters. Now the refusal phrase must account for (nearly) the whole answer
    (`len(answer) <= len(refusal) * 1.5`), not merely appear somewhere inside it. The
    self-check block reproduces this exact case as a permanent regression test.
  - **Fixed bug (any single ungrounded sentence could veto an accurate answer):** observed
    live, twice, across two separate runs — a trailing "I don't have enough information..."
    hedge sentence, and separately a leading "According to the sources:" framing sentence,
    each blocked an otherwise well-grounded multi-sentence answer (accurate bullet points,
    correctly attributed to different source chunks) purely because that one *structural*
    sentence has no medical content of its own to match against context. Neither sentence is
    a claim; neither should be able to single-handedly discard several correct ones. Fixed by
    blocking on a **majority** of sentences failing (`len(ungrounded) > len(ans_sents) / 2`),
    not any single one. Still fails closed on real fabrication — a wrong single-sentence
    answer is still 100% ungrounded. **Known ceiling, not yet observed in practice:** exactly
    half-ungrounded on a very short (e.g. 2-sentence) answer does not trigger, since majority
    requires strictly >50%.
  - **Known ceiling: it cannot catch negation.** Cosine measures topical overlap, not
    entailment — *"take this with alcohol"* against context *"do NOT take this with alcohol"*
    scores high and passes. It is a **fabrication filter, not a faithfulness guarantee**.
    Closing that needs a real NLI model (e.g. an mDeBERTa XNLI checkpoint) as a second GPU
    resident.
  - **Known ceiling: it cannot catch wrong-drug substitution.** Observed live via the
    Component 7 demo: asking "What is the recommended dose of ibuprofen for a 5-year-old?"
    (ibuprofen is not in this corpus at all) retrieved an unrelated leaflet for a drug called
    "Batlor" (`document_id` from `11226.xlsx`) whose pediatric-dosing section is phrased
    similarly enough to rank top-1, and the model answered with Batlor's real dose — correctly
    *naming* it as Batlor, not fabricating an "ibuprofen dose," but never flagging that this
    isn't the medication asked about. The gate scored it `grounded: True` (0.75 similarity)
    because every claim in the answer genuinely is supported by the retrieved text — the gate
    only checks claim-vs-context support, it has no concept of query-vs-document identity, so
    this is invisible to it by design. Root cause is one layer upstream, in retrieval:
    dense/BM25 fusion matches on phrasing and semantic structure ("dose for a child of this
    age/weight"), not on drug identity, so a pediatric-dosing chunk for the wrong drug can
    outrank "nothing relevant found." This is the more consequential sibling of the negation
    gap above — a confident, real-numbers answer about the wrong medication is a worse failure
    mode than a refusal, since a user skimming the answer could easily miss that the drug name
    changed. Closing it would need a retrieval-time drug/entity check (e.g. verify the queried
    drug name — or its known synonyms/brand names — actually appears in the candidate chunk's
    `file_name`/`document_id` metadata before it's allowed to answer), not a generation-time or
    gate-time fix.
- **rag_answer** — glues Component 4 (`run_pipeline`) and Component 5 (`generate_answer`)
  into one end-to-end call, then applies `check_grounding`. Returns the delivered `answer`
  (post-gate), plus `answer_raw` and the `grounding` verdict.
  - **Component 6 consequence:** `generate_eval_answers` calls `rag_answer`, so `eval_results`
    `"answer"` is the *post-gate* answer. If the gate blocks, BERTScore/DeepEval score a
    refusal string and a metric drop becomes ambiguous — worse generation, or the gate firing?
    `eval_results` therefore also carries `answer_raw`, `gate_blocked`, `gate_reason`,
    `gate_min_sim`, and `gate_bad_numbers`. **Always read the metric drop and the block count
    together.**

Component 4b (cross-lingual retrieval verification) exists because "retrieves across languages
directly instead of translating first" was an *assumed* property of bge-m3's multilingual
training, never measured. It matters that the corpus has **no parallel AR/EN document pairs** —
all 464 `document_id`s are single-language — so the only thing "cross-lingual retrieval" can
mean here is a query in one language surfacing chunks written in the other. The cell runs 4
semantically-parallel AR/EN probe queries through three tests that deliberately separate
failure modes the normal pipeline conflates:
- **A — unfiltered language mix:** what fraction of the natural top-20 is in the *other*
  language. Run twice, on the raw query and on `rewrite_query`'s output, because the rewrite
  appends *language-specific* medical keywords (Arabic terms for AR, English for EN) and is the
  prime suspect for pinning results to the query's own language. A drop from raw → rewritten
  localizes the problem to `rewrite_query` rather than to the embedding model.
- **B — forced cross-language:** pins retrieval to the opposite `language` payload value and
  compares best cosine against same-language best. This separates *"bge-m3 cannot align these
  languages"* (large gap) from *"it can, but same-language chunks simply out-compete"* (small
  gap) — indistinguishable from test A alone.
- **C — reranker second opinion:** `bge-reranker-v2-m3` is also multilingual, so its score on
  the (query, cross-language chunk) pair independently confirms or rejects the embedding's
  judgment. A positive cosine with a negative rerank score means the embedding neighbourhood is
  an artifact, not real topical relevance.

Note these probes deliberately run **unfiltered by `document_id`** — the eval-only
`document_id_filter` used by `EVAL_QA` pins EN questions to doc 502 and AR to doc 498, which
makes cross-lingual retrieval structurally impossible to observe (each of those documents is
single-language). Component 4b therefore cannot be folded into the Component 6 eval loop.

Component 6 (evaluation) runs three metrics over a hardcoded 12-item `EVAL_QA` set (7 EN + 5
AR question/ground-truth pairs): BERTScore (per-language model — `roberta-large` for EN,
`bert-base-multilingual-cased` for AR, both forced to `device="cpu"` so they don't compete
for GPU memory with the generation model/`bge-m3`/`bge-reranker-v2-m3`), DeepEval's RAG
metrics (**only** `AnswerRelevancyMetric` — `FaithfulnessMetric`, `ContextualRecallMetric`,
and `ContextualPrecisionMetric` were all removed entirely, not just unreportable: measured
0/12, 3/12, and (under Qwen2.5-7B) 0/12 success respectively with this judge, and each failed
attempt still burned a multi-step internal generation chain before failing, ~26 minutes for a
single question's DeepEval section almost entirely spent on the first two; see "Measured
DeepEval reliability" below. The loaded model (`MODEL_NAME`) is wrapped as DeepEval's judge
via a `DeepEvalBaseLLM` subclass, `LocalDeepEvalModel` — calling `model.generate(...)`
through the same
`apply_chat_template` + greedy-decode pattern as `generate_answer`), and LLM-as-judge (the
model self-judges its own answers 1–5 on accuracy/safety/coherence via `JUDGE_PROMPT`,
JSON-parsed with a neutral 3/3/3 fallback on parse failure — same pattern). Each retrieved
chunk passed into DeepEval's `retrieval_context` is capped at 500 chars
(`[c[:500] for c in r["contexts"]]`) — unlike `generate_answer`'s context,
`eval_results["contexts"]` holds the raw untruncated chunk texts, and `context_precision`
processes all 5 chunks together, so this remains a reasonable safety margin against oversized
judge inputs independent of which model is loaded (originally motivated by Falcon-H1's
SSM-scaling issue specifically, but standard self-attention scales with sequence length too).

**Each `EVAL_QA` question is grounded in one specific real document**, not generic
"this medication" phrasing — EN questions target `document_id=502` (Linopril/lisinopril), AR
questions target `document_id=498` (Logynon, an oral contraceptive), and `ground_truth` is
copied verbatim from that document's actual chunk text. Earlier, generic questions like "What
are the side effects of this medication?" retrieved whichever of the corpus's 464 different
drugs was semantically closest, giving inconsistent grounding question to question and
language to language, against hand-typed ground truths that didn't necessarily match anything
actually retrievable from this corpus. Retrieval is pinned to the named document via
`retrieve_chunks`'s `document_id_filter` param (a Qdrant `Filter`/`FieldCondition` on the
`document_id` payload field), threaded through `RAGState` → `run_pipeline` → `rag_answer`, all
defaulting to `None`/unfiltered — **live queries from real end users still search the whole
corpus as normal**; only `generate_eval_answers`' loop over `EVAL_QA` passes a filter, since
it's the only caller that has a `document_id` to pass.

**RAGAS was replaced with DeepEval** — the original notebook's `ragas` install (cells 7–9)
failed on version conflicts with an old `ragas==0.1.21` pin; current `ragas>=0.4` was tried
next but has a packaging bug (`ragas/llms/base.py` unconditionally imports `ChatVertexAI`
from a `langchain_community` path removed after VertexAI support moved to
`langchain-google-vertexai` — see github.com/vibrantlabsai/ragas/issues/2741), and pinning
`ragas<0.4` didn't resolve cleanly in Colab either. DeepEval has no such import chain and
wraps a local HF model directly, so it was used instead. Each of the four DeepEval metrics is
wrapped in its own `try/except` per question (`deepeval_rows` loop) — a small local judge
model occasionally produces output DeepEval's own parser can't handle, and that should
degrade one metric on one question (score recorded as `None`, averaged out via pandas'
NaN-skipping `.mean()`) rather than aborting the whole `EVAL_QA` run.

**Measured DeepEval reliability with this judge — most of it is not usable.** On a full
12-question run: `faithfulness` **0/12** succeeded (mean is `nan`), `context_recall` **3/12**,
`context_precision` **11/12**, `answer_relevancy` **12/12**. Failures are `'verdicts'`
KeyErrors and *"Evaluation LLM outputted an invalid JSON"* — a 1.5B model cannot reliably
emit DeepEval's verdict schema, and this is **not** a token-budget problem (already ruled out
by raising `max_new_tokens` to 384 then 640).

**`faithfulness` and `context_recall` are no longer run at all** — not just unreportable, but
actively removed from `deepeval_metrics`. Each failed attempt still burns a multi-step
internal generation chain (claim extraction, then per-claim verdicts) before failing to
parse: measured at **~26 minutes for a single question's** full DeepEval section
(**measured under Falcon-H1** — its naive, un-accelerated SSM path was a major factor),
almost entirely spent on these two before they failed. Across 12 questions that's 4-5+
hours for data already known to be unusable.

**`context_precision` is also no longer run** — it worked reasonably under Falcon-H1
(10/12, after a token-budget fix from 384→512), but **failed 0/12 under Qwen2.5-7B**, every
failure the identical *"Evaluation LLM outputted an invalid JSON"* error, even at the same
512-token budget. In the same run, `answer_relevancy` (a simpler single-verdict JSON task)
succeeded 12/12 — so this isn't a general judge breakdown, it's specific to DeepEval's
multi-item verdict-list schema. Counter-intuitively, the *larger, more capable* model did
worse on this specific task than the smaller one — plausibly because Qwen wraps its answer
in more reasoning/preamble that breaks DeepEval's strict parser, though unconfirmed. That
makes three multi-item-verdict-list metrics (`faithfulness`, `context_recall`,
`context_precision`) that have now failed across both models tried in this project — a
consistent pattern, not per-model noise. **Only `answer_relevancy` has proven reliable with
either model and is the only DeepEval metric still run.**

## Arabic generation quality investigation

Arabic generation scores materially worse than English on every independent metric (BERTScore
0.649 vs 0.883, judge coherence 2.80 vs 4.57), with directly observable corruption — one
answer contained `ليس限اً`, a Chinese character (meaning "limit") embedded mid-Arabic where
the model was reaching for "not limited to." This is being investigated as a sequence of
controlled experiments, each isolating one variable, per a decision tree: if an experiment
measurably improves Arabic quality, stop and adopt it; if not, move to the next lever without
concluding the model itself is at fault until decoding *and* quantization have both been ruled
out.

**Experiment 1 — greedy decoding (`do_sample=False` in `generate_answer`): REJECTED.**
Hypothesis was that sampling under 4-bit NF4 quantization drew noisy low-probability Arabic
tokens. Tested and rejected on two independent grounds:
- The exact `ليس限اً` corruption reproduced **deterministically, twice**, in the same phrase,
  across two separate greedy runs. Greedy always takes the argmax — this is the model's own
  top-ranked choice at that position, not an unlucky sampling draw.
- Component 6 numbers moved the wrong way: AR BERTScore **0.649 → 0.630** (down), AR coherence
  **unchanged** (2.80 → 2.80), and the groundedness gate caught a **second** fabrication the
  sampled run didn't have (an invented "4" in the Logynon missed-dose answer). English metrics
  and retrieval-only metrics (Context Precision/Recall) were unchanged, confirming the
  comparison was clean — only decoding differed between runs.

`temperature`/`top_p`/`top_k` are fully removed from `generate_answer`; greedy is now
consistent with Component 6's judge and DeepEval calls, which already used it.

**Experiment 2 — 8-bit quantization (`USE_8BIT` in Setup): REJECTED.** Same model, greedy
held constant. `USE_8BIT = True` remains the default (kept for reference/reproducibility, not
because it's recommended) — both quantization paths stay in the code, this was a controlled
A/B toggle, not a one-way migration.

Measured on a full 12-question Component 6 run, confounds isolated (a self-inflicted judge
token-budget cut was found and fixed mid-investigation — see `PROJECT_REPORT.md` §7 for the
full before/after):

| Signal | Baseline | Greedy+8bit |
|---|---|---|
| AR BERTScore | 0.649 | 0.640 (down) |
| AR judge accuracy | 3.00 | 1.80 (down hard) |
| AR judge coherence | 2.80 | 3.40 (up — the one improvement) |
| EN judge accuracy/safety/coherence | 4.57/4.71/4.57 | 3.71/3.86/3.43 (all down — unexpected) |
| CJK token corruption (qualitative) | present | **absent in 3/3 checks** |

Two of three quantitative Arabic signals moved the wrong direction; only coherence improved.
The one clean win — no more CJK corruption — is real but narrow: it resolves one specific
symptom, not overall Arabic quality. English regressing on judge scores while BERTScore (an
independent encoder) stayed flat is a genuine, unresolved ambiguity: it suggests 8-bit may be
making **Falcon's own self-judging** less reliable rather than making its generation worse,
but this data can't distinguish the two, and no further run was designed to.

**This does not clear the "measurably improves Arabic quality" bar**, so per the decision
tree the next lever would have been bf16/fp16 — **skipped by deliberate choice**, not
oversight. Rationale: two experiments (decoding, then quantization) each failed to produce a
clean signal and each surfaced a *new* ambiguity instead of resolving the original one. That
pattern — more precision tuning producing more confusion rather than convergence — is itself
evidence worth acting on, not something to keep chasing through a third experiment.

**Conclusion: model capacity is the dominant limitation**, not decoding or quantization
settings. Next real lever is evaluating a stronger multilingual/Arabic-capable model.

**Experiment 3 — model swap to `Qwen/Qwen2.5-7B-Instruct`: AMBIGUOUS, not a clean result
either direction.** Full 12-question Component 6 run, greedy decoding, 4-bit NF4 (same
config category as the original baseline):

| Signal | Falcon-H1 baseline | Qwen2.5-7B |
|---|---|---|
| EN BERTScore (judge-independent) | 0.883 | **0.899 (up)** |
| AR BERTScore (judge-independent) | 0.649 | 0.637 (flat) |
| EN judge accuracy | 4.57 | **3.00 (collapsed)** |
| AR judge accuracy | 3.00 | **1.20 (collapsed further)** |
| AR judge coherence | 2.80 | **4.80 (all-time high)** |
| DeepEval Answer Relevancy | 0.733 | **0.394 (collapsed)** |
| DeepEval Context Precision | 0.594 (11/12) | **0/12 — total failure** |

**Qualitative smoke tests (Component 5, before this run) were the cleanest of the whole
investigation** — zero CJK corruption across multiple Arabic answers (first time all
session), all gate checks passed, fluent well-structured output. That qualitative signal
still stands; it's the quantitative judge-based numbers that don't.

**The load-bearing evidence: BERTScore (the one metric with zero judge involvement) says
quality is flat-to-improved, while every judge-routed metric collapsed.** If generation had
truly regressed the way judge-accuracy suggests, BERTScore — a pure embedding comparison
against the correct reference answer, indifferent to model size or output formatting —
should show it too. It didn't move. The most defensible reading: **this is a self-judging
confound, not a real quality regression.** `generate_answer` and the judge
(`LocalDeepEvalModel`, the LLM-judge loop) all route through the *same* `model` — the swap
changed the judge at the same time as the generator, so these metrics cannot distinguish
"the answers got worse" from "the new judge scores differently/more strictly." A materially
more capable general model (Qwen2.5-7B vs. Falcon-H1-1.5B) judging its own output plausibly
applies harsher, more medically-literate scrutiny than the weak Falcon-H1 judge ever could —
correctly catching real issues the old judge missed, not because generation degraded.

**Do not report the judge-based accuracy collapse as "Qwen generates worse answers."**
Report it as a disclosed measurement limitation: self-judged accuracy is not comparable
across generator-model swaps in this design, because the model changes on both sides of the
comparison simultaneously. Resolving this would need either a third, fixed, independent
judge model, or a human rating pass on a sample — neither has been done.

**Two bugs were found and fixed in the runtime groundedness gate while running these
experiments** (both now covered by permanent regression asserts in Component 5's self-check):
- **Bullet-list bypass** — answers formatted as short bullet points (common in Arabic output)
  fragmented below `min_sentence_chars`, filtered to an empty list, and the old code returned
  a vacuous `min_similarity: 1.0` **pass** without running the semantic check at all. Fixed by
  falling back to scoring the whole answer as one unit.
- **Refusal-substring bypass** — observed live: a real dosage answer (containing real,
  unverified numbers) quoted the refusal phrase mid-sentence as an aside. The old refusal
  check was a bare substring test, so it matched and short-circuited to `grounded: True`,
  skipping both the numeric and semantic checks for an answer with real numbers in it — the
  fail-closed guarantee broke exactly where it mattered. Fixed by requiring the refusal phrase
  to account for (nearly) the whole answer, not merely appear somewhere inside it.

## Key Constraints When Modifying

- Any change to embedding model, normalization, or vector dimension must stay consistent
  with the existing Qdrant collection (`meta.json`: 1024-dim, Cosine) or the archive must be
  rebuilt/re-indexed.
- Prompt templates and the refusal string are matched pairs per language (AR/EN); keep them
  in sync if editing one.
- `CONFIG_C4` (`top_k_retrieve`, `top_k_rerank`, `score_threshold`, `max_rewrites`, `rrf_k`)
  is the single place retrieval/rerank/rewrite/fusion behavior is tuned.
- `score_threshold` is a **cosine** threshold. If `retrieval_score` is ever changed to carry
  the RRF or BM25 score instead of the dense cosine, this threshold becomes meaningless and
  must be recalibrated — see the `retrieve_chunks` notes above.
