# GroundedRx — Bilingual Arabic/English Medical RAG

_Was "FalconMed AI" — renamed after the Falcon-H1 → Qwen2.5 model swap left the old name
referring to an architecture no longer in use._

Complete project report. Every number in this document came from an actual run — nothing
here is estimated, projected, or copied from a paper. Where a measurement is unreliable or
missing, it says so explicitly rather than omitting it.

**Last updated:** 2026-08-10 (renamed to GroundedRx; production/portfolio direction — see §8
item 4)

---

## 1. What this project is

A retrieval-augmented generation (RAG) system that answers patient questions about
medications in **both Arabic and English**, grounded in the PEACH patient-information-leaflet
dataset.

The distinguishing property is **true cross-lingual retrieval**: an Arabic question can be
answered from English source documents and vice versa, without translating the query first
and searching second. This is measured, not assumed — see §5.1.

| | |
|---|---|
| **Domain** | Medical / patient information leaflets |
| **Languages** | Arabic, English |
| **Target** | Portfolio / resume project (originally scoped toward an ArabicNLP Workshop submission; superseded — see §8 item 4) |
| **Runtime** | Google Colab / Kaggle free tier (T4 GPU, ~14.5 GB VRAM) |
| **Form factor** | Single notebook today — `GroundedRx_Colab.ipynb`; packaging in progress |

---

## 2. Technology stack

| Layer | Choice | Why |
|---|---|---|
| Orchestration | LangGraph `StateGraph` | Needed a conditional feedback loop, not a linear chain |
| Vector store | Qdrant (local file mode) | No server needed; runs from a zip in Colab |
| Embeddings | `BAAI/bge-m3` (1024-dim) | Multilingual, aligns AR/EN in one vector space |
| Sparse retrieval | `rank_bm25` (BM25Okapi) | Exact drug names/doses that dense embeddings blur |
| Reranking | `BAAI/bge-reranker-v2-m3` | Multilingual cross-encoder, more accurate than cosine |
| Generation | `Qwen/Qwen2.5-7B-Instruct` (was `tiiuae/Falcon-H1-1.5B-Deep-Instruct`) | Standard attention transformer, avoids Falcon-H1's Mamba2-specific issues; swapped after decoding/quantization experiments failed to fix Arabic quality (§7) |
| Quantization | bitsandbytes NF4 4-bit | Required to fit a T4 |
| Evaluation | BERTScore + DeepEval + LLM-as-judge | See §6 for which of these actually worked |

---

## 3. The corpus

Inspected directly from the Qdrant store, not assumed:

| Property | Value |
|---|---|
| Total chunks | **2,365** |
| English chunks | 1,197 |
| Arabic chunks | 1,168 |
| Unique documents | **464** |
| Vector dimensions | 1,024 |
| Distance metric | Cosine |
| Median chunk length | 817 characters |
| Payload fields | `chunk_text`, `language`, `category`, `document_id`, `chunk_id`, `file_name` |

**Critical structural fact:** every `document_id` is single-language. There are **no parallel
Arabic/English document pairs**. This shapes the entire cross-lingual evaluation — "cross-lingual
retrieval" here can only mean *a query in one language surfacing chunks written in the other*,
never *matching a document to its translation*, because no translations exist in the corpus.

---

## 4. Architecture

```
detect_language → rewrite_query → embed_query → retrieve_chunks → check_retrieval_quality
                       ^                                                    |
                       └──────────── (score < 0.5, retries < 2) ────────────┤
                                                                            v
                                                       rerank_chunks → build_context
                                                                            v
                                                          generate_answer → GROUNDEDNESS GATE
                                                                            v
                                                                    answer to user
```

### Pipeline stages

| Stage | What it does |
|---|---|
| `detect_language` | `langdetect` → `"ar"` / `"en"`; picks the prompt template |
| `rewrite_query` | **Pass 0: returns query unchanged.** Only retries append medical keywords |
| `embed_query` | bge-m3, `normalize_embeddings=True` (must match how the index was built) |
| `retrieve_chunks` | **Hybrid**: dense top-20 + BM25 top-20, fused by Reciprocal Rank Fusion |
| `check_retrieval_quality` | If best dense cosine < 0.5 and retries remain → loop back |
| `rerank_chunks` | Cross-encoder, 20 → 5 |
| `build_context` | Formats top-5 into a numbered, language-aware context block |
| `generate_answer` | Qwen2.5-7B (was Falcon-H1), context-only prompt with an explicit refusal string |
| `check_grounding` | **Runtime safety gate** — blocks ungrounded answers (§5.3) |

### Key design decisions

**Fusion is on rank, not score.** Cosine is bounded 0–1; BM25 is unbounded and its scale
shifts per query. Blending raw scores would need per-query normalization. RRF (`k=60`)
sidesteps this. The join key is the Qdrant point id — verified unique across all 2,365 chunks.

**`retrieval_score` stays the dense cosine, deliberately.** The quality gate is calibrated
against cosine (threshold 0.5). Feeding it an RRF score (max ≈0.016) would make it fire on
every query; a raw BM25 score would make it fire on none. Fusion changes *what is retrieved*,
not *how retrieval is judged*.

**The gate judges the truncated context.** Context is capped at 2,000 chars before generation.
The gate checks against that same truncated text — judging against the full context would
credit the model for content it was never shown.

---

## 5. Measured results

### 5.1 Cross-lingual retrieval — VERIFIED ✅

Tested with 4 semantically parallel AR/EN query pairs, three independent tests:

| Measurement | Result | Meaning |
|---|---|---|
| Other-language share of top-20 (raw query) | **37.5%** | Cross-lingual retrieval genuinely happens |
| Other-language share (after retry expansion) | 25.0% | The keyword expansion *suppresses* it |
| Mean cosine gap (same-lang − cross-lang) | **+0.016** | bge-m3's alignment is essentially symmetric |
| Mean cross-language rerank score | +0.669 | The reranker independently agrees |

Per-language breakdown of the raw-query share:

| Query language | Cross-lingual share |
|---|---|
| Arabic → English | 40–70% |
| English → Arabic | 10–20% |

**Arabic queries retrieve English content far more readily than the reverse.** Consistent with
bge-m3's embedding space being English-centric.

**Bug found and fixed here.** `rewrite_query` originally appended language-specific medical
keywords on *every* pass, which cut Arabic cross-lingual reach roughly in half (60% → 25%) by
pulling the embedding back toward the query's own language. English was unaffected — so the
damage was asymmetric and invisible to English-only testing. Raw queries already clear the 0.5
gate (cosine 0.63–0.73), so the expansion bought nothing measurable. It is now a retry-only lever.

**Second bug found:** the expansion was built from `state["query"]` (always the raw query), so
both loop passes produced a **byte-identical string**. The LangGraph feedback loop re-ran
identical retrieval and could never change its own outcome. It was a no-op costing an extra
embed+retrieve round. Now pass 0 is raw and pass 1 is expanded — two genuinely different attempts.

**Caveat, stated honestly:** the mean cross-language rerank score of +0.669 hides real variance.
Two of eight probes were effectively *rejected* by the reranker (+0.214 and +0.005), both Arabic
queries. Cross-lingual retrieval works, but its precision is uneven, and the reranker is
currently the only thing filtering the bad matches.

### 5.2 Hybrid search — VERIFIED ✅

BM25 characteristics measured directly on this corpus:

| Query | BM25 behaviour | Verdict |
|---|---|---|
| `lisinopril dose 10 mg` | Top hits doc 499 (20.44) and doc 502 (20.03) — both lisinopril leaflets | **Strong** |
| `side effects` | ~7.9, scattered across unrelated documents | **Useless** |
| `Logynon` | **0.00** — token absent from index | **Blind spot** |

The `Logynon` result is a genuine limitation: Arabic leaflets spell drug names in Arabic script,
so BM25 can never lexically match a Latin-script drug name against them. Cross-language matching
stays the dense half's job. Because RRF is additive, this costs nothing that wasn't already true.

End-to-end result on the exact-term smoke test (`"What is the lisinopril 10 mg dose?"`):

| Metric | Value |
|---|---|
| Chunks contributed by BM25 alone | **7** (dense retrieval never surfaced them) |
| Chunks found by both halves | 7 |
| Documents in final top-5 | **499 and 502** — both lisinopril products |
| Dense cosine on this query | 0.6462 — the **lowest** of all five smoke tests |

That last row is the point: dense retrieval was *least* confident exactly where sparse was
strongest. Textbook complementarity, which is why fusion beats either alone.

Generic queries (tests 1–4) show identical results with and without fusion. That is the
**expected** outcome, not a regression — generic phrasing is precisely where BM25 adds nothing.

### 5.3 Runtime groundedness gate — WORKS, WITH LIMITS ⚠️

Two deterministic checks. Deliberately **not** an LLM self-judge, because DeepEval's
FaithfulnessMetric already failed on 12/12 questions with this same model, a second Falcon
call per query is expensive, and any LLM judge reintroduces a parse-failure mode.

| Check | Mechanism | Cost |
|---|---|---|
| **Numeric grounding** | Every number in the answer must appear in the context. Arabic-Indic and Persian digits normalized (`١٠` ≡ `10`) | Regex — free |
| **Semantic grounding** | Answer split into sentences, max cosine vs context sentences via bge-m3 | Reuses a loaded model |

**Result on 12 evaluation questions: 1 blocked.**

```
BLOCKED [en] sim=0.637  nums=['4']  |  What are the contraindications of Linopril?
```

The answer asserted a `4` appearing nowhere in the context — a fabricated figure in a
*contraindications* answer, the highest-consequence place to invent one.

**The important detail:** `sim=0.637` was **above** the 0.50 threshold, meaning the semantic
half *passed this answer*. The numeric check is the only reason it was caught. Across all 17
real answers generated so far (5 smoke + 12 eval), **the semantic half has never once fired.**
Numeric grounding is the load-bearing check; the cosine half remains unproven.

**Safety properties:**
- **Fails closed** — empty context, empty answer, or any exception → `grounded: False`. In
  offline evaluation a broken metric costs a data point; at runtime it would ship an unverified
  medical answer.
- **Correct refusals pass** — without this, the model's own honesty would be flagged as
  hallucination.
- **Blocked answers preserved** as `answer_raw` for inspection.

**Bug found and fixed:** bulleted answers — which this model produces constantly in Arabic —
split into fragments below the 25-char minimum, were filtered to an empty list, and the old code
then returned `min_similarity: 1.0` and **passed without running the semantic check at all**.
A list-shaped hallucination bypassed the check entirely while reporting a perfect score. This is
why the first calibration run showed a median `min_sim` of exactly 1.000. Now falls back to
scoring the whole answer as one unit.

**⚠️ KNOWN CEILING — this gate cannot catch negation.** Cosine measures topical overlap, not
logical entailment. *"Take this with alcohol"* against context *"Do NOT take this with alcohol"*
scores very high and passes. **It is a fabrication filter, not a faithfulness guarantee.**
Closing this requires a real NLI/entailment model (e.g. mDeBERTa XNLI) as a second GPU resident.

**⚠️ KNOWN CEILING — this gate cannot catch wrong-drug substitution, confirmed live via the
Component 7 demo.** Asking "What is the recommended dose of ibuprofen for a 5-year-old?"
(ibuprofen isn't in this corpus) retrieved an unrelated leaflet for a drug called "Batlor"
(`11226.xlsx`) — its pediatric-dosing section is phrased similarly enough ("Children 1 through
5 years... 2.5 ml...") to rank top-1 for the query. The model answered with Batlor's real dose,
correctly *named* as Batlor rather than fabricated as "the ibuprofen dose" — but never flagged
that this isn't the medication asked about. The gate scored it `grounded: True` (0.75
similarity): every claim genuinely is supported by the retrieved text, so numeric + semantic
grounding both correctly pass it. **The gate only checks claim-vs-context support — it has no
concept of query-vs-document identity**, so this failure mode is invisible to it by design, not
a bug in the gate itself. Root cause is one layer upstream, in retrieval: dense/BM25 fusion
matches on phrasing/structure ("dose for a child of this age/weight"), not on drug identity, so
a pediatric-dosing chunk for the wrong drug can outrank "nothing relevant found." **This is the
more consequential sibling of the negation ceiling above** — a confident, real-numbers answer
about the wrong medication is a worse failure mode than a refusal, since a user skimming the
answer could easily miss that the drug name changed. Closing it needs a retrieval-time
drug/entity check (verify the queried drug name, or a known synonym/brand, actually appears in
the candidate chunk's `file_name`/`document_id` metadata before it's allowed to answer) — not a
generation-time or gate-time fix.

**Threshold calibration.** `min_sentence_similarity = 0.50` is a starting guess, not a derived
constant. Observed scores on grounded answers: 0.601–0.883, and the ordering tracked subjective
quality (most precise answer 0.883, weakest 0.601) — evidence the metric measures something real.

### 5.4 Retrieval quality audit

Judge-independent check using BERTScore between retrieved chunks and ground truth:

| Metric | Value | Interpretation |
|---|---|---|
| Mean best-chunk retrieval F1 | **0.774** | Well above the 0.6 "good" line |

This matters because it is **independent of the LLM judge**. It confirms retrieval surfaces the
right chunks, which isolates any downstream problem to generation or to the judge itself.

### 5.5 Generation quality

**BERTScore** (independent of any LLM judge):

| Language | F1 |
|---|---|
| English | **0.8832** |
| Arabic | **0.6494** |
| Overall | 0.7663 |

**LLM-as-judge** (1–5 scale, Falcon self-judging):

| Metric | English | Arabic |
|---|---|---|
| Accuracy | 4.57 | **3.00** |
| Safety | 4.71 | **3.40** |
| Coherence | 4.57 | **2.80** |

---

## 6. Evaluation reliability — read this before citing any number

DeepEval metrics are wrapped in per-question try/except. Because pandas `.mean()` silently
skips NaN, **a mean over 3 questions prints identically to a mean over 12**. Measured success
rates **under Falcon-H1** (the model these were first measured against):

| Metric | Succeeded | Score | Citable? |
|---|---|---|---|
| Answer Relevancy | **12/12** | EN 0.857 / AR 0.560 | ✅ Yes |
| Context Precision | **11/12** | 0.594 | ✅ Yes, *under Falcon-H1 only* — see below |
| Context Recall | **3/12** | 0.867 | ❌ **No** |
| Faithfulness | **0/12** | `nan` | ❌ **No** |

Failures are `'verdicts'` KeyErrors and *"Evaluation LLM outputted an invalid JSON"*. A 1.5B
model cannot reliably emit DeepEval's verdict schema. **This is not a token-budget problem** —
already ruled out by raising `max_new_tokens` to 384 and then 640.

> **Do not put Faithfulness or Context Recall in the paper.** These are not weak numbers —
> they are absent ones. Reporting Context Recall 0.867 would mean citing a 3-sample mean
> against earlier 12-sample figures.

**Update after the Qwen2.5-7B swap: Context Precision failed 0/12 too — same error, every
question.** It worked under Falcon-H1 (11/12, later 10/12 at a corrected token budget) but
failed completely under Qwen2.5-7B, with the identical *"Evaluation LLM outputted an invalid
JSON"* message every time — even though `answer_relevancy` (a simpler single-verdict JSON
task) succeeded 12/12 in the same run. That makes **three** DeepEval metrics requiring a
multi-item verdict-list JSON that have now failed across both models tested
(`faithfulness`, `context_recall`, `context_precision`) — a consistent pattern tied to that
task shape, not per-model noise. `ContextualPrecisionMetric` has been **removed from the
notebook entirely**; `answer_relevancy` is the only DeepEval metric still run, and the only
one that's proven reliable with more than one model. The summary block prints
`n=<succeeded>/<total>` beside it and stamps `DO NOT report this figure` if success drops
under 75%.

---

## 7. Current status

### Works ✅

- **Retrieval** — 0.774 best-chunk F1, verified independently of any LLM judge
- **Cross-lingual retrieval** — 37.5% cross-boundary, 0.016 cosine gap, reranker concurs
- **Hybrid search** — contributes 7 chunks dense retrieval misses on exact-term queries
- **English generation** — BERTScore 0.883 under Falcon-H1, 0.899 under Qwen2.5-7B (both
  judge-independent, both solid); judge-based accuracy numbers below are Falcon-H1-era and
  **not directly comparable to Qwen** — see the model-swap section above for why
- **Safety gate** — caught a real fabricated number in a contraindications answer
- **Pipeline stability** — runs end to end on Colab/Kaggle free without CUDA errors

### Broken ❌ (as measured under Falcon-H1 — see model-swap section above for current status)

**1. Arabic generation — the primary open defect under Falcon-H1.**

Four independent metrics agreed at the time, so this was not judge noise:

| Signal | English | Arabic |
|---|---|---|
| BERTScore (no LLM judge involved) | 0.883 | 0.649 |
| Judge accuracy | 4.57 | 3.00 |
| Judge coherence | 4.57 | 2.80 |
| Answer relevancy | 0.857 | 0.560 |

Directly observable corruption: the Arabic answer contained `ليس限اً` — a **Chinese character
embedded mid-Arabic**. No measurement is needed to see that is wrong.

This is a **generation** problem, not a retrieval one: retrieval scores 0.774 and hands the
model good Arabic chunks; the reranker confirms their relevance.

**Hypothesis tested and REJECTED.** Switched `generate_answer` to greedy decoding
(`do_sample=False`, matching what Component 6's judge/DeepEval calls already used) on the
theory that sampling under 4-bit NF4 quantization was drawing noisy low-probability Arabic
tokens. Two independent pieces of evidence say no:

1. **The exact corrupted token reproduced deterministically, twice.** `ليس限اً` (a Chinese
   character meaning "limit," embedded where the model was reaching for "not limited to")
   appeared in the same phrase, in the same position, across two separate greedy runs. Greedy
   always takes the argmax — this is the model's own top-ranked choice at that position, not
   an unlucky sampling draw.
2. **Component 6 numbers moved the wrong way.** AR BERTScore went **down** (0.649 → 0.630),
   AR coherence was **unchanged** (2.80 → 2.80), and the groundedness gate caught a **second**
   fabrication that the sampled run didn't have (an invented "4" in the Logynon missed-dose
   answer). English metrics were bit-for-bit identical, as expected. Retrieval-only metrics
   (Context Precision/Recall) were exactly unchanged too — a clean confirmation that only the
   decoding path changed between runs.

**Conclusion: decoding strategy is not the primary limitation.** Per the investigation's
decision tree, the next step is an 8-bit precision experiment (same model, greedy held
constant) to separate "quantization-related" from "model-capacity-related" before considering
a different model.

**Experiment 2 — 8-bit quantization: PRELIMINARY, NOT CONCLUSIVE.** The quantitative
comparison (full 12-question Component 6 run under 8-bit + greedy) did not complete — Colab
free's GPU usage quota was hit mid-experiment. What exists so far is two Arabic smoke-test
answers from Component 5, not enough to conclude anything against the decision tree's bar.
Recorded here so the observations aren't lost, not as a result:

- **Encouraging, single data point:** for the first time across four runs (sampled/4-bit,
  greedy/4-bit ×2, greedy/8-bit), the `ليس限اً` corruption did **not** appear in the Arabic
  side-effects answer. Same model, same greedy decoding, only precision changed between this
  run and the prior greedy/4-bit run — a real controlled comparison, but of one example.
- **New failure mode observed, not previously seen:** that same answer was still **blocked**
  by the gate, for a different reason — the model appended a trailing "I don't have enough
  information..." hedge sentence after an otherwise substantive, on-topic answer. That one
  low-similarity sentence (topical meta-commentary has no cosine match to medical context)
  dragged the whole answer's `min_similarity` below threshold. Defensible as conservative
  safety behavior, but worth naming: the gate currently discards a whole answer for one
  uncertain trailing sentence rather than isolating it.
- **A separate, unrelated concern surfaced in an English smoke test:** "What is the
  recommended dosage for adults?" (unfiltered retrieval, as Component 5's smoke tests always
  run) produced an answer that appears to conflate sertraline dosing with unrelated
  furosemide/Lasix titration figures (40mg → 80mg → 160mg), without attributing which figure
  belongs to which drug. The gate passed it — the numbers were likely genuinely present
  somewhere in the retrieved context, so numeric grounding correctly found them "in context."
  This is a **different blind spot than negation**: the gate checks whether content exists in
  context, not whether the answer correctly attributes it to the right document. **Since
  confirmed, cleanly, via the Component 7 Gradio demo** — see the wrong-drug-substitution
  ceiling below.

**Full 12-question Component 6 run completed — result is CONFOUNDED, not a clean answer.**

| Metric | Baseline (sampled+4bit) | Greedy+4bit | Greedy+8bit (this run) |
|---|---|---|---|
| EN BERTScore | 0.883 | 0.883 | 0.885 (flat, as expected) |
| AR BERTScore | 0.649 | 0.630 | 0.640 (between the two, still below baseline) |
| EN judge accuracy | 4.57 | 4.57 | **3.71 (dropped)** |
| EN judge safety | 4.71 | 4.71 | **3.86 (dropped)** |
| EN judge coherence | 4.57 | 4.57 | **3.43 (dropped)** |
| AR judge accuracy | 3.00 | 3.00 | **1.80 (dropped further)** |
| AR judge coherence | 2.80 | 2.80 | **3.40 (first real movement)** |
| `context_precision` success rate | 11/12 | — | **7/12 (dropped)** |

**The English regression is the actual finding here, and it undercuts the whole premise of
this experiment.** English was bit-for-bit stable through the entire greedy-decoding
experiment — a targeted fix for Arabic-specific corruption should not move English judge
scores by a full point. That means 8-bit is not the surgical, isolated change the
intermediate-experiment design assumed; it's changing generation or judging behavior more
broadly, in a direction that isn't obviously good.

**A confound was self-inflicted and has been fixed, not yet re-tested.** DeepEval's judge
`max_new_tokens` had been cut 640→384 in the same patch that removed `faithfulness`
(justified at the time: 640 only existed for faithfulness's 2-step chain). But 4 of 5 AR
`context_precision` failures in this run were exactly the long-context questions (22k-28k
chars, nearly 3x the EN questions) — a longer context needs a longer per-chunk verdict list,
so 384 tokens plausibly starved those specifically. Reverted to 512 (a middle ground, still
short of 640) — **not yet re-run**, so this run's AR numbers should be read with that caveat.

**AR signals contradict each other on n=5**, too small to resolve on its own: coherence rose
for the first time all session (2.80→3.40), while accuracy fell further (3.00→1.80). Could be
genuine (more fluent, more confidently wrong) or judge noise on a tiny sample — not resolvable
from this data alone.

**Re-run at the corrected token budget (512) — the confound is isolated, and the picture is
now clear enough to conclude on.**

| Metric | Baseline | Greedy+4bit | Greedy+8bit (384 tok) | Greedy+8bit (512 tok, clean) |
|---|---|---|---|---|
| `context_precision` success | 11/12 | — | 7/12 | **10/12** (fix confirmed) |
| EN BERTScore | 0.883 | 0.883 | 0.885 | 0.885 (unchanged) |
| EN judge accuracy | 4.57 | 4.57 | 3.71 | **3.71 (unchanged — confirmed real)** |
| EN judge coherence | 4.57 | 4.57 | 3.43 | **3.43 (unchanged — confirmed real)** |
| AR BERTScore | 0.649 | 0.630 | 0.640 | 0.640 (unchanged) |
| AR judge accuracy | 3.00 | 3.00 | 1.80 | **1.80 (unchanged — confirmed real)** |
| AR judge coherence | 2.80 | 2.80 | 3.40 | **3.40 (unchanged — confirmed real)** |

The token-budget fix only ever touched `context_precision` (the one metric that depends on
DeepEval's judge call budget). Everything else — BERTScore, both LLM-judge loops — was never
affected by that bug and reproduced identically between the confounded and clean runs. That
confirms the English regression is **real**, not an artifact of my earlier token-budget cut.

**But "real" doesn't mean "generation got worse" — a genuine ambiguity, not yet resolved.**
BERTScore (an independent encoder, no connection to Falcon) says English answers are
semantically unchanged from baseline (0.885 vs 0.883). Falcon's own self-judge — now running
in 8-bit — says English quality dropped a full point across accuracy, safety, and coherence.
Those two signals disagree. The more defensible reading is that **8-bit may be degrading this
model's reliability at self-judging** (a harder, more precision-sensitive structured-output
task) rather than degrading the free-form generation patients would actually receive — but
this data cannot distinguish the two explanations, and no further run was designed to.

**Verdict against the decision tree's own bar: 8-bit does NOT measurably improve Arabic
generation overall.** Of three quantitative AR signals, two moved the wrong direction
(BERTScore 0.649→0.640, accuracy 3.00→1.80) and only one improved (coherence 2.80→3.40). The
one clear win — no CJK corruption in 3/3 qualitative smoke tests — is real but narrow: it
fixes one specific symptom, not overall Arabic quality. This does not clear the "measurably
improves" threshold the plan set for stopping here and calling it quantization-related.

**Do not cite "8-bit fixed Arabic generation."** The corruption-specific claim has decent
support; the broader quality claim does not, and English regressing where it was previously
untouchable is reason for real caution about this quantization path generally.

**Decision: quantization path stopped here, deliberately, not by oversight.** Per the
decision tree, the next lever after an unclear 8-bit result would be bf16/fp16. That step was
explicitly skipped: two consecutive experiments (greedy decoding, then 8-bit quantization)
each failed to produce a clean signal, and each surfaced a *new* ambiguity rather than
resolving the original one (decoding: reproduced the same corruption deterministically but
told us nothing about magnitude; 8-bit: fixed the corruption but worsened two of three other
Arabic signals and unexpectedly regressed English). More precision tuning producing more
confusion instead of convergence is itself a signal, not a reason to run a third variant.

**Conclusion: treat model capacity as the dominant limitation.** The next real lever is
evaluating a stronger multilingual/Arabic-capable model in place of Falcon-H1-1.5B — not
a decoding or quantization setting. This has not been started.

**2. Two of four DeepEval metrics — a broken measuring tool, not a broken system.** See §6.

### Not built (and why)

| Gap | Status |
|---|---|
| Chunking strategy review | **Not buildable** — the source documents no longer exist; only pre-chunked vectors remain |
| Multi-turn conversation memory | Belongs to a production service, not a batch-eval notebook |
| Caching / streaming | Same — no meaning in a notebook context |

---

## 8. Recommended next steps, in order

1. ~~Switch Arabic generation to greedy decoding~~ — **done, rejected** (§7).
2. ~~Test 8-bit quantization~~ — **done, rejected** (§7). bf16/fp16 deliberately skipped —
   see §7 for why running a third precision variant wasn't the right next move.
3. ~~Evaluate a stronger multilingual/Arabic-capable model~~ — **done: Falcon-H1-1.5B →
   Qwen2.5-7B-Instruct.** Result is genuinely ambiguous, not a clean win (see "Experiment 3"
   above) — BERTScore flat-to-improved, judge-based metrics collapsed, most likely a
   self-judging confound rather than a real regression. Treat as closed for now; resolving
   the ambiguity further (a third, independent judge, or a human rating pass) is optional
   future work, not a blocker.
4. **AMENDED — goal changed, this note no longer applies as written.** This originally said
   "do not start production work yet," reasoned against wrapping a system whose Arabic
   accuracy score was 1.80/5. That reasoning was sound for a *research-paper* goal (a known
   defect undermines a paper's claims); it does not apply to the current goal, which is a
   portfolio-ready, resume-worthy system, explicitly not a research paper (see Project
   Overview note in `CLAUDE.md`). Production work — git repo, packaging, tests, CI, API,
   on-premises Docker deployment — is now in progress. The known limitations (Arabic
   judge-metric ambiguity above, the groundedness gate's documented negation and
   wrong-drug-substitution ceilings) are being carried forward as **disclosed, evidenced
   limitations** in the README rather than as blockers — an honest limitations section is
   itself part of a production-grade deliverable, not a reason to withhold one.
5. ~~If a future paper needs Faithfulness, it needs a different judge~~ — **not currently
   applicable**; no paper is planned. Kept as a true technical finding (§6) in case research
   framing is revisited later.

---

## 9. Engineering history worth preserving

**SGLang was tried and reverted.** Component 6 kept hitting CUDA OOM; the root cause was
`transformers` falling back to a naive, non-optimized SSM scan for Falcon-H1's Mamba2 layers
(no `causal-conv1d`/`mamba-ssm` kernels), which scales memory ~O(seq_len²). SGLang ships
optimized Falcon-H1 kernels and looked like the fix. It was not: `pip install "sglang[all]"`
pulled a torch/CUDA-13/numpy≥2.5 combination that conflicts with Colab free's preinstalled
RAPIDS/cuDF/cuML/numba/dask-cuda stack (all pinned to CUDA 12.x). Three attempted fixes each
broke a *different* part of the environment. Three fixes, three different failure sites — that
is an architectural incompatibility, not a version pin to chase. Reverted to
`transformers` + `bitsandbytes`.

**RAGAS was replaced with DeepEval.** `ragas==0.1.21` failed on version conflicts; `ragas>=0.4`
has a packaging bug (unconditionally imports `ChatVertexAI` from a removed `langchain_community`
path); pinning `ragas<0.4` did not resolve cleanly in Colab. DeepEval has no such import chain.

**fp16 under 4-bit quantization overflowed to NaN/Inf** mid-generation on this hybrid
architecture — first as garbled mixed-language tokens, then a hard `AcceleratorError` inside
`torch.multinomial`. Once that fires the CUDA context is poisoned for the whole runtime; restart
the session, do not just re-run the cell. Fixed by auto-selecting bf16 via
`torch.cuda.is_bf16_supported()`.

**Falcon-H1-Instruct requires `apply_chat_template`.** It has no `chat_template` in
`tokenizer_config.json` but ships one via `chat_template.jinja`. Using a raw `tokenizer(prompt)`
call makes the model hallucinate literal `assistant` role-turn text and repeat its own answer.

**Evaluation questions were re-grounded in real documents.** Originally generic ("What are the
side effects of this medication?"), which retrieved whichever of 464 drugs was semantically
closest — inconsistent grounding question-to-question and language-to-language, against
hand-typed ground truths that didn't match anything retrievable. Now English questions target
document 502 (Linopril/lisinopril) and Arabic target 498 (Logynon), with ground truths copied
verbatim from those documents, pinned via a `document_id` filter that **only the evaluation loop
uses** — live user queries still search the whole corpus.

---

## 10. How to run

1. Open `GroundedRx_Colab.ipynb` in Colab. Runtime → Change runtime type → **GPU**.
2. Upload `qdrant_db_archive.zip` to `/content/`.
3. Run cells top to bottom:

| Cell | Contents | Notes |
|---|---|---|
| Setup | Installs deps, unzips Qdrant, loads all models | Slowest cell |
| Component 4 | LangGraph pipeline + BM25 index + 5 smoke tests | Test 5 proves hybrid search fires |
| Component 4b | Cross-lingual verification | Retrieval only — seconds, safe to re-run |
| Component 5 | Generation + groundedness gate + 5 tests | Self-check asserts before any Falcon call |
| Component 6 | Full evaluation over 12 questions | Longest; saves CSV + JSON |

Call `rag_answer("<question>")` for a one-off end-to-end query.

**If Component 6 OOMs on a genuinely clean session,** uncomment the
`causal-conv1d` / `mamba-ssm` install in Setup — it fixes the O(seq_len²) SSM scan at the root.
A failed install there is self-contained (a clear pip error, nothing else breaks), unlike the
SGLang cascade.

**Colab note:** "Restart session" does **not** reset installed packages. For a genuinely clean
environment use **"Disconnect and delete runtime"**.

---

## 11. Honest summary

The **architecture** is sound and every claim about it is backed by a measurement rather than an
assumption. Retrieval, cross-lingual matching, hybrid fusion, and the safety gate all work and
all have numbers behind them.

The **English system** worked well under Falcon-H1 — 0.883 BERTScore, 4.57/5 judged accuracy —
but that was measured before the model swap below; not yet re-confirmed under Qwen2.5.

The **Arabic system did not work well under Falcon-H1** — 0.649 BERTScore, 2.80/5 coherence,
with visible token corruption (`ليس限اً`, a Chinese character embedded mid-Arabic). Two
targeted fixes were tried and rejected: greedy decoding (the corruption reproduced
deterministically — a model choice, not sampling noise) and 8-bit quantization (fixed the
corruption specifically, but worsened 2 of 3 broader Arabic signals and unexpectedly
regressed English). Both experiments converged on the same conclusion: **model capacity, not
a decoding or quantization setting, was the limitation.**

**Current status: the model has been swapped to `Qwen/Qwen2.5-7B-Instruct` and evaluated —
result is genuinely ambiguous, not a clean confirmation or rejection.**

| Signal | Falcon-H1 baseline | Qwen2.5-7B |
|---|---|---|
| EN BERTScore (judge-independent) | 0.883 | **0.899 (up)** |
| AR BERTScore (judge-independent) | 0.649 | 0.637 (flat) |
| EN judge accuracy | 4.57 | **3.00 (collapsed)** |
| AR judge accuracy | 3.00 | **1.20 (collapsed further)** |
| AR judge coherence | 2.80 | **4.80 (all-time high)** |
| DeepEval Answer Relevancy | 0.733 | **0.394 (collapsed)** |
| DeepEval Context Precision | 0.594 (11/12) | **0/12 — total failure** |

Qualitative smoke tests were the cleanest of the whole investigation — zero CJK corruption
across multiple Arabic answers for the first time all session, every gate check passed. That
qualitative signal still stands.

**The quantitative judge-based numbers most likely don't mean what they look like they
mean.** BERTScore — the one metric with zero judge involvement — says quality is
flat-to-improved. Every metric routed through the model judging itself collapsed. Since
`generate_answer` and the judge (DeepEval's wrapper, the LLM-judge loop) all use the *same*
loaded model, the swap changed the judge at the same moment it changed the generator — these
metrics cannot distinguish "the answers got worse" from "the new, more capable judge scores
more strictly." **Do not report the accuracy collapse as "Qwen generates worse answers."**
Report it as a disclosed limitation of this self-judging design. See §7 for full detail,
including why `context_precision` was removed entirely (0/12, same schema-complexity
failure pattern as Falcon-H1's `faithfulness`/`context_recall`, now confirmed across both
models tested).

Two claims to avoid making:

- The groundedness gate is a **fabrication filter, not a faithfulness guarantee**. It cannot
  catch negation, and it cannot catch wrong-drug substitution (confirmed live: an ibuprofen
  query answered, accurately, with a different drug's real dose — see §5.5 in the calibration
  section above).
- **Faithfulness and Context Recall are not measured.** 0/12 and 3/12 (under Falcon-H1) are
  not weak results; they are missing ones — and these two metrics are no longer run at all
  (removed, not just unreportable — see §6).
