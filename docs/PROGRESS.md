# FinFact-BD: Benchmark Progress Log

## Overall Status: Kaggle Rewrite Planning Stage

**Target Venue:** FinNLP Workshop at COLING 2026
**Last Updated:** July 14, 2026
**Timeline:** 12 weeks to submission (Weeks 1-12, targeting January 2027 workshop)

---

## Executive Summary

FinFact-BD started as a dataset generation project for Bengali financial misinformation. We've pivoted toward a controlled planning-guided claim rewriting benchmark. The dataset is a means, not the end. The real contribution is a rigorously validated benchmark with clear research questions, human-annotated ground truth, real-world test cases, and comprehensive model evaluation.

Frozen release: `FinFact-BD-v1.0` regenerated on `2026-07-14`.

**What changed:** We shifted from "generate a dataset and train a classifier" to "build a benchmark that the community can use." This means human validation, difficulty calibration, reasoning labels, real-world examples, and deep error analysis. The generation side moved from rule-based token replacement to planning-guided Bangla claim rewriting using role-specific models: Qwen3-8B for extraction/planning, Aya Expanse 8B for rewriting, multilingual-e5-large for similarity, mDeBERTa-XNLI for contradiction, and BanglaBERT for language quality. The benchmark construction is where we are now.

**Why this matters:** A dataset without validation is just noise. A benchmark with research questions, human labels, and real-world coverage is a lasting contribution.

**Current state:**
- Dataset generation (v1): 20K samples created via rule-based perturbation, 10K quality-filtered
- Dataset generation (v2): Planning-guided Bangla claim rewriting pipeline planned in `docs/KAGGLE_REWRITE_PLAN.md`
- Human validation: Protocol designed, 300 samples selected, annotators needed
- Kaggle rewrite stage: Pipeline design complete, implementation pending
- Timeline: On track for 12-week submission

---

## Research Questions

These RQs drive every phase of the benchmark. Nothing happens without serving at least one RQ.

| RQ | Question | Phase | Status |
|----|----------|-------|--------|
| **RQ1** | Can controlled claim rewriting generate realistic Bengali financial misinformation? | Phase 1-2 | Dataset generated, quality filtering in progress |
| **RQ2** | Do multilingual NLI models effectively filter low-quality perturbations in Bengali? | Phase 2 | mDeBERTa filter built, awaiting Kaggle run |
| **RQ3** | How do LLMs (GPT-4, Claude, Gemini) perform on Bengali financial misinformation detection compared to fine-tuned models? | Phase 5 | Not started |
| **RQ4** | Which perturbation types are hardest for models to detect, and what does error analysis reveal about model limitations? | Phase 7 | Not started |
| **RQ5** | How does performance on synthetic data compare to real-world Bengali financial misinformation? | Phase 6 | Not started |

---

## Dataset Schema

FinFact-BD carries rich metadata per sample. This schema supports every downstream analysis.

```json
{
  "sample_id": "finfact_00001",
  "original_id": "beni_v2_12345",
  "source": "bengali_economy_news",
  "language": "bn",
  "domain": "finance",
  "split": "train|validation|test",
  "perturbation_type": "numerical_fact_change|policy_reversal|entity_replacement|temporal_shift|causal_inversion|original",
  "rewrite_family": "numerical_fact_change",
  "original_text": "Original Bengali financial news article...",
  "perturbed_text": "Modified version with perturbation...",
  "label": "original|perturbed",
  "target_sentence_index": 3,
  "target_span": "১০ শতাংশ বৃদ্ধি পাওয়ার প্রত্যাশা",
  "desired_change": "Flip positive growth to decline",
  "difficulty": "easy|medium|hard",
  "importance_score": 0.85,
  "editability_score": 0.72,
  "diversity_bonus": 0.15,
  "expected_scope": "sentence",
  "expected_changed_claim": "Revenue growth expectation changed from positive to negative",
  "generation_model": "CohereLabs/aya-expanse-8b",
  "regeneration_attempts": 1,
  "verification_result": {
    "claim_integrity": 0.9,
    "surface_quality": 0.8,
    "semantic_quality": 0.85
  },
  "quality_score": 0.87,
  "reasoning_type": "numerical_fraud|policy_reversal|entity_confusion|temporal_distortion|causal_fallacy",
  "human_annotation": {
    "annotator_1": "original|perturbed|not sure",
    "annotator_2": "original|perturbed|not sure",
    "annotator_3": "original|perturbed|not sure",
    "agreement_score": 0.83,
    "gold_label": "perturbed"
  },
  "real_world_match": "dse_insider_trading|bank_fraud|ponzi_scheme|none",
  "metadata": {
    "word_count": 245,
    "entity_count": 3,
    "perturbation_confidence": 0.91
  }
}
```

---

## Phase 1: Dataset Generation ✅ COMPLETED (v1: Rule-Based) → 🔄 REGENERATING (v2: Claim-Guided)

### v1: Rule-Based Perturbation (Completed)

| Metric | Value |
|--------|-------|
| Source articles | 38,107 Bengali economy articles (BENI v2) |
| Original samples | 10,000 unique Bengali financial news articles |
| Perturbations generated | 10,000 (5 perturbation types, balanced) |
| Total samples | 20,000 (10K original + 10K perturbed) |
| Rule-filtered | 9,981 (99.8% pass rate) |
| Perturbation types | 5 (numerical fact change, policy reversal, entity replacement, temporal shift, causal inversion) |

### v2: Planning-Guided Bangla Claim Rewriting (Planned)

The v1 rule-based pipeline produced symbolic edits that were often too small to be human-legible in long Bengali financial articles. The v2 pipeline replaces token-level perturbation with planning-guided claim rewriting using separate reasoning, generation, and verification models. The full plan is documented in `docs/KAGGLE_REWRITE_PLAN.md`.

**Pipeline stages:**

1. **Source loading**: Load articles from BENI v2
2. **Claim extraction**: Identify candidate propositions (numeric, policy, entity, temporal, causal claims)
3. **Claim selection**: Score and select claims using importance_score, editability_score, diversity_bonus
4. **Claim planning**: Build structured rewrite plan per sample
5. **Controlled rewriting**: Use Aya Expanse 8B to realize the planned local change
   - Extraction/planning: `Qwen/Qwen3-8B`
   - Rewriting: `CohereLabs/aya-expanse-8b`
   - Verification: `intfloat/multilingual-e5-large`, `mDeBERTa-XNLI`, `csebuetnlp/banglabert`, deterministic checks
6. **Multi-stage verification**: Three-stage quality gate
   - Stage 1: Claim integrity (did the intended claim change? any extras?)
   - Stage 2: Surface quality (fluent Bangla, journalistic style)
   - Stage 3: Semantic quality (believable news, contradiction score, duplicate detection)
7. **Regeneration loop**: Up to 3 attempts per failed sample
8. **Export**: Raw, filtered, and metadata files

### Perturbation Types

| Type | Samples | Strategy | Why It Matters |
|------|---------|----------|----------------|
| Numerical Fact Change | 2,000 | Rewrite one numeric value in-place | Tests numerical reasoning |
| Policy Reversal | 2,000 | Flip direction words in policy-like or market statements | Tests directional reversals |
| Entity Replacement | 2,000 | Ontology-constrained entity replacement | Tests entity grounding |
| Temporal Shift | 2,000 | Shift year/month/relative time phrase | Tests temporal reasoning |
| Causal Inversion | 2,000 | Reverse causal outcome or connector | Tests causal reasoning |

### Quality Filtering Pipeline

**Core principle:** Validate the filter, don't complicate it.

**New pipeline (v2: Claim-Guided Rewriting):**

```
Original articles
    ↓
Claim / proposition extraction
    ↓
Claim selection (importance + editability + diversity)
    ↓
Claim planning (structured metadata)
    ↓
Bangla generator (sentence/paragraph rewrite)
    ↓
Multi-stage verification (3 stages)
    ↓
Regeneration loop (up to 3 attempts)
    ↓
Human validation pack
    ↓
FinFact-BD release
```

**Old pipeline (v1: Rule-Based) — kept for reference:**

```
Original articles
    ↓
Rule-based perturbation
    ↓
mDeBERTa contradiction score (threshold ≥ 0.4)
    ↓
Human validation (300 samples × 3 annotators)
    ↓
Final filtered dataset
```

**Why mDeBERTa alone is sufficient for filtering:**
- One of the strongest multilingual NLI models (trained on 2M+ multilingual NLI examples)
- Used as a **filter**, not as the benchmark model
- Simplifies pipeline without sacrificing quality

**Note on v2 pipeline:** The new planning-guided rewriting pipeline has built-in multi-stage verification (claim integrity, surface quality, semantic quality) that handles quality control during generation. mDeBERTa remains available as an additional downstream filter for further quality assurance if needed, but it is no longer the sole quality gate.

**Why human validation is essential for the paper:**
- Reviewers will ask: "Why should I trust mDeBERTa on Bengali financial text?"
- Answer: Because we validated it against human judgment
- Report: Fleiss' κ, model-human agreement, examples of accepted/rejected perturbations

**Dual-model decision (mDeBERTa + BanglaBERT):**
- Only justified if it shows >2-3% improvement in human agreement
- Measure: Human agreement with mDeBERTa alone vs. dual model
- If improvement is marginal (<2%), keep single model for simplicity

### Bugs Fixed During Generation (v1 Rule-Based Pipeline)

> These fixes apply to the old rule-based pipeline (`src/generation/perturbation_pipeline.py`), which is kept for reference. The v2 planning-guided pipeline uses a different generation approach.

1. `numerical_perturbation` only matched `%` numbers → fixed for plain Bengali digits (31 → 2,100)
2. `causal_distortion` used silent `.replace()` lambdas → rewritten with regex negation + sentence swap (230 → 2,100)
3. Missing `io` import in pipeline script
4. Leftover dead code in `generate_dataset` function

### Key Files

**v2 Pipeline (Planning-Guided Rewriting):**
- `docs/KAGGLE_REWRITE_PLAN.md` — Full pipeline design document
- `scripts/run_rewrite_pipeline.py` — Active rewrite pipeline entry point
- `Qwen/Qwen3-8B` — Claim extraction and rewrite planning
- `CohereLabs/aya-expanse-8b` — Controlled local Bangla rewrite model
- `intfloat/multilingual-e5-large` — Semantic similarity verifier
- `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` — NLI contradiction verifier
- `csebuetnlp/banglabert` — Bangla ELECTRA discriminator language-quality verifier

**v1 Pipeline (Rule-Based — kept for reference):**
- `src/generation/perturbation_pipeline.py` — Old rule-based generation script (563 lines)
- `src/generation/extract_originals.py` — BENI v2 extraction

---

## Phase 2: Quality Filtering 🔄 IN PROGRESS

| Task | Status | Notes |
|------|--------|-------|
| Rule-based filter | ✅ Done | 9,981/10,000 kept (99.8% pass) |
| XNLI filter attempt | ❌ Failed | Bengali not in training set, only 1.3% contradiction rate |
| mDeBERTa filter script | ✅ Done | `scripts/mdeberta_filter.py` |
| Kaggle notebook | ✅ Done | `scripts/finfact_bd_mdeberta_filter_kaggle.ipynb` |
| Dataset upload to Kaggle | ✅ Done | 13 MB compressed |
| Run mDeBERTa scoring on T4 | ⏳ Pending | ~10-15 min on T4 |
| Download filtered results | ⏳ Waiting | `finfact_bd_perturbed_filtered.csv` |
| Rebalance dataset | ⏳ Waiting | `rebalance_dataset.py` ready |

### Multi-Model Verification Approach (Active Pipeline)

**Models:** multilingual-e5-large for similarity, mDeBERTa-XNLI for contradiction, BanglaBERT for language quality, plus deterministic locality and hallucination checks.
**Method:** Accept a rewrite only if the intended claim changes, unrelated claims remain stable, article similarity remains high, NLI supports the planned contradiction, language quality is acceptable, and no unplanned entities/numbers/dates appear outside the target sentence.
**Why multi-model:** The generator should not verify itself. Independent verification makes failures diagnosable and keeps acceptance separate from generation.

### Historical Context
The original approach used a dual-model ensemble (weight 0.6 mDeBERTa + 0.4 BanglaBERT fine-tuned) to combine multilingual and Bengali-native NLI signals. After evaluation, the BanglaBERT fine-tuning step was unnecessary. The ensemble scripts (`dual_model_judge.py`, `dual_model_filter.py`) remain in the repository for reference.

### Dataset Loading Fix (Jul 14, 2026)
Three failures on Kaggle, fixed in `dual_model_filter.py`:
1. `RuntimeError: Dataset scripts are no longer supported` → tried `trust_remote_code=True`
2. `RemoteProtocolError: peer closed connection` → switched to `hf_hub_download` + parquet
3. `RepositoryNotFoundError: 401 Unauthorized` → **Root cause:** `hf_hub_download` missing `repo_type="dataset"` param

**Solution:** Added `repo_type="dataset"` to all `hf_hub_download` calls. Added automatic fallback to `Divyanshu/indicxnli` (392K Bengali NLI pairs, EMNLP 2022) if primary dataset fails.

---

## Phase 3: Human Validation 🆕 NOT STARTED

Human annotation is the backbone of benchmark quality. We need to validate that our synthetic perturbations are human-recognizable, not just algorithmic noise.

### Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Samples | 300 | Stratified: 60 per perturbation type |
| Annotators | 3 per sample | Majority vote for gold label |
| Agreement threshold | Fleiss' κ ≥ 0.6 | Substantial agreement required |
| Annotation task | Claim-centric: `original` / `perturbed` / `not sure` | Focuses annotation on the edited claim rather than whole-article hunting |
| Training | 10 calibration samples with known labels | Ensures annotators understand the task |
| Time estimate | ~2 hours per annotator | 300 samples, ~24 seconds each |

Annotators receive separate XLSX workbooks with required confidence, justification fields, and adaptive claim windows. The `Samples` sheet shows the headline, a focused claim sentence, nearby context, and the full article text; the optional `Full Articles` sheet is still available only if needed for easier scrolling. A hidden `claim_type` column is reserved for later analysis only. A separate manifest keeps the sample-to-source mapping for merging labels after annotation.

### Annotation Interface

**Task description:** "Given a Bengali financial claim window, determine whether the highlighted claim appears original, perturbed, or not sure. Judge only the highlighted claim, not the whole article. Mark `perturbed` when the claim appears to contain a factual edit (number, entity, time, policy direction, comparison, quotation, or cause-effect relation) and `original` when no clear edit is visible."

**Labels:**
- `original`: The highlighted claim reads like an unmodified article and no clear factual edit is visible from the provided evidence
- `perturbed`: The highlighted claim appears to contain a clear factual alteration or other synthetic edit
- `not sure`: The annotator cannot confidently determine whether the claim is original or perturbed from the provided evidence

### Selection Strategy

We select 300 samples stratified by:
- Perturbation type (60 per type, all 5 types covered)
- Quality score range (low, medium, high from mDeBERTa filter)
- Article length (short, medium, long)
- Entity density (few, many named entities)

This ensures we test the filter across diverse conditions.

### Impact on Other Phases

- **Phase 2:** Filtered dataset may be adjusted based on human validation results
- **Phase 4:** Human labels become the gold standard for benchmark construction
- **Phase 5:** Models evaluated against human-validated ground truth
- **Phase 7:** Error analysis grounded in human judgment

### Key Files
- `src/validation/human_validation_protocol.py` — Selection algorithm + interface
- `docs/ANNOTATION_GUIDELINES.md` — Annotator instructions
- `data/validation_samples/` — 300 selected samples

---

## Phase 4: Benchmark Construction 🆕 NOT STARTED

This phase transforms the validated dataset into a usable benchmark with metadata, difficulty levels, and reasoning labels.

| Task | Status | Notes |
|------|--------|-------|
| Assign difficulty levels | ⏳ Pending | Based on human validation agreement scores |
| Add reasoning type labels | ⏳ Pending | Per perturbation type: numerical fraud, policy reversal, entity confusion, temporal distortion, causal inversion |
| Create train/test/validation splits | ⏳ Pending | 80/10/10 with stratification |
| Write benchmark documentation | ⏳ Pending | Usage guide, known limitations, citation info |
| Build leaderboard infrastructure | ⏳ Pending | GitHub README with score tables |
| Package for Hugging Face | ⏳ Pending | Dataset card, metadata, viewer |

### Difficulty Calibration

| Level | Criteria | Purpose |
|-------|----------|---------|
| Easy | Human agreement > 90%, filter score > 0.8 | Sanity check — models should get these right |
| Medium | Human agreement 70-90%, filter score 0.5-0.8 | Core benchmark difficulty |
| Hard | Human agreement 50-70%, filter score 0.3-0.5 | Challenging for current models |
| Very Hard | Human agreement < 50%, filter score < 0.3 | Edge cases, near-impossible — exposes model limits |

### Reasoning Labels

Each sample gets a reasoning type label based on the perturbation type:
- `numerical_fraud` — Numerical fact change
- `policy_reversal` — Policy reversal
- `entity_confusion` — Entity replacement
- `temporal_distortion` — Temporal shift
- `causal_fallacy` — Causal inversion

### Splits

| Split | Percentage | Purpose |
|-------|------------|---------|
| Train | 80% | Model fine-tuning |
| Validation | 10% | Hyperparameter tuning |
| Test | 10% | Final evaluation (held out) |
| **Real-world test** | 200-500 samples | Out-of-distribution evaluation |

---

## Phase 5: Model Evaluation 🆕 NOT STARTED

Comprehensive baselines across model sizes and paradigms.

### Model Hierarchy

| Category | Model | Params | Why Include |
|----------|-------|--------|-------------|
| **Small fine-tuned** | BengaliBERT | 110M | Bengali-native, fine-tuned baseline |
| **Small fine-tuned** | mDeBERTa-v3 | 280M | Multilingual, used in filtering |
| **Zero-shot multilingual** | XLM-RoBERTa-base | 278M | Cross-lingual baseline |
| **Zero-shot multilingual** | mBERT | 178M | Older multilingual baseline |
| **LLM zero-shot** | GPT-4 | Large | State-of-the-art reasoning |
| **LLM zero-shot** | Claude 3.5 | Large | Alternative LLM |
| **LLM zero-shot** | Gemini 1.5 | Large | Google's model |
| **LLM zero-shot** | Llama-3-8B | 8B | Open-source LLM |
| **LLM fine-tuned** | Llama-3-8B LoRA | 8B | Open-source fine-tuned |

### Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| Accuracy | Overall performance |
| Macro-F1 | Balanced across classes |
| Per-perturbation F1 | Which types are hardest? |
| Per-difficulty F1 | How does difficulty calibration work? |
| Human agreement correlation | Do model errors align with human disagreements? |
| Confusion matrix | Detailed error patterns |
| Latency | Practical deployment consideration |

### Evaluation Protocol

1. **Zero-shot:** Prompt LLMs with "Is this Bengali financial article misinformation? Answer YES or NO."
2. **Fine-tuned:** Train on 80% split, evaluate on 10% test + real-world test set
3. **All models:** Same test set, same metrics, same random seeds
4. **Reporting:** Mean ± std over 3 runs with different seeds

---

## Phase 6: Real-World Validation 🆕 NOT STARTED

Synthetic data has limitations. We need real-world examples to test generalization.

| Task | Status | Notes |
|------|--------|-------|
| Collect 200-500 real misinformation examples | ⏳ Pending | From Bengali financial news, social media |
| Expert annotation | ⏳ Pending | Financial domain experts label real cases |
| Cross-domain evaluation | ⏳ Pending | Test synthetic-trained models on real data |
| Gap analysis | ⏳ Pending | Where do models fail on real vs synthetic? |

### Real-World Sources

- **Dhaka Stock Exchange news:** Insider trading rumors, stock manipulation claims
- **Banking fraud reports:** Phishing scams, fake loan offers, bKash/Nagad fraud
- **Ponzi scheme announcements:** Investment scam patterns
- **Social media:** Facebook groups, WhatsApp forwards about financial schemes

### Expected Outcomes

- **Performance drop:** Synthetic-trained models will likely perform worse on real data
- **Error patterns:** Different failure modes on real vs synthetic
- **Contribution:** Quantifying the synthetic-to-real gap is itself a contribution

---

## Phase 7: Error Analysis 🆕 NOT STARTED

Deep analysis of model failures. This phase produces insights, not just numbers.

| Task | Status | Notes |
|------|--------|-------|
| Error taxonomy | ⏳ Pending | Categorize all model errors |
| Feature analysis | ⏳ Pending | Which linguistic features predict errors? |
| Perturbation failure modes | ⏳ Pending | Why do perturbations succeed or fail? |
| Model comparison errors | ⏳ Pending | Do different models fail on same examples? |
| Write analysis section | ⏳ Pending | Core contribution to paper |

### Error Categories

| Category | Description | Expected Insight |
|----------|-------------|------------------|
| Policy reversal failure | Model misses policy-direction changes | Limited understanding of financial directionality |
| Numerical innumeracy | Model ignores changed numbers | Weak numerical reasoning in Bengali |
| Entity confusion | Model can't track entity replacement | Poor named entity recognition |
| Temporal failure | Model misses time shifts | Insufficient time reasoning |
| Causal failure | Model misses causal inversion | Insufficient discourse understanding |
| Over-reliance on surface features | Model uses shortcuts | Dataset bias or annotation artifacts |

### Analysis Methods

- **Confusion matrix analysis:** Which errors are most common?
- **Attention visualization:** Where do transformer models look?
- **Feature ablation:** What makes an example hard?
- **Error propagation:** Do filtering errors compound in evaluation?

---

## Phase 8: Paper Writing 🆕 NOT STARTED

RQ-focused structure. Every section serves a research question.

| Section | Content | RQ Served | Status |
|---------|---------|-----------|--------|
| Abstract | Problem, method, results, contribution | All | ⏳ |
| Introduction | Motivation, research questions, contributions | All | ⏳ |
| Related Work | Financial misinformation detection, Bengali NLP, benchmarks | RQ1 | ⏳ |
| Benchmark Construction | Dataset generation, filtering, human validation | RQ1, RQ2 | ⏳ |
| Experimental Setup | Models, metrics, splits, real-world test set | RQ3, RQ5 | ⏳ |
| Results | Main findings, per-RQ analysis | RQ3, RQ4, RQ5 | ⏳ |
| Error Analysis | Failure modes, linguistic analysis | RQ4 | ⏳ |
| Discussion | Implications, limitations, future work | All | ⏳ |
| Conclusion | Summary, contributions | All | ⏳ |

### Contribution Claims

1. **FinFact-BD:** First benchmark for Bengali financial misinformation with human validation
2. **Perturbation taxonomy:** 5 types of financial misinformation with analysis of detection difficulty
3. **Human validation:** 300 samples × 3 annotators with inter-annotator agreement analysis
4. **Real-world bridge:** Quantifying synthetic-to-real performance gap
5. **Error analysis:** Linguistic features that predict model failure on Bengali financial text

---

## Files Inventory

```
papers/IDEA_3_FinFact_BD/
├── README.md                          ✅ Updated with simplified pipeline
├── docs/
│   ├── Agent.md                       ✅ Updated with Kaggle T4 workflow
│   ├── CHECKLIST.md                   ✅
│   ├── TIMELINE.md                    ✅
│   ├── PROGRESS.md                    ✅ This file (rewritten for benchmark vision)
│   ├── ANNOTATION_GUIDELINES.md       ⏳ TODO (Phase 3)
│   └── Research_philosophy.md         ✅
├── configs/
│   └── default.yaml                   ✅ All configuration
├── src/
│   ├── config.py                      ✅ YAML config loader
│   ├── generation/
│   │   ├── pipeline.py                ✅ Planning-guided orchestration
│   │   ├── claim_extraction.py        ✅ Heuristic or Qwen JSON extraction
│   │   ├── perturbation_planner.py    ✅ Heuristic or Qwen JSON planning
│   │   ├── rewrite_generator.py       ✅ Aya constrained local rewrite
│   │   └── verifier.py                ✅ e5/mDeBERTa/BanglaBERT + deterministic checks
│   ├── validation/
│   │   ├── legacy filters             ✅ Archived under legacy/rule_based_pipeline/
│   │   ├── human_validation_protocol.py  ⏳ TODO (Phase 3)
│   │   └── rebalance_dataset.py       ✅ Ready
│   ├── benchmark/
│   │   ├── difficulty_calibration.py  ⏳ TODO (Phase 4)
│   │   ├── reasoning_labels.py        ⏳ TODO (Phase 4)
│   │   └── split_builder.py           ⏳ TODO (Phase 4)
│   ├── evaluation/
│   │   ├── metrics.py                 ⏳ TODO (Phase 5)
│   │   ├── baselines.py               ⏳ TODO (Phase 5)
│   │   └── llm_evaluation.py          ⏳ TODO (Phase 5)
│   └── analysis/
│       ├── error_taxonomy.py          ⏳ TODO (Phase 7)
│       └── feature_analysis.py        ⏳ TODO (Phase 7)
├── scripts/
│   ├── finfact_bd_xnli_filter_kaggle.ipynb                ❌ Deprecated
│   ├── finfact_bd_dual_model_filter_kaggle.ipynb          ✅ Archived
│   └── finfact_bd_mdeberta_filter_kaggle.ipynb            ✅ Active
├── tests/                            ⏳ TODO
├── paper/                            ⏳ TODO
└── outputs/                          ⏳ TODO

papers/data/finfact_bd/
├── finfact_bd_originals.csv           ✅ 10,000 original rows
├── finfact_bd_perturbed.csv           ✅ 10,000 perturbations
├── finfact_bd_perturbed_rule_filtered.csv      ✅ 9,981 after rule filter
├── finfact_bd_perturbed_rule_filtered.csv.zst  ✅ 13 MB (compressed for Kaggle)
├── finfact_bd_combined.csv            ✅ 20,000 total
└── metadata.json                      ✅ v2.0 schema + stats
```

---

## Timeline Check (12-Week Plan)

| Week | Phase | Milestone | Status |
|------|-------|-----------|--------|
| 1-2 | Phase 1-2 | Dataset generation + rule filtering | ✅ Done |
| 3 | Phase 2 | mDeBERTa quality filtering (Kaggle) | 🔄 In progress |
| 4 | Phase 2-3 | Filter results + human validation protocol | ⏳ |
| 5-6 | Phase 3 | Human annotation (300 × 3 annotators) | ⏳ |
| 7 | Phase 4 | Benchmark construction (difficulty, reasoning, splits) | ⏳ |
| 8-9 | Phase 5 | Model evaluation (all baselines) | ⏳ |
| 10 | Phase 6 | Real-world validation (200-500 examples) | ⏳ |
| 11 | Phase 7 | Error analysis + insights | ⏳ |
| 12 | Phase 8 | Paper writing + submission | ⏳ |

### Key Dependencies

```
Phase 2 (filtering) → Phase 3 (human validation) → Phase 4 (benchmark) → Phase 5 (evaluation)
                                                                          ↓
                                                          Phase 6 (real-world) → Phase 7 (errors) → Phase 8 (paper)
```

### Risk Factors

| Risk | Impact | Mitigation |
|------|--------|------------|
| mDeBERTa filtering fails | Delays everything | Rule-based filter provides fallback (99.8% pass rate) |
| Annotators unavailable | Delays human validation | Use 2 annotators + adjudication instead of 3 |
| LLM API costs exceed budget | Limits evaluation scope | Start with open-source models, LLMs as stretch goal |
| Real-world data hard to collect | Weakens generalization claim | Use publicly available news, focus on synthetic analysis |

---

## Historical Context

This project evolved through several phases. The original vision was a dataset generation pipeline with dual-model quality filtering. Key decisions that shaped the current approach:

1. **Dataset generation:** Rule-based perturbations chosen over LLM-based generation for reproducibility and zero cost
2. **Quality filtering:** Simplified from dual-model ensemble to mDeBERTa-only after testing showed unnecessary complexity
3. **Benchmark pivot:** Shifted from "dataset announcement" to "rigorous benchmark" after recognizing that human validation and real-world testing are essential for lasting contribution
4. **Venue change:** Targeting FinNLP Workshop at COLING 2026 (January 2027) instead of standalone paper
5. **Generation approach shift:** Moved from rule-based token replacement to planning-guided Bangla claim rewriting using role-specific models. The symbolic edits in v1 were controlled and reproducible, but many were too small to be human-legible in long Bengali financial articles. The new approach keeps the claim-level control, uses Qwen for structured extraction/planning, constrains Aya to realize a pre-specified local change, and verifies independently before acceptance.

The original approach (dataset + dual-model filter + train classifier) would have produced a weaker paper. The benchmark approach (human validation + real-world testing + error analysis) produces a stronger contribution.

---

*Last updated: July 14, 2026*
