# AI Research Agent Guide: FinFact-BD

**Version:** 4.0  
**Target Venue:** FinNLP Workshop / COLING 2026  
**Last Updated:** July 14, 2026

---

## Mission

FinFact-BD is a benchmark for Bengali financial misinformation detection. It is
not a dataset announcement and it is not a free-form synthetic news generator.

The scientific contribution is a planning-guided claim rewriting framework:
controlled misinformation is produced by changing exactly one factual claim
inside an authentic Bangla financial news article while preserving the
surrounding journalistic context.

Every agent working on this repository must prioritize scientific rigor,
auditability, reproducibility, and human-legible evaluation over implementation
shortcuts.

---

## Core Philosophy

The framework does not fabricate complete synthetic articles. It treats each
source article as a structured collection of factual propositions.

The pipeline:

```text
Authentic article
  -> Claim extraction
  -> Claim ranking
  -> Explicit perturbation planning
  -> Constrained local generation
  -> Independent verification
  -> Regeneration if rejected
  -> Acceptance
  -> Dataset export
  -> Human validation workbook
```

The generation model is one component in a constrained generation system. It is
not responsible for deciding what misinformation to create. The planner decides:

- which claim will be modified
- which perturbation family applies
- what factual change is intended
- where the edit is allowed to occur
- which facts must remain unchanged
- which verification constraints must pass

The model only realizes that planned modification in fluent Bangla. Independent
verification governs acceptance. The model never has the final say.

---

## Non-Negotiables

- Do not restore the legacy rule-based generator into the active package.
- Do not implement regex/entity/numeric substitution as the default generator.
- Do not let the model freely rewrite or invent articles.
- Do not accept failed generations by relaxing verification thresholds silently.
- Do not claim multi-hop perturbation unless code explicitly implements it.
- Do not infer implementation from comments, TODOs, or documentation.
- Do not add hidden fallbacks or silent exception handlers.
- Do not drop provenance fields from exported samples.

Legacy code is preserved only for historical comparison in:

```text
legacy/rule_based_pipeline/
```

Active code must not import from that directory.

---

## Active Architecture

```text
src/generation/
  metadata.py              Shared dataclasses and sample schema
  utils.py                 Sentence splitting, feature extraction, IDs, JSON helpers
  claim_extraction.py      Factual proposition extraction interface
  claim_selection.py       Claim ranking and quality gate
  perturbation_planner.py  Structured rewrite-plan generation
  prompts.py               Prompt templates and prompt versioning
  models.py                Generation, embedding, NLI, and fluency model interfaces
  rewrite_generator.py     Constrained localized rewrite realization
  verifier.py              Independent verification modules
  regeneration.py          Retry controller with attempt records
  exporter.py              Dataset and human-validation workbook export
  pipeline.py              End-to-end orchestration and checkpointing
```

The default pipeline class is:

```python
PlanningGuidedRewritePipeline
```

The default CLI is:

```bash
python scripts/run_rewrite_pipeline.py \
  --config configs/rewrite_pipeline.yaml \
  --input data/finfact_bd/finfact_bd_originals.csv \
  --output-dir data/generated/rewrite_generation \
  --num-samples 100
```

---

## Model Role Allocation

Use separate models for separate cognitive roles. Do not collapse extraction,
planning, generation, and verification into a single model call.

| Role | Default model | Rule |
|------|---------------|------|
| Claim extraction | `Qwen/Qwen3-8B` | Produce structured claim JSON only |
| Rewrite planning | `Qwen/Qwen3-8B` | Produce a rewrite plan only; never rewrite text |
| Controlled rewrite | `CohereLabs/aya-expanse-8b` | Realize the planned local edit in Bangla |
| Semantic similarity | `intfloat/multilingual-e5-large` | Compare original and rewritten articles |
| Contradiction | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | Verify original and rewritten target claims |
| Language quality | `csebuetnlp/banglabert` | Provide masked-LM Bangla language-quality signal |
| Locality/hallucination | deterministic checks | Enforce changed sentence count and unplanned-new-fact rejection |

The generator must never be reused as the verifier. Verification must remain
independent from generation.

---

## Pipeline Contracts

### 1. Claim Extraction

Input:

- `headline`
- `article`
- `metadata`

Output:

```python
Claim(
    sentence_index,
    sentence,
    claim_text,
    claim_type,
    entities,
    numbers,
    policies,
    dates,
    confidence,
    extractor_model,
)
```

The extractor identifies sentence-level factual propositions, not isolated
regex spans. The current implementation is heuristic, but the interface must
allow an LLM or parser-based extractor to replace it without changing downstream
code.

### 2. Claim Ranking

Each claim receives:

- `importance_score`
- `editability_score`
- `verification_score`
- `locality_score`
- `risk_score`

Overall score:

```text
0.35 * importance
+ 0.25 * editability
+ 0.20 * locality
+ 0.20 * verification
```

Only high-quality, low-risk claims should proceed to planning.

### 3. Perturbation Planning

The planner produces a structured `RewritePlan`:

```python
RewritePlan(
    family,
    target_claim,
    edit_instruction,
    edit_scope,
    expected_change,
    verification_constraints,
    target_span,
    replacement,
    planner_model,
)
```

Supported families:

- `numerical_fact`
- `policy_reversal`
- `entity_replacement`
- `temporal_shift`
- `causal_inversion`

The planner decides what changes. The generator does not.

### 4. Constrained Local Generation

Generation is prompt-based and localized. Prompt templates belong in
`src/generation/prompts.py`, not inline inside orchestration code.

Prompts must require:

- preserve journalistic style
- rewrite only the selected claim
- preserve unrelated facts
- avoid additional hallucinations
- return the complete rewritten article

The model may rewrite the selected sentence or local paragraph as directed by
the plan. It must not invent a new article.

### 5. Verification

Verification is the acceptance gate. Each verifier returns:

```python
VerifierResult(
    name,
    score,
    passed,
    reason,
    details,
)
```

Required verifier categories:

- intended claim modification
- locality
- semantic similarity using embeddings
- contradiction using NLI
- fluency using perplexity or equivalent model signal
- journalistic style
- hallucination / new-fact detection
- duplicate detection

Verification failures must be visible in exported metadata and logs.

### 6. Regeneration

If verification fails, regenerate up to the configured attempt limit, currently
three attempts by default. Attempts may vary temperature or prompt wording, but
must keep the same planned factual change and constraints.

Failed generations are rejected. They are not accepted through weakened
criteria.

### 7. Export

Every accepted sample must preserve:

- `sample_id`
- `article_id`
- `headline`
- `original_article`
- `rewritten_article`
- `selected_claim`
- `claim_index`
- `claim_type`
- `perturbation_family`
- `rewrite_plan`
- `generator_model`
- `model_revision`
- `prompt_version`
- `temperature`
- `seed`
- `verification_scores`
- `regeneration_attempts`
- `timestamp`

Exports must include dataset files, metadata, and human-validation workbooks
when enabled.

### 8. Human Validation Workbook

Human validation is claim-first. Annotators should not be asked to hunt through
full articles before seeing the relevant claim.

Workbook sheets:

- `Instructions`
- `Samples`
- `Full Articles`

The `Samples` sheet must include:

- headline
- claim focus
- context window
- label
- confidence
- justification

---

## Research Questions

Every task should serve at least one research question.

| ID | Question | Why It Matters |
|----|----------|----------------|
| RQ1 | Which rewrite families are hardest for current models? | Reveals specific linguistic weaknesses in Bengali NLP systems |
| RQ2 | Do Bengali-specific PLMs outperform multilingual models on financial text? | Tests whether language-specific pretraining helps in this domain |
| RQ3 | How do open-weight LLMs compare to fine-tuned smaller models? | Probes scale versus specialization |
| RQ4 | Can models generalize from controlled rewrites to real-world misinformation? | Tests ecological validity |
| RQ5 | What systematic errors do models make across perturbation families? | Enables targeted future work |

If a task does not serve any RQ, challenge whether it belongs in the project.

---

## Human Validation Protocol

Human annotation is mandatory for benchmark credibility.

| Parameter | Value |
|-----------|-------|
| Samples | 300 |
| Annotators | 3 per sample |
| Agreement threshold | Fleiss' kappa >= 0.6 |
| Task | Claim-centric `original` / `perturbed` / `not sure` |
| Evidence shown first | Headline, claim focus, context window |
| Full article | Available after the focused claim view |

Annotators judge whether the focused claim appears original, perturbed, or
uncertain. They provide confidence and a short justification. The protocol is
designed to evaluate plausibility of localized factual misinformation, not the
ability to find a hidden edit in a long article.

---

## Engineering Standards

Research code is still production code for a paper.

### Required

- Python 3.11+
- dataclasses for shared records
- type hints throughout active code
- YAML configuration
- dependency injection for generation, embedding, NLI, and fluency models
- logging at pipeline boundaries and failure points
- explicit random seeds
- checkpointing for long runs
- tests for every pipeline stage
- no active module above roughly 400 lines
- no circular imports
- no silent exception swallowing

### Preferred

- batch generation and embedding where possible
- GPU use only through model adapters
- lazy imports for heavyweight ML dependencies
- small, composable modules over monolithic scripts

### Validation Commands

```bash
python -m pytest -q
python -m compileall -q src scripts tests
```

---

## Compute Environment

| Environment | Hardware | Purpose |
|-------------|----------|---------|
| Local machine | Limited GPU | Code development, tests, preprocessing, documentation |
| Kaggle | T4 GPU | Large generation runs, embedding/NLI/fluency verification, model evaluation |

GPU-intensive tasks should run on Kaggle or an equivalent GPU environment. Local
runs should remain lightweight and testable through injected fake models.

When using Hugging Face dataset downloads, include `repo_type="dataset"` where
required so the hub does not search model repositories.

---

## Data Integrity

- Never overwrite original BENI data.
- Generated releases are immutable once published.
- Every generated sample must be traceable back to source article and rewrite
  plan.
- Train/validation/test splits must be grouped by source article ID to prevent
  leakage.
- Do not commit large generated artifacts unless they are intended release
  artifacts.
- Record model name, model revision, prompt version, seed, temperature, and
  verification results for every accepted sample.

---

## Testing Standards

Pipeline tests should cover:

- claim extraction returns sentence-level factual claims
- ranking applies the documented score formula
- planner emits structured `RewritePlan` metadata
- generator receives prompts rather than performing symbolic replacements
- verifier modules return score/pass/reason records
- failed verification triggers regeneration
- exports include required metadata fields
- workbook is claim-first
- active package imports no legacy generation code

Tests may use fake model adapters for speed. Fakes belong in tests, not
production modules.

---

## Paper Writing Standards

The paper should emphasize the scientific contribution, not implementation
mechanics.

Use this framing:

> FinFact-BD introduces planning-guided claim rewriting for Bangla financial
> misinformation generation, integrating explicit perturbation planning,
> constrained local generation, independent verification, regeneration, and full
> provenance.

Avoid weaker or misleading framing:

- "LLM as executor"
- "we generated fake news articles"
- "rule-based perturbation benchmark"
- "article-level rewriting"
- "multi-hop perturbation" unless implemented and validated

Every claim must be backed by implementation, experiment, citation, or human
validation. If the code does not implement it, the paper must not claim it.

---

## Agent Behavior

When assigned a task:

1. Read the relevant code before making claims.
2. Identify which RQ or pipeline contract the task serves.
3. Preserve the planning-guided architecture.
4. Keep legacy code isolated.
5. Make modular changes with tests.
6. Update documentation when architecture or outputs change.
7. Report remaining assumptions and validation gaps.

Before finishing, confirm:

- code runs or explain why it was not run
- tests pass or failures are reported
- metadata/provenance is preserved
- no hidden fallbacks were added
- no legacy imports were introduced
- research claims match implemented code

---

## Literature Monitoring

Continuously track work on:

- Bengali misinformation detection
- financial misinformation and fact verification
- adversarial benchmark construction
- controlled text generation
- counterfactual data augmentation
- human validation protocols
- low-resource financial NLP

Report missing citations or new conflicting evidence immediately.

---

## Long-Term Vision

FinFact-BD should become a trusted benchmark for Bengali financial
misinformation detection. Success is not measured by the number of generated
samples. It is measured by whether future researchers can reproduce, audit, and
use the benchmark to understand model behavior.

The benchmark should be localized, auditable, human-legible, and scientifically
defensible.
