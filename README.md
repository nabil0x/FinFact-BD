# FinFact-BD Planning-Guided Claim Rewriting

FinFact-BD is a planning-guided rewriting framework for Bangla financial
misinformation generation.

The active implementation is no longer the old symbolic perturbation engine.
Legacy rule-based code is archived under `legacy/rule_based_pipeline/` for
historical comparison only.

## Core Philosophy

The objective is not to fabricate synthetic news articles. The framework
generates controlled misinformation by rewriting exactly one factual claim within
an authentic Bangla financial news article while preserving the surrounding
journalistic context.

Generation operates at the level of factual propositions rather than documents
or isolated tokens. Each source article is treated as a structured collection of
claims. The system extracts candidate claims, selects one according to predefined
criteria, constructs an explicit perturbation plan, realizes that plan through
constrained generation, and independently verifies that the output satisfies all
required constraints before acceptance.

The generation model does not decide what misinformation to create. Its role is
limited to expressing a planned factual modification in natural Bangla while
preserving the style, coherence, and unrelated facts of the original article.

The resulting benchmark consists of localized, auditable, human-legible factual
distortions rather than unrestricted synthetic news.

## Guiding Principles

- Claim-centric generation: the fundamental unit is a factual proposition.
- Planning before generation: every perturbation is explicit before rewriting.
- Constrained realization: the model linguistically realizes the planned change.
- Locality preservation: unrelated claims, entities, numbers, dates, and
  discourse structure must remain unchanged.
- Verification-governed acceptance: independent checks decide whether a sample
  is accepted.
- Regeneration instead of relaxation: failed generations are retried, not
  accepted under weaker constraints.
- Full provenance: every sample records the source, plan, prompt, model,
  decoding parameters, verification results, and regeneration history.
- Human-centered evaluation: annotators judge the focused rewritten claim before
  inspecting the full article.

## Active Pipeline

```text
Article
  -> Claim Extraction
  -> Claim Ranking
  -> Perturbation Planning
  -> Controlled Local Generation
  -> Independent Verification
  -> Regeneration if rejected
  -> Acceptance
  -> Export
  -> Human Validation Workbook
```

## Architecture

```text
src/generation/
  metadata.py              Shared dataclasses and sample schema
  utils.py                 Sentence splitting, feature extraction, IDs, JSON helpers
  claim_extraction.py      Factual proposition extraction interface and heuristic extractor
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

## Model Roles

The pipeline deliberately uses different models for different cognitive roles.
No model is asked to extract, plan, generate, and verify at the same time.

| Role | Default model | Responsibility |
|------|---------------|----------------|
| Claim extraction | `Qwen/Qwen3-8B` | Convert Bangla article sentences into structured factual claim JSON |
| Rewrite planning | `Qwen/Qwen3-8B` | Decide the target span, perturbation family, intended change, and constraints |
| Controlled rewrite | `CohereLabs/aya-expanse-8b` | Rewrite the planned local claim in fluent Bangla |
| Semantic similarity | `intfloat/multilingual-e5-large` | Check that the article remains globally close to the source |
| Contradiction | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` | Check claim-level contradiction / intended change |
| Language quality | `csebuetnlp/banglabert` | ELECTRA discriminator signal for Bangla language quality |
| Locality and hallucination | deterministic checks | Detect sentence drift and unplanned entities, numbers, dates, and organizations |

Qwen is used for structured reasoning, Aya is used for Bangla realization, and
the verifier stack is independent from the generator. This separation is a core
methodological constraint, not an implementation detail.

For Kaggle T4 runs, the two 8B models are configured with lazy loading and
`unload_after_call: true` so Qwen and Aya are not resident on GPU at the same
time. Aya is hosted behind Hugging Face access controls; set `HF_TOKEN` in the
Kaggle notebook and accept the model license before running the production
config.

## Rewrite Families

- `numerical_fact`
- `policy_reversal`
- `entity_replacement`
- `temporal_shift`
- `causal_inversion`

## Verification

The verifier is modular. Each stage returns a score, pass/fail flag, reason, and
details:

- intended change
- locality
- multilingual-e5 semantic similarity
- mDeBERTa-XNLI contradiction
- BanglaBERT ELECTRA language quality / fluency
- journalistic style
- hallucination/new-fact detection
- corpus-level embedding duplicate detection

Failed generations are never accepted silently. The regeneration controller
tries up to the configured attempt limit and exports only passing samples.

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the active pipeline:

```bash
python scripts/run_rewrite_pipeline.py \
  --config configs/rewrite_pipeline.yaml \
  --input data/finfact_bd/finfact_bd_originals.csv \
  --output-dir data/generated/rewrite_generation \
  --num-samples 100
```

The production config uses Hugging Face models for extraction, planning,
generation, embeddings, NLI, and language-quality verification. Tests inject
lightweight fake models so unit tests do not require a GPU or model downloads.

Run tests:

```bash
python -m pytest -q
```

For Kaggle smoke, pilot, full-run, resume, and output-inspection commands, see
`docs/KAGGLE_RUN_COMMANDS.md`.

## Outputs

The exporter writes:

- `finfact_bd_rewritten.csv`
- `finfact_bd_rewritten.jsonl`
- `metadata.json`
- `human_validation.xlsx` when enabled

Every accepted sample contains:

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

## Legacy Archive

The previous symbolic perturbation implementation is preserved in:

```text
legacy/rule_based_pipeline/
```

It is not imported by `src/generation/` and is excluded from default test
discovery. Use it only for historical comparison.
