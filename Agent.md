# AI Research Agent Guide: FinFact-BD Benchmark

**Version:** 3.0
**Target Venue:** FinNLP Workshop, COLING 2026
**Last Updated:** July 14, 2026

---

## Mission

FinFact-BD is a rigorously validated benchmark for evaluating automatic detection of Bengali financial misinformation. It is not a dataset announcement. It is a controlled experimental setup designed to answer questions about model behavior on Bengali financial text.

Every decision should prioritize scientific rigor, reproducibility, and lasting community value over shortcuts or speed.

---

## Research Philosophy

Five principles guide every decision in this project.

**Benchmark > Dataset.**
A dataset is raw material. A benchmark is a validated instrument for measuring model performance. We are building an instrument. The dataset is a component, not the contribution.

**Research Questions > "We built X".**
Papers framed around "we built a dataset" are forgettable. Papers framed around "we answer these questions" are cited. Every experiment exists to serve at least one RQ. If you cannot name the RQ, do not run the experiment.

**Human validation is mandatory.**
A benchmark without human-validated ground truth is just algorithmic noise. Human annotation answers questions machines cannot: Does this look realistic? Could this spread on social media? Would an investor believe it? We validate 300 samples across 5 claim-rewriting families, including multi-hop compositions, with 3 annotators each.

**Error analysis is a contribution.**
Accuracy numbers are commodities. Understanding why models fail is insight. Our error analysis is not an afterthought. It is a core contribution. False negatives, false positives, confusion patterns, and linguistic feature analysis reveal what current Bengali NLP models actually understand.

**Reproducibility is non-negotiable.**
Every figure, table, metric, and dataset split must be regeneratable from source code. No manual edits. No hidden preprocessing. No undocumented scripts. Another researcher should be able to reproduce every result in the paper.

---

## Research Questions

These RQs drive every phase of the benchmark. Nothing happens without serving at least one.

| ID | Question | Why It Matters |
|----|----------|----------------|
| **RQ1** | Which claim-rewriting families are hardest for current models? | Reveals specific linguistic weaknesses in Bengali NLP systems |
| **RQ2** | Do Bengali-specific PLMs outperform multilingual models on financial text? | Tests whether language-specific pretraining helps for domain-specific tasks |
| **RQ3** | How do open-weight LLMs compare to fine-tuned smaller models? | Probes the tradeoff between generalization and specialization |
| **RQ4** | Can models generalize from synthetic rewrites to real-world misinformation? | Validates ecological validity of the controlled rewriting approach |
| **RQ5** | What systematic errors do models make across perturbation types? | Enables targeted improvement in future work |

When assigned a task, identify which RQ it serves. If it does not serve any RQ, question whether it should exist.

---

## Compute Environment

| Environment | Hardware | Purpose |
|-------------|----------|---------|
| Local machine | 6GB GPU (limited) | Code development, data preprocessing, analysis, git operations |
| Kaggle | T4 GPU (16GB VRAM) | Primary compute: model fine-tuning, quality filtering, large-scale inference |

**Rule:** All GPU-intensive tasks run on Kaggle. Local machine is for CPU work only.

### Kaggle Workflow

1. Develop scripts locally in `src/`
2. Upload scripts + compressed data to Kaggle notebook (use zstd, max 20MB per file)
3. Run GPU-intensive tasks on T4
4. Download results (filtered CSVs, model checkpoints, metrics)
5. Continue analysis locally

**Dataset Loading Note:** When using `hf_hub_download`, always include `repo_type="dataset"`. Without it, HuggingFace searches model repos instead of dataset repos, causing 401 errors.

**Active Quality Filter:** The generator now uses a multi-signal validation gate (proposition extraction, fluency, semantic similarity, contradiction scoring). `scripts/mdeberta_filter.py` and `scripts/finfact_bd_mdeberta_filter_kaggle.ipynb` remain as an optional downstream NLI filter (mDeBERTa-v3-only, threshold >= 0.4).

---

## Human Validation Protocol

Human annotation is the backbone of benchmark quality. Without it, we cannot claim our perturbations produce detectable misinformation.

### Protocol

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Samples | 300 | Stratified: 60 per perturbation type |
| Annotators | 3 per sample | Majority vote for gold label |
| Agreement threshold | Fleiss' κ ≥ 0.6 | Substantial agreement required |
| Annotation task | Claim-centric: `original` / `perturbed` / `not sure` | Focuses annotation on the edited claim rather than whole-article hunting |
| Training | 10 calibration samples with known labels | Ensures annotators understand the task |
| Time estimate | ~2 hours per annotator | 300 samples, ~24 seconds each |

Annotators receive separate XLSX workbooks. The `Samples` sheet shows the headline, a focused claim sentence, adaptive nearby context, and the full article text; the optional `Full Articles` sheet is available only if needed for quicker scrolling. Dropdown labels, required confidence, and justification fields keep the annotation structured. A hidden `claim_type` column is reserved for later analysis only. A separate manifest keeps the sample-to-source mapping for merging labels after annotation.

### Annotation Task

"Given a Bengali financial claim window, determine whether the highlighted claim appears original, perturbed, or not sure. Judge only the highlighted claim, not the whole article. Mark `perturbed` when the claim appears to contain a factual edit (number, entity, time, policy direction, comparison, quotation, or cause-effect relation) and `original` when no clear edit is visible."

### Labels

- `original`: The highlighted claim reads like an unmodified article and no clear factual edit is visible from the provided evidence
- `perturbed`: The highlighted claim appears to contain a clear factual alteration or other synthetic edit
- `not sure`: The annotator cannot confidently determine whether the claim is original or perturbed from the provided evidence

### Selection Strategy

300 samples stratified by:
- Perturbation type (60 per type, all 5 types covered)
- Quality score range (low, medium, high from the validation gate)
- Article length (short, medium, long)
- Entity density (few, many named entities)

This ensures we test the filter across diverse conditions.

### Impact

- Filtered dataset may be adjusted based on human validation results
- Human labels become the gold standard for benchmark construction
- Models evaluated against human-validated ground truth
- Error analysis grounded in human judgment

---

## Benchmark Standards

### Model Hierarchy

We evaluate across five model families, ordered by increasing scale and complexity. Each tier tests a different hypothesis about what matters for Bengali financial misinformation detection.

| Tier | Models | Hypothesis Tested |
|------|--------|-------------------|
| **Classical** | TF-IDF + LogReg, FastText, SVM | Can sparse features solve this at all? |
| **Bengali PLM** | BanglaBERT (110M), Bengali Electra (110M) | Does language-specific pretraining help? |
| **Multilingual PLM** | mBERT (110M), XLM-R (270M), ModernBERT (395M) | Does cross-lingual transfer close the gap? |
| **Open LLM** | Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B (all LoRA) | Can general reasoning compensate for domain weakness? |
| **Closed LLM** | GPT-4o, Gemini-1.5-Pro, Claude-3.5-Sonnet (zero-shot only) | What is the ceiling without fine-tuning? |

**Rules:**
- Always use identical train/validation/test splits
- Never compare models trained on different splits
- Report mean ± std over 3 runs with different random seeds
- Closed models are evaluated zero-shot only. No fine-tuning, no few-shot prompting.

### Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| Accuracy | Overall correct classification rate |
| Macro F1 | Balanced across classes regardless of size |
| Per-class F1 | Performance per label (precision, recall) |
| Per-perturbation F1 | Which perturbation types are hardest? |
| Per-difficulty F1 | Does difficulty calibration work? |
| Confusion Matrix | Error distribution across perturbation types |
| Cross-perturbation robustness | Can models handle unseen perturbation types? |
| Human agreement correlation | Do model errors align with human disagreements? |
| Calibration | Are model confidence scores meaningful? |

### Statistical Significance

- McNemar's test for pairwise model comparisons
- Bootstrap confidence intervals for all reported metrics
- Report p-values for all main claims
- Effect sizes (Cohen's h) for practical significance

### Evaluation Protocol

1. **Splits:** 80/10/10 train/validation/test, grouped by `original_id` so no source article leaks across partitions
2. **Cross-validation:** 5-fold on training set for model selection
3. **Test set:** Held-out, reported once per model
4. **Real-world test set:** Separate set of actual Bengali financial misinformation (not synthetically perturbed) for ecological validity
5. **Reporting:** Mean ± std over 3 runs with different seeds

---

## Dataset Schema

FinFact-BD carries rich metadata per sample. This schema supports every downstream analysis.

```json
{
  "id": "finfact_001",
  "text": "প্রতিবেদনে বলা হয়েছে...",
  "label": 0,
  "split": "train",
  "perturbation_type": "numerical_fact_change",
  "perturbation_operator": "numeric_shift",
  "perturbation_mode": "multi_hop",
  "hop_count": 3,
  "original_id": "beni_4521",
  "source": "BENI v2",
  "quality_score": 0.87,
  "difficulty": "medium",
  "planned_families": "[\"numerical_fact_change\", \"causal_inversion\", \"temporal_shift\"]",
  "perturbation_families": "[\"numerical_fact_change\", \"causal_inversion\", \"temporal_shift\"]",
  "reasoning_type": "numerical_fraud",
  "changed_span_original": "১০",
  "changed_span_replacement": "৯.৩",
  "changed_span_role": "value",
  "proposition_schema": "{\"family\":\"numerical_fact_change\",...}",
  "validation_scores": "{\"passed\":true,...}",
  "perturbation_plan": "{\"primary_family\":\"numerical_fact_change\",...}",
  "human_annotation": {
    "annotator_2": "perturbed",
    "annotator_1": "perturbed",
    "annotator_3": "not sure",
    "agreement_score": 0.67,
    "gold_label": "perturbed"
  },
  "metadata": {
    "word_count": 245,
    "entity_count": 3,
    "article_length": "medium",
    "date_range": "2020-2024"
  }
}
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier |
| `text` | string | Article text (original or perturbed) |
| `label` | int | 0 = original (reliable), 1 = perturbed (misleading) |
| `split` | string | `train`, `validation`, or `test` |
| `perturbation_type` | string | One of: `numerical_fact_change`, `policy_reversal`, `entity_replacement`, `temporal_shift`, `causal_inversion`, or `null` for originals |
| `perturbation_operator` | string | Specific rewrite operator used for the perturbation |
| `perturbation_mode` | string | `single_hop` or `multi_hop` depending on how many propositions changed |
| `original_id` | string | Reference to source article in BENI v2 |
| `source` | string | Data source identifier |
| `quality_score` | float | mDeBERTa contradiction probability (0-1) |
| `difficulty` | string | `easy`, `medium`, or `hard` based on hop count |
| `hop_count` | int | Number of applied perturbation operations |
| `planned_families` | string | JSON list of families selected by the perturbation plan |
| `perturbation_families` | string | JSON list of families actually applied |
| `operation_reasoning_types` | string | JSON list of reasoning labels for each applied operation |
| `reasoning_type` | string | Cognitive demand category |
| `changed_span_original` | string | Original proposition span that was rewritten |
| `changed_span_replacement` | string | Replacement span inserted by the perturbation |
| `changed_span_role` | string | Role of the changed span within the proposition |
| `proposition_schema` | string | JSON metadata describing the rewritten proposition |
| `validation_scores` | string | JSON validation summary with contradiction, similarity, and fluency scores |
| `validation_passed` | bool | Whether the sample passed the multi-signal validation gate |
| `validation_issues` | string | JSON list of validation issues, if any |
| `perturbation_plan` | string | JSON provenance for the full multi-hop perturbation plan |
| `human_annotation` | object | Per-annotator labels, agreement score, gold label |
| `metadata` | object | Word count, entity count, sentiment, length category, date range |

### Difficulty Calibration

| Level | Criteria | Purpose |
|-------|----------|---------|
| Easy | Single-hop perturbation | Sanity check |
| Medium | Two coordinated changes | Core benchmark difficulty |
| Hard | Three coordinated changes | Exposes model limits |

### Reasoning Labels

| Label | Perturbation Type | Cognitive Demand |
|-------|-------------------|------------------|
| `numerical_fraud` | Numerical fact change | Digit attention |
| `policy_reversal` | Policy reversal | Direction reversal / policy tracking |
| `entity_confusion` | Entity replacement | World knowledge grounding |
| `temporal_distortion` | Temporal shift | Time tracking |
| `causal_fallacy` | Causal inversion | Discourse structure |

### Perturbation Types

| Type | Strategy | What It Tests |
|------|----------|---------------|
| Numerical Fact Change | Rewrite one numeric value while preserving the sentence frame | Do models attend to explicit financial values? |
| Policy Reversal | Flip direction words in policy-like or market-direction statements | Can models detect reversals in financial actions? |
| Entity Replacement | Swap an entity with another entity from the same ontology class | Do models ground claims in the right institution or company? |
| Temporal Shift | Change a year, month, or relative time phrase | Do models track when the fact occurred? |
| Causal Inversion | Flip the outcome of a causal relation or causal connector | Do models track cause-effect structure? |

### Difficulty Policy

Difficulty is family-aware rather than purely random. Policy reversal and entity replacement are biased toward easier one-hop edits, numerical and temporal changes center on medium difficulty, and causal inversion is biased toward harder multi-hop composition. The sampler also tracks global difficulty budgets so the final benchmark stays close to the intended 25/45/30 easy/medium/hard mix without introducing new perturbation operators.

The validation gate uses difficulty-aware contradiction thresholds so easy samples are not over-filtered by the same cutoff used for harder samples.
Because the perturbations are proposition-level edits, the semantic-similarity upper bound is intentionally permissive: we want to keep near-original factual rewrites and reject only pathological duplicates, not high-similarity misinformation.

Each perturbation should resemble misinformation that could realistically appear in newspapers, Facebook posts, investment groups, stock market discussions, banking rumors, or economic commentary. A single sample can combine multiple operations, but the result should still read like plausible financial prose. If a perturbation looks obviously fake, it weakens the benchmark.

---

## Paper Writing Standards

The paper is not the contribution. The benchmark is. The paper explains it.

### Structure

Every section serves a research question. If a section does not answer an RQ, it should not exist.

| Section | RQ Served | Content |
|---------|-----------|---------|
| Abstract | All | Problem, method, results, contribution |
| Introduction | All | Motivation, research questions, contributions |
| Related Work | RQ1 | Financial misinformation detection, Bengali NLP, benchmarks |
| Benchmark Construction | RQ1, RQ2 | Dataset generation, filtering, human validation |
| Experimental Setup | RQ3, RQ5 | Models, metrics, splits, real-world test set |
| Results | RQ3, RQ4, RQ5 | Main findings, per-RQ analysis |
| Error Analysis | RQ4 | Failure modes, linguistic analysis |
| Discussion | All | Implications, limitations, future work |
| Conclusion | All | Summary, contributions |

### Writing Rules

- Frame around RQs, not "we built a dataset"
- Use "To the best of our knowledge" instead of "first" unless verified through literature review
- Error analysis is a major contribution, not an afterthought
- Limitations section is required. Every limitation stated honestly increases credibility.
- Every claim must be backed by experiment, citation, or analysis
- Avoid marketing language. "Novel", "state-of-the-art", "first ever" are red flags unless rigorously justified.
- Avoid unsupported claims. If you cannot prove it, do not say it.
- Report negative results. If an experiment fails, document why. Negative results are still knowledge.

### Contribution Claims

1. **FinFact-BD:** A benchmark for Bengali financial misinformation with human validation
2. **Perturbation taxonomy:** Five types of financial misinformation with analysis of detection difficulty
3. **Human validation:** 300 samples × 3 annotators with inter-annotator agreement analysis
4. **Real-world bridge:** Quantifying synthetic-to-real performance gap
5. **Error analysis:** Linguistic features that predict model failure on Bengali financial text

---

## Error Analysis

This is not optional. It is a core contribution.

### Error Categories

| Category | Description | Expected Insight |
|----------|-------------|------------------|
| Numerical innumeracy | Model ignores changed numbers | Weak numerical reasoning in Bengali |
| Entity confusion | Model can't track entity replacement | Poor named entity recognition |
| Policy reversal failure | Model misses direction flips | Weak tracking of financial actions |
| Temporal drift | Model misses date shifts | Limited time reasoning |
| Causal failure | Model misses causal inversion | Insufficient discourse understanding |
| Surface shortcuts | Model uses statistical artifacts | Dataset bias or annotation artifacts |

### Analysis Methods

- Confusion matrix analysis across perturbation types
- Attention visualization for transformer models
- Feature ablation: what makes an example hard?
- Error propagation: do filtering errors compound in evaluation?
- Per-difficulty breakdown: where do models succeed and fail?
- Real-world vs synthetic comparison: do synthetic failures predict real failures?

### Reporting

Every error analysis must produce:
- Quantitative summary (error counts, rates, distributions)
- Qualitative examples (at least 5 per error category)
- Linguistic feature analysis (what makes hard examples hard?)
- Actionable insights (what should future work focus on?)

---

## Coding Standards

Research code should communicate ideas, not cleverness.

### Requirements

- Python 3.11+
- Use `pathlib`, `pandas`, `numpy`, `polars` where useful, `dataclasses`, `typer`, `pydantic`
- Avoid notebooks for production code
- Everything runnable from CLI

```bash
python src/generation/perturbation_pipeline.py
python src/validation/rule_based_filter.py
python scripts/mdeberta_filter.py
python src/training/train.py --model banglabert --split train
python src/evaluation/evaluate.py --model banglabert --split test
python src/evaluation/error_analysis.py --model banglabert
```

### Documentation

Every module must explain:
- Purpose
- Inputs
- Outputs
- Assumptions
- Limitations
- Time complexity if relevant

---

## Repository Structure

```
IDEA_3_FinFact_BD/
├── configs/
│   └── default.yaml
├── src/
│   ├── config.py
│   ├── generation/
│   │   ├── perturbation_pipeline.py
│   │   └── extract_originals.py
│   ├── validation/
│   │   ├── rule_based_filter.py
│   │   ├── mdeberta_filter.py
│   │   ├── human_validation_protocol.py
│   │   └── rebalance_dataset.py
│   ├── benchmark/
│   │   ├── difficulty_calibration.py
│   │   ├── reasoning_labels.py
│   │   └── split_builder.py
│   ├── training/
│   │   └── train.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── baselines.py
│   │   ├── llm_evaluation.py
│   │   └── error_analysis.py
│   ├── analysis/
│   │   ├── error_taxonomy.py
│   │   └── feature_analysis.py
│   └── visualization/
├── scripts/
│   ├── mdeberta_filter.py
│   └── finfact_bd_mdeberta_filter_kaggle.ipynb
├── paper/
├── docs/
├── outputs/
├── tests/
└── README.md
```

---

## Testing Standards

Every component needs tests. Research code is still code.

### Perturbation Tests

Each perturbation module requires unit tests verifying:
- Input produces expected output
- Metadata updated correctly
- Original text preserved
- Edge cases handled (empty text, no matching entities, Bengali Unicode)

```
Input:  GDP increased by 5%.
Output: GDP decreased by 5%.
Check:  token changed, metadata updated, original preserved
```

### Validation Tests

- Rule-based filter correctly rejects/accepts samples
- mDeBERTa filter produces scores in expected range
- Human validation protocol selects stratified samples
- Rebalancing maintains class distribution

### Evaluation Tests

- Metrics computed correctly on known inputs
- Split sizes match specifications
- Statistical tests produce valid p-values
- Error analysis produces expected categories

### Performance Targets

- Generate 20,000 samples in under 30 minutes on modern desktop
- Prefer vectorization over loops
- Profile before optimizing

---

## Data Integrity

- Never overwrite original BENI data
- Generated datasets are immutable once released
- Use versioning: `FinFact-BD-v1.0` for the frozen public release; reserve `v1.1+` for future regenerated releases
- Preserve provenance: every generated sample must track its source, generator version, timestamp, and validation status

### Provenance Fields

Every generated misinformation sample must keep:
- `sample_id`
- `original_article_id`
- `perturbation_type`
- `original_text`
- `generated_text`
- `modified_tokens`
- `generator_version`
- `timestamp`
- `validation_status`

---

## Git Standards

- Meaningful commits
- Example: `feat: implement numerical_fact_change`, `fix: preserve Bengali unicode digits`
- Never commit generated artifacts unless intended

---

## Agent Behavior

When assigned a task:
1. Understand which RQ it serves
2. Review existing implementation
3. Suggest improvements
4. Implement modular code
5. Write tests
6. Document changes
7. Identify research implications

Before finishing any task, confirm:
- Code runs
- Tests pass
- Documentation updated
- Reproducibility maintained
- Metadata preserved
- No hidden assumptions
- Research quality improved

---

## Literature

Continuously search for:
- Financial misinformation detection
- Fake news detection
- Financial NLP
- Low-resource NLP
- Dataset papers
- ACL benchmark papers

Report missing citations immediately.

---

## Long-Term Vision

FinFact-BD should become the reference benchmark for Bengali financial misinformation detection. It should enable reproducible evaluation of financial NLP models and serve as a foundation for future work in low-resource financial AI.

Success is not measured by downloads. It is measured by whether future papers cite the dataset because they trust it.

---

*Last updated: July 14, 2026*
