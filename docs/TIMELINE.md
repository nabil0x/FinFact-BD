# Timeline & Milestones: FinFact-BD Benchmark

## Overview

Twelve weeks from quality filtering to FinNLP Workshop submission. Each week maps to a single phase with a concrete deliverable.

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Quality Filtering | Filtered dataset + quality report |
| 2 | Human Validation | Inter-annotator agreement report |
| 3 | Benchmark Construction | Annotated dataset with splits |
| 4 | Classical + Bengali PLM Baselines | Baseline results table |
| 5 | Multilingual + LLM Evaluation | Full model comparison |
| 6 | Real-World Validation | Real-world test set results |
| 7 | Error Analysis | Per-failure-type breakdown |
| 8 | Ablations + Visualizations | Ablation tables + figures |
| 9 | Dataset Release Prep | GitHub + HuggingFace + Zenodo |
| 10 | Paper Writing | Complete draft |
| 11 | Revision | Revised draft + reproducibility artifacts |
| 12 | Submission | FinNLP submission |

**Target venue:** FinNLP Workshop, COLING 2026
**Total duration:** 12 weeks

---

## Phase 1: Quality Filtering & Planning-Guided Rewriting (Week 1)

Run the planning-guided Bangla rewriting pipeline on Kaggle with role-specific models: Qwen3-8B for extraction/planning, Aya Expanse 8B for controlled rewriting, multilingual-e5-large for similarity, mDeBERTa-XNLI for contradiction, and BanglaBERT for language quality. The generation pipeline includes built-in multi-stage verification, so mDeBERTa acts as one verifier rather than the sole quality gate.

### Tasks

| Task | Details |
|------|---------|
| Implement claim extraction & selection on Kaggle | Extract financial propositions, score by centrality/importance/editability/diversity |
| Load role-specific models on Kaggle T4 | Load Qwen and Aya sequentially in 4-bit plus verifier models |
| Generate planning-guided rewrites | Rewrite targeted claims under planner control, one factual change per sample |
| Run multi-stage verification | Claim integrity, surface quality, semantic quality checks before acceptance |
| Run regeneration loop for failed samples | Up to 3 attempts per failed sample, log failure reasons for dropped samples |
| Export raw, filtered, and metadata files | Write finfact_bd_rewritten_raw.csv, finfact_bd_rewritten_filtered.csv, metadata.json |
| Run mDeBERTa-v3 on Kaggle T4 | Score all rewritten pairs for contradiction probability as downstream quality gate |
| Generate filtering statistics | Count retained/removed per rewrite family |
| Produce quality report | Threshold sensitivity analysis, distribution plots |
| Document removed examples | 10 examples per rewrite family showing what was filtered and why |

### Deliverables

- `finfact_bd_rewritten_raw.csv` with all generated rewrites before filtering
- `finfact_bd_rewritten_filtered.csv` with accepted samples after multi-stage verification and mDeBERTa filtering
- `metadata.json` with full provenance (source article, target claim, selection scores, generator model, rewrite plan, verification result, regeneration attempts)
- Filtering statistics summary (table + histogram)
- Quality report with threshold analysis

---

## Phase 2: Human Validation (Week 2)

Validate the dataset through human annotation. Three Bengali-native annotators with finance domain familiarity independently label a stratified random sample.

### Protocol

| Parameter | Value |
|-----------|-------|
| Sample size | 300 articles (150 original, 150 perturbed) |
| Annotators | 3 (A, B, C) |
| Task | Claim-centric `original` / `perturbed` / `not sure` labels with free-text justification and required confidence |
| Stratification | Equal representation across perturbation types |
| Training | 10 shared calibration examples before annotation |

### Metrics

| Metric | Purpose |
|--------|---------|
| Fleiss' kappa | Inter-annotator agreement beyond chance |
| Percent agreement | Raw agreement rate |
| Confidence intervals | Bootstrap CIs for all metrics (1000 resamples) |
| Per-perturbation agreement | Which perturbation types do annotators agree on most? |
| Annotation time | Average time per sample (effort metric) |

### Annotation Dimensions

In addition to the binary label, annotators rate each sample on four 1-5 Likert scales:

- **Realism:** Does this text read like a real news article?
- **Fluency:** Is the Bengali grammatically correct and natural?
- **Contradiction:** Does the perturbed version contradict the original?
- **Financial plausibility:** Could this claim appear in real financial reporting?

### Deliverables

- Inter-annotator agreement report with Fleiss' kappa, CIs, and per-type breakdowns
- Annotated XLSX workbook with per-annotator labels and ratings
- Calibration analysis comparing human labels to mDeBERTa filter labels

---

## Phase 3: Benchmark Construction (Week 3)

Build the final benchmark from the filtered and validated dataset. Every sample receives rich metadata to support the analysis planned in later phases.

### Metadata Fields

| Field | Values | Purpose |
|-------|--------|---------|
| Difficulty level | Easy, Medium, Hard | Based on mDeBERTa contradiction score quantiles |
| Reasoning type | Numerical, Causal, Temporal, Entity, Sentiment | Matches perturbation type |
| Annotation consensus | Unanimous, Majority, Split | Human validation agreement level |
| Text length | Token count | Length-controlled analysis |
| Perturbation intensity | Low, Medium, High | Magnitude of change (e.g., how much the number changed) |
| importance_score | Float | Claim centrality and financial importance |
| editability_score | Float | How suitable the claim is for controlled rewriting |
| diversity_bonus | Float | Reward for selecting underrepresented claim types |
| target_span | String | Original proposition span targeted for rewriting |
| generation_model | String | Model used for rewriting (default: CohereLabs/aya-expanse-8b) |
| regeneration_attempts | Int | Number of regeneration attempts before acceptance or drop |
| verification_result | JSON | Multi-stage verification outcome (claim integrity, surface quality, semantic quality) |

### Splits

| Split | Percentage | Purpose |
|-------|-----------|---------|
| Train | 70% | Model training |
| Dev | 10% | Hyperparameter tuning |
| Test | 20% | Held-out evaluation |

All splits are stratified by perturbation type and difficulty level. The real-world test set (Phase 6) is separate and never used for training or tuning.

### Deliverables

- Final dataset CSV with all metadata fields
- Split files: `train.csv`, `dev.csv`, `test.csv`
- Data card documenting schema, splits, and metadata generation process

---

## Phase 4: Classical + Bengali PLM Baselines (Week 4)

Establish baselines using classical methods and Bengali-specific pretrained language models. These baselines answer RQ2: do Bengali-specific models outperform multilingual ones on financial text?

### Classical Baselines

| Model | Features | Classifier |
|-------|----------|------------|
| TF-IDF + LogReg | Unigram/bigram TF-IDF, max 10K features | Logistic Regression (C=1.0) |
| FastText | Subword n-grams (3-5), 100-dim embeddings | FastText classifier |
| SVM | TF-IDF features | Linear SVM (C=1.0) |

### Bengali PLMs

| Model | Params | Fine-tuning |
|-------|--------|-------------|
| BanglaBERT (`csebuetnlp/banglabert`) | 110M | Full fine-tuning, lr=2e-5, 3 epochs |
| Bengali Electra | 110M | Full fine-tuning, lr=2e-5, 3 epochs |

### Training Setup

- Optimizer: AdamW, weight decay 0.01
- Batch size: 16 (PLMs), 128 (classical)
- Max sequence length: 512 tokens
- Early stopping: patience 3 on dev macro-F1
- Hardware: Kaggle T4 or institutional GPU

### Deliverables

- Results table: accuracy, macro-F1, per-perturbation F1
- Training logs and hyperparameter configurations
- Per-model confusion matrices

---

## Phase 5: Multilingual + LLM Evaluation (Week 5)

Extend the evaluation to multilingual models and large language models. This phase addresses RQ1 (which perturbation types are hardest), RQ2 (multilingual vs. Bengali-specific), and RQ3 (open-weight LLMs vs. fine-tuned smaller models).

### Multilingual PLMs

| Model | Params | Fine-tuning |
|-------|--------|-------------|
| mBERT | 110M | Full fine-tuning |
| XLM-RoBERTa-base | 270M | Full fine-tuning |
| ModernBERT multilingual | 395M | Full fine-tuning |

### Open-Weight LLMs

| Model | Params | Method |
|-------|--------|--------|
| Llama-3-8B | 8B | LoRA (rank 16, alpha 32) |
| Qwen-2.5-7B | 7B | LoRA (rank 16, alpha 32) |
| Gemma-2-9B | 9B | LoRA (rank 16, alpha 32) |
| Mistral-7B | 7B | LoRA (rank 16, alpha 32) |

### Closed LLMs (Zero-Shot Only)

| Model | Access | Setting |
|-------|--------|---------|
| GPT-4o | API | Zero-shot classification |
| Gemini-1.5-Pro | API | Zero-shot classification |
| Claude-3.5-Sonnet | API | Zero-shot classification |

No fine-tuning or few-shot prompting for closed models. The task instruction alone is provided.

### Deliverables

- Full comparison table across all model families
- Per-perturbation-type performance breakdown
- Statistical significance tests (McNemar's) for pairwise model comparisons

---

## Phase 6: Real-World Validation (Week 6)

Collect actual Bengali financial misinformation to test whether models trained on synthetic perturbations generalize to real-world data. This phase addresses RQ4.

### Sources

| Source | Expected Volume | Type |
|--------|----------------|------|
| Facebook financial groups | 50-100 | Viral posts, rumors |
| Bangladeshi fact-checking organizations | 50-100 | Verified misinformation |
| Bangladesh Bank press releases | 20-50 | Official corrections to rumors |
| DSE stock market rumors | 50-100 | Market manipulation claims |

### Protocol

- Collect 200-500 real examples with verified labels
- Each example includes: original text, label (misleading/reliable), source, verification method
- Test-only: no training or tuning on real-world data
- Compare model performance on synthetic vs. real-world test sets

### Deliverables

- Real-world test set CSV with metadata
- Synthetic-to-real domain gap analysis
- Per-model performance comparison (synthetic vs. real)

---

## Phase 7: Error Analysis (Week 7)

Deep analysis of what models get wrong and why. This phase addresses RQ5: what systematic errors do models make across perturbation types?

### Failure Categories

| Category | What to Examine |
|----------|----------------|
| Numerical failures | Does the model miss changed digits? Does it attend to numerical tokens at all? |
| Entity failures | Does the model recognize swapped company names, bank names, or financial terms? |
| Causal failures | Does the model track reversed cause-effect relationships? |
| Sentiment failures | Does the model distinguish genuine from amplified or flipped sentiment? |

### Analysis Methods

- Confusion matrices per perturbation type and difficulty level
- Attention visualization for transformer models on failed samples
- Feature importance analysis for classical models (top TF-IDF features by class)
- Error rate by text length, perturbation intensity, and annotation consensus level

### Deliverables

- Error analysis report with representative examples per failure category
- Confusion matrices and error distribution plots
- Feature analysis summary for classical baselines

---

## Phase 8: Ablations + Visualizations (Week 8)

Systematic ablation studies to understand what contributes to model performance. All results accompanied by statistical significance tests.

### Ablation Studies

| Ablation | Question |
|----------|----------|
| Perturbation-wise | Which perturbation types are individually hardest? |
| Difficulty-wise | How does performance vary across Easy/Medium/Hard? |
| Reasoning-wise | Which reasoning types (numerical, causal, etc.) are most challenging? |
| Model scale | Does larger model size consistently improve performance? |
| Language specificity | Do Bengali-specific models outperform multilingual ones? |

### Statistical Tests

- McNemar's test for pairwise model comparisons
- Bootstrap confidence intervals for all metrics
- Paired t-tests for ablation comparisons
- Bonferroni correction for multiple comparisons

### Visualizations

- Model performance heatmap (model x perturbation type)
- Difficulty distribution chart
- Error flow diagram (which failures cascade)
- Radar chart per model showing strengths across reasoning types

### Deliverables

- Ablation results tables
- All figures in publication-ready format (PDF vector graphics)
- Statistical significance summary

---

## Phase 9: Dataset Release Prep (Week 9)

Prepare all artifacts for public release. The dataset, code, and evaluation scripts must be reproducible by anyone with the listed dependencies.

### Release Components

| Component | Platform | Details |
|-----------|----------|---------|
| GitHub repository | github.com | Full code, configs, training scripts, evaluation scripts |
| Hugging Face dataset card | huggingface.co/datasets | Dataset with documentation, schema, intended use, limitations |
| Zenodo archive | zenodo.org | DOI-stamped snapshot for citation stability |
| Evaluation scripts | GitHub | Scripts to reproduce all results in the paper |
| Leaderboard template | GitHub | Template for community contributions |

### Dataset Card Sections

- Dataset summary
- Supported tasks and metrics
- Data instances (examples)
- Data fields with types
- Dataset splits with sizes
- Licensing and intended use
- Citation and limitations

### Deliverables

- Public GitHub repository with README, license, and reproducibility instructions
- Hugging Face dataset page with card
- Zenodo DOI
- Evaluation script test run confirming all results reproduce

---

## Phase 10: Paper Writing (Week 10)

Write the complete paper draft. Target venue: FinNLP Workshop at COLING 2026. Typical workshop paper length: 4-8 pages (excluding references).

### Sections

| Section | Key Content | Approx. Length |
|---------|-------------|----------------|
| Introduction | Problem statement, motivation, contributions | 0.5 page |
| Related Work | Bengali NLP, financial misinformation, benchmark datasets | 1 page |
| FinFact-BD | Dataset overview, perturbation types, research questions | 0.5 page |
| Dataset Construction | BENI v2, perturbation pipeline, quality filtering | 1 page |
| Human Validation | Protocol, annotator details, agreement metrics | 0.5 page |
| Benchmark Models | Classical, Bengali PLM, multilingual PLM, LLMs | 0.5 page |
| Experimental Results | Main results table, per-perturbation breakdown | 1 page |
| Error Analysis | Failure categories, representative examples | 0.5 page |
| Limitations | Synthetic data, language coverage, annotation bias | 0.25 page |
| Ethics | Potential misuse, mitigation strategies | 0.25 page |
| Conclusion | Summary, future work | 0.25 page |

### Deliverables

- Complete paper draft (LaTeX)
- All tables and figures embedded
- Reference list (40-60 citations)

---

## Phase 11: Revision (Week 11)

Incorporate feedback and finalize all reproducibility artifacts.

### Tasks

| Task | Details |
|------|---------|
| Internal feedback | Share draft with advisor and 1-2 colleagues |
| Revise content | Address all factual, methodological, and writing feedback |
| Proofread | Grammar, spelling, consistency check |
| Finalize reproducibility | Confirm all code runs, all results match paper |
| Prepare supplementary | Appendix with additional tables, hyperparameter details |

### Deliverables

- Revised paper draft
- Final reproducibility package (code + data + instructions)
- Supplementary materials

---

## Phase 12: Submission (Week 12)

Submit to FinNLP Workshop at COLING 2026.

### Tasks

| Task | Details |
|------|---------|
| Final formatting | Follow FinNLP style guidelines exactly |
| Submit via OpenReview | Or workshop submission system |
| Preprint decision | Upload to arXiv if venue allows and timing is appropriate |
| Artifact submission | Submit dataset + code as companion artifacts |

### Deliverables

- Submitted paper PDF
- Preprint (if applicable)
- Dataset and code publicly available

---

## Key Deadlines

| Milestone | Week | Target Date |
|-----------|------|-------------|
| Quality filtering & rewriting complete | 1 | Week 1 |
| Human validation complete | 2 | Week 2 |
| Benchmark construction complete | 3 | Week 3 |
| Classical + Bengali PLM baselines | 4 | Week 4 |
| Multilingual + LLM evaluation | 5 | Week 5 |
| Real-world validation complete | 6 | Week 6 |
| Error analysis complete | 7 | Week 7 |
| Ablations + visualizations complete | 8 | Week 8 |
| Dataset release prep complete | 9 | Week 9 |
| Paper draft complete | 10 | Week 10 |
| Revisions complete | 11 | Week 11 |
| **FinNLP submission** | **12** | **Week 12** |

---

## Contingency Plans

### If delayed by 2 weeks

- Skip ablation studies (Phase 8), fold key insights into error analysis
- Reduce open-weight LLM evaluation to one model (Llama-3-8B)
- Submit to arXiv as workshop paper instead

### If delayed by 4 weeks

- Drop real-world validation (Phase 6), acknowledge as limitation
- Reduce to classical + Bengali PLM baselines only (no multilingual or LLM evaluation)
- Target arXiv preprint instead of workshop submission

### If human validation fails (low agreement)

- Increase sample size to 500 and re-annotate
- Add a adjudication round where disagreements are resolved by discussion
- If still low, report disagreement as a finding rather than treating it as noise

---

## Dependencies

| Dependency | Phase | Risk | Mitigation |
|------------|-------|------|------------|
| Kaggle GPU access | 1, 4, 5 | Queue times, session limits | Batch runs, save checkpoints |
| Kaggle GPU for Bangla generation model (T4) | 1 | ~10-15 min generation pass, queue delays | Save model checkpoints, batch generation in chunks |
| Bengali annotators | 2 | Availability, quality | Recruit 4 annotators, drop 1 if needed |
| API access (GPT, Gemini, Claude) | 5 | Cost, rate limits | Budget $100, batch requests |
| Real-world data collection | 6 | Scarcity, legal issues | Start collection in Week 1, pivot to public datasets if needed |
| COLING 2026 deadline | 12 | No flexibility | Build 1-week buffer into Phase 11 |
| Advisor feedback | 11 | Delayed review | Share early, schedule check-ins in Week 8 |

---

*Last updated: July 14, 2026*
