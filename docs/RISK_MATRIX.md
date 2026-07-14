# Risk Matrix: FinFact-BD Benchmark

## Overview

This document identifies, categorizes, and plans mitigation for risks specific to the FinFact-BD benchmark validation pipeline. The benchmark centers on human validation, real-world testing, and rigorous model evaluation for Bengali financial misinformation detection, targeting the FinNLP Workshop at COLING 2026.

---

## Risk Assessment Summary

| Risk Level | Count | Description |
|------------|-------|-------------|
| 🔴 High | 4 | Could invalidate contribution or block submission |
| 🟡 Medium | 10 | Could weaken results or delay timeline |
| 🟢 Low | 6 | Minor issues, manageable with planning |

---

## Human Validation Risks

### H1: Low Inter-Annotator Agreement

**Likelihood**: Medium
**Impact**: High

**Description**: Three Bengali-native annotators may disagree substantially on whether articles are reliable or misleading. Low Fleiss' kappa undermines the gold-standard claims and weakens the benchmark's validity.

**Mitigation**:
1. Provide detailed annotation guidelines with examples for each perturbation type
2. Run a pilot round on 50 samples before full annotation to calibrate annotators
3. Hold calibration meetings after pilot to resolve disagreements
4. Stratify samples across perturbation types to ensure balanced coverage

**Contingency**: If kappa falls below 0.6, add two more annotators and use majority voting. If still low, report the low agreement as a finding itself (disagreement as a feature of the task, not a bug).

---

### H2: Annotator Fatigue and Bias

**Likelihood**: Medium
**Impact**: Medium

**Description**: Annotating 300 Bengali financial articles is tedious. Fatigue leads to careless labels, inconsistent decisions, and degradation over time. Annotators may also develop systematic biases toward certain perturbation types.

**Mitigation**:
1. Break annotation into 5 sessions of 60 articles each, spread across multiple days
2. Randomize sample order across annotators to prevent position effects
3. Include 20 gold-standard anchor samples (pre-labeled) to detect drift
4. Provide rest breaks and compensation aligned with fair wages

**Contingency**: If fatigue indicators appear (error rate on gold anchors exceeds 15%), pause annotation, recalibrate, and resume with fresh sessions.

---

### H3: Ambiguous Samples

**Likelihood**: High
**Impact**: Medium

**Description**: Some perturbations produce text that is genuinely ambiguous. A numerical change from ৫০০ to ৫৫০ might be subtle enough that even finance experts disagree on whether it constitutes misinformation. These edge cases complicate labeling.

**Mitigation**:
1. Allow a "borderline" label during annotation (later resolved by majority vote)
2. Track which perturbation types generate the most ambiguous cases
3. Include ambiguous samples in the error analysis as a separate category
4. Report inter-annotator agreement per perturbation type, not just overall

**Contingency**: If more than 30% of samples are borderline, reduce the ambiguity threshold in the perturbation pipeline and regenerate those samples.

---

### H4: Time-Consuming Process

**Likelihood**: High
**Impact**: Medium

**Description**: Three annotators reading 300 Bengali financial articles with justification takes significant time. Estimates range from 3 to 5 weeks for the full annotation round, which threatens the COLING 2026 timeline.

**Mitigation**:
1. Start annotation as early as possible, before dataset generation completes
2. Use a shared Google Sheet or Label Studio for real-time tracking
3. Pre-filter obviously clear samples to reduce annotator workload
4. Set weekly milestones (75 articles per week per annotator)

**Contingency**: If annotation falls behind, reduce to 200 samples (maintaining stratification) and note the reduced sample size as a limitation.

---

## Real-World Data Risks

### RW1: Difficulty Collecting Sufficient Examples

**Likelihood**: Medium
**Impact**: High

**Description**: The real-world test set requires actual Bengali financial misinformation, not synthetically perturbed text. Finding 200+ verified examples of genuine Bengali financial misinformation is non-trivial. Fact-checking databases for Bengali finance are sparse.

**Mitigation**:
1. Source from Bangladeshi fact-checking organizations (e.g., BD Fact Check)
2. Scrape financial news portals and cross-reference with Bangladesh Bank advisories
3. Use social media posts flagged by community notes or reports
4. Partner with local journalism networks for curated examples

**Contingency**: If fewer than 100 verified examples are collected, supplement with machine-translated English financial misinformation (clearly labeled as such) and report results separately.

---

### RW2: Legal and Ethical Concerns

**Likelihood**: Low
**Impact**: High

**Description**: Collecting real misinformation raises legal questions about defamation, copyright, and responsible disclosure. Publishing misinformation examples, even for research, requires careful handling.

**Mitigation**:
1. Obtain IRB approval before collecting real-world examples
2. Anonymize source attributions where possible
3. Include a responsible use statement in the dataset card
4. Consult with the university legal office on data collection scope

**Contingency**: If legal review raises concerns, restrict the real-world test set to paraphrased examples (original content transformed but meaning preserved).

---

### RW3: Source Reliability

**Likelihood**: Medium
**Impact**: Medium

**Description**: The ground truth labels for real-world examples depend on source reliability. A fact-checking site might label something as false that is actually disputed. Source quality varies across Bengali media.

**Mitigation**:
1. Require at least two independent sources confirming each example's label
2. Prioritize established fact-checking organizations over social media
3. Document the provenance of every real-world example
4. Flag disputed cases and exclude them from evaluation

**Contingency**: If source reliability is questionable for a subset, run evaluation on verified-only and disputed subsets separately.

---

### RW4: Class Imbalance in Real Data

**Likelihood**: High
**Impact**: Medium

**Description**: Genuine Bengali financial misinformation skews heavily toward certain categories (stock tips, loan scams, crypto schemes). The real-world test set may not cover the full range of perturbation types, making cross-perturbation analysis impossible.

**Mitigation**:
1. Stratify collection by misinformation type, not just availability
2. Accept lower counts per category rather than inflating popular ones
3. Report per-category results with confidence intervals
4. Use stratified sampling for the final test set

**Contingency**: If certain categories have fewer than 20 examples, merge them into a "miscellaneous" category and note this as a limitation.

---

## Model Evaluation Risks

### E1: LLM API Costs

**Likelihood**: Medium
**Impact**: Medium

**Description**: Evaluating GPT-4o, Gemini-1.5-Pro, and Claude-3.5-Sonnet on 4,000 test samples each costs significant money. At roughly $0.01 per query, three models across 4,000 samples is $120 per evaluation run, and multiple runs for significance testing multiply this.

**Mitigation**:
1. Run closed LLM evaluation once on the held-out test set (not per fold)
2. Use batch API endpoints where available (discounted rates)
3. Cache all API responses for reproducibility
4. Set a hard budget cap of $200 for closed LLM evaluation

**Contingency**: If costs exceed budget, evaluate only GPT-4o as the closed LLM representative and note the reduced scope.

---

### E2: Compute Limitations

**Likelihood**: Medium
**Impact**: Medium

**Description**: Fine-tuning four 7-9B parameter LLMs with LoRA requires substantial GPU memory. Kaggle T4 GPUs have 16GB VRAM and 30-hour weekly limits. Multiple models and hyperparameter searches may exceed available compute.

**Mitigation**:
1. Use LoRA with rank 16 to keep memory footprint manageable
2. Schedule fine-tuning across multiple Kaggle sessions
3. Use gradient checkpointing to reduce memory usage
4. Pre-test on a small validation set before full fine-tuning

**Contingency**: If GPU access is constrained, reduce the LLM evaluation to two models (Llama-3-8B and one other) and note compute limitations as a constraint.

---

### E3: Reproducibility Issues

**Likelihood**: Medium
**Impact**: High

**Description**: Model results vary across runs due to random seeds, floating-point non-determinism, and hardware differences. Reviewers expect reproducible numbers, and inconsistent results across runs undermine credibility.

**Mitigation**:
1. Fix random seeds for all training runs (42, 123, 456 for three runs)
2. Report mean and standard deviation across runs, not single numbers
3. Use deterministic algorithms where available (CUBLAS deterministic mode)
4. Save and publish all model checkpoints

**Contingency**: If variance is high (std > 2%), run additional seeds and report the wider confidence interval. Acknowledge sources of non-determinism explicitly.

---

### E4: Benchmark Saturation

**Likelihood**: Low
**Impact**: High

**Description**: If all models score above 90% accuracy, the benchmark lacks discriminative power. The perturbations may be too obvious for modern multilingual models, making the benchmark less useful for future research.

**Mitigation**:
1. Monitor accuracy during pilot runs and adjust perturbation difficulty if needed
2. Include the hardest perturbation types (causal, entity) which are less trivial
3. Report per-perturbation-type results to show where models still struggle
4. Frame the benchmark as revealing specific weaknesses, not just overall scores

**Contingency**: If all models exceed 95% accuracy, add harder perturbation variants (nested numerical changes, multi-entity replacements) to the test set.

---

## Generation Pipeline Risks

### G1: Generation Model Quality

**Likelihood**: Medium
**Impact**: High

**Description**: The Bangla generation model (banglat5 or BanglaByT5) may produce rewrites that are too subtle, too generic, or grammatically incorrect. The model may not reliably rewrite only the target sentence while preserving the rest of the article.

**Mitigation**:
1. Start with the smaller, cleaner model (banglat5) and evaluate rewrite quality on a pilot set
2. If quality is insufficient, escalate to a stronger instruction model
3. Run multi-stage verification to catch low-quality rewrites
4. Regenerate failed samples (up to 3 attempts)

**Contingency**: If generation quality is consistently poor, fall back to the rule-based perturbation pipeline (already implemented and producing 20K samples).

---

### G2: Generation Hallucination

**Likelihood**: Medium
**Impact**: High

**Description**: The generator may introduce facts not present in the original article, change more than the target claim, or invent new entities/numbers. This breaks the controlled nature of the benchmark.

**Mitigation**:
1. Constrain the generator to rewrite only the target sentence or a single local paragraph
2. Run claim integrity verification (Stage 1) to check that only the intended claim changed
3. Run span-locality verification to ensure the change is localized
4. Log and discard samples where hallucination is detected

**Contingency**: If hallucination rate exceeds 30%, tighten the generation prompt and reduce the allowed rewrite scope.

---

### G3: Regeneration Loop Failure

**Likelihood**: Medium
**Impact**: Medium

**Description**: The regeneration loop (up to 3 attempts) may not produce a passing sample for all candidates. High failure rates reduce the final dataset size.

**Mitigation**:
1. Track failure reasons (claim integrity, surface quality, semantic quality)
2. Adjust verification thresholds if they are too strict
3. Log all failure reasons for analysis
4. Accept that some samples will be dropped — this is a feature, not a bug

**Contingency**: If more than 20% of samples fail regeneration, review and loosen verification thresholds or expand the regeneration budget.

---

### G4: Model Resource Constraints on Kaggle

**Likelihood**: Medium
**Impact**: Medium

**Description**: Loading a Bangla generation model on Kaggle T4 (16GB VRAM) may be constrained. The model + input processing may exceed memory or time limits.

**Mitigation**:
1. Use the smaller banglat5 model first
2. Process articles in batches
3. Use gradient checkpointing if needed
4. Save intermediate results to avoid reprocessing

**Contingency**: If T4 is insufficient, process in smaller batches across multiple Kaggle sessions.

---

## Paper Submission Risks

### P1: Rejection from FinNLP

**Likelihood**: Medium
**Impact**: High

**Description**: FinNLP Workshop at COLING 2026 is competitive. The paper may be rejected for insufficient novelty, weak methodology, or perceived low contribution.

**Mitigation**:
1. Emphasize the benchmark design (research questions, human validation, real-world testing) over dataset size
2. Include deep error analysis that reveals actionable insights
3. Ensure all baselines are properly implemented and fairly compared
4. Get feedback from colleagues before submission

**Contingency**: If rejected from FinNLP, revise and submit to ACL Findings, EMNLP Findings, or TALLIP journal.

---

### P2: Missing the Deadline

**Likelihood**: Low-Medium
**Impact**: High

**Description**: COLING 2026 workshop deadlines are firm. If human validation, evaluation, or writing takes longer than expected, the submission window closes.

**Mitigation**:
1. Work backward from the deadline with weekly milestones
2. Start writing the paper in parallel with experiments, not after
3. Prepare figures and tables incrementally
4. Have a "minimum viable paper" ready two weeks before the deadline

**Contingency**: If the deadline is missed, target the next available workshop or conference. Use the extra time to strengthen the contribution.

---

### P3: Insufficient Novelty Claim

**Likelihood**: Medium
**Impact**: Medium

**Description**: Reviewers may argue that creating a benchmark for a new language/domain is not sufficiently novel. The contribution must go beyond "we made a dataset."

**Mitigation**:
1. Frame contributions around research questions, not dataset creation
2. Emphasize the human validation and real-world testing components
3. Include error analysis that reveals model limitations
4. Position the benchmark as an evaluation framework, not just data

**Contingency**: If novelty is questioned, add an analysis section showing how existing multilingual benchmarks fail to capture Bengali financial nuances.

---

### P4: Reviewer Criticism of Methodology

**Likelihood**: Medium
**Impact**: Medium

**Description**: Reviewers may criticize the perturbation-based approach (synthetic vs. natural), the mDeBERTa filtering, or the evaluation protocol. Methodological weaknesses can sink a workshop paper.

**Mitigation**:
1. Acknowledge limitations explicitly in the paper
2. Include ablation studies where feasible
3. Compare mDeBERTa-filtered labels to human labels in the validation section
4. Use standard evaluation protocols (stratified splits, McNemar's test)

**Contingency**: If methodology is criticized, address in the camera-ready with additional analysis or clarification.

---

## Technical Risks

### T1: Data Leakage Between Splits

**Likelihood**: Low
**Impact**: High

**Description**: If the same original article appears in both train and test splits (or if perturbations from the same article cross splits), evaluation results are inflated. Data leakage is a common benchmark failure mode.

**Mitigation**:
1. Split at the original article level, not the perturbation level
2. Verify no original_id appears in both train and test
3. Run a leakage audit script before any training
4. Log the split statistics (unique originals per split)

**Contingency**: If leakage is detected, regenerate the split at the article level and re-run all experiments.

---

### T2: Overfitting to Synthetic Data

**Likelihood**: Medium
**Impact**: High

**Description**: Models trained on synthetic perturbations may learn artifacts of the perturbation pipeline rather than genuine misinformation detection. This would make the benchmark results misleading.

**Mitigation**:
1. Evaluate on the real-world test set as a separate check
2. Report both synthetic and real-world test set results
3. Compare model behavior across perturbation types for consistency
4. Include a discussion of ecological validity in the paper

**Contingency**: If models perform well on synthetic but poorly on real-world data, frame this as a key finding rather than a failure.

---

### T3: Model Hallucination in Evaluation

**Likelihood**: Medium
**Impact**: Medium

**Description**: When evaluating LLMs zero-shot, the models may generate explanations that hallucinate facts or misinterpret Bengali text. This complicates the error analysis.

**Mitigation**:
1. Evaluate only on the classification output, not explanations
2. Use structured prompts that require binary output
3. Validate a sample of LLM outputs manually
4. Report classification metrics only, not generation quality

**Contingency**: If hallucination rates are high on explanations, drop the explanation analysis and focus on classification performance only.

---

### T4: Reproducibility Failures

**Likelihood**: Low-Medium
**Impact**: High

**Description**: Reviewers or other researchers may fail to reproduce the results due to missing dependencies, undocumented hyperparameters, or hardware-specific behavior.

**Mitigation**:
1. Publish all code, configs, and trained models
2. Include a full requirements.txt with pinned versions
3. Document every hyperparameter in the paper's appendix
4. Test the full pipeline on a clean environment before release

**Contingency**: If reproducibility issues are reported post-submission, release a patched version and update the GitHub repository.

---

## Risk Monitoring

### Weekly Risk Review

Every Friday, review:

1. **Annotation progress**: Are annotators on track? Any fatigue indicators?
2. **Data collection**: How many real-world examples confirmed? Any legal blockers?
3. **Experiment results**: Any models significantly underperforming? Compute budget status?
4. **Timeline**: Are we on track for the submission deadline?

### Risk Dashboard

| Category | Active Risks | Status | Next Review |
|----------|-------------|--------|-------------|
| Human Validation | H1-H4 | Monitoring | Weekly |
| Real-World Data | RW1-RW4 | Monitoring | Weekly |
| Model Evaluation | E1-E4 | Monitoring | Weekly |
| Generation Pipeline | G1-G4 | Planning | Weekly |
| Paper Submission | P1-P4 | Planning | Bi-weekly |
| Technical | T1-T4 | Monitoring | Weekly |

### Escalation Triggers

Escalate to supervisor/PI if:

1. Inter-annotator kappa drops below 0.6 after calibration
2. Real-world data collection falls below 100 verified examples by week 4
3. Compute costs exceed $300 total
4. Any model fails to converge after 3 hyperparameter attempts
5. The deadline is at risk (paper not draft-complete 2 weeks before submission)

---

## Escalation Procedures

### Level 1: Self-Resolution

**Scope**: Minor issues that can be fixed within a day.
**Examples**: GPU scheduling conflicts, annotation formatting issues, minor code bugs.
**Action**: Fix directly, log in risk dashboard.

### Level 2: Peer Consultation

**Scope**: Issues requiring input from labmates or collaborators.
**Examples**: Annotation disagreement patterns, methodology questions, ambiguous samples.
**Action**: Discuss in lab meeting or with a specific colleague, document the resolution.

### Level 3: Supervisor Escalation

**Scope**: Issues affecting the timeline or contribution significantly.
**Examples**: Low inter-annotator agreement, insufficient real-world data, compute budget exceeded.
**Action**: Schedule a meeting with supervisor, present options and tradeoffs, get guidance.

### Level 4: Pivot Decision

**Scope**: Fundamental issues that threaten the paper's viability.
**Examples**: All models saturate the benchmark, legal review blocks real-world data, novelty claim rejected.
**Action**: Full discussion with supervisor, consider scope reduction, venue change, or pivot to a different contribution angle.

---

## Contingency Plans

### Plan A: Full Execution (Default)

- 20K dataset, human validation on 300 samples, real-world test set
- All baseline models evaluated, deep error analysis
- Target: FinNLP Workshop at COLING 2026

### Plan B: Reduced Scope

- 10K dataset, human validation on 200 samples
- Core baselines only (mDeBERTa, BanglaBERT, XLM-R, one LLM)
- Target: Workshop paper with narrower focus

### Plan C: Analysis-First

- If data collection stalls, pivot to error analysis of existing models
- Use public Bengali NLP benchmarks for comparison
- Target: Short paper or poster at a regional workshop

### Plan D: Extended Timeline

- If COLING 2026 deadline is missed, use extra time to strengthen the contribution
- Target: ACL or EMNLP 2027, or TALLIP journal

---

*Last updated: July 14, 2026*
*Status: Benchmark validation phase — planning human annotation and real-world data collection*
