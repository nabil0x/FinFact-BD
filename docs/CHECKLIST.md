# Benchmark Validation Checklist

**Project:** FinFact-BD  
**Target:** FinNLP Workshop, COLING 2026  
**Timeline:** 12 weeks to submission  
**Last updated:** July 14, 2026  

This checklist tracks progress toward a validated benchmark for Bengali financial misinformation detection. Human validation, real-world testing, and rigorous evaluation take priority over dataset size or novelty claims. Each phase has a clear definition of done. Do not advance to the next phase until its quality gate passes.

---

## Phase 0: Kaggle Rewrite Pipeline (Week 0)

Generate controlled Bangla rewrites using planning-guided claim generation instead of rule-based token replacement. The planner decides what claim changes, how it changes, what must remain unchanged, and which checks must pass. The generator realizes the planned local rewrite. The verifier decides whether the result is acceptable.

- [ ] Implement claim extraction (split articles into sentences, identify candidate propositions)
- [ ] Implement claim selection scoring (importance_score, editability_score, diversity_bonus)
- [ ] Implement claim planning (structured metadata: sample_id, rewrite_family, target_sentence_index, target_span, desired_change, difficulty)
- [ ] Load role-specific models (Qwen extraction/planning, Aya rewriting, e5 similarity, mDeBERTa NLI, BanglaBERT language quality)
- [ ] Generate rewrites (target sentence/paragraph only, not full article)
- [ ] Run multi-stage verification:
  - [ ] Stage 1: Claim integrity (did intended claim change? any extra changes? span-locality)
  - [ ] Stage 2: Surface quality (fluent Bangla? journalistic style? not too close to source?)
  - [ ] Stage 3: Semantic quality (believable news? contradiction score? duplicate detection?)
- [ ] Implement regeneration loop (up to 3 attempts, keep best candidate)
- [ ] Export raw, filtered, and metadata files with full provenance

**Definition of Done:** Generated dataset exists with claim planning metadata, verification results, and full provenance for every accepted sample.

**Quality Gate:** Multi-stage verification passes for accepted samples. Regeneration failure rate below 15%. Every exported row traces back to source article, target claim, selection scores, generator model, and verification result.

---

## Phase 1: Quality Filtering (Week 1)

The planning-guided rewriting pipeline (Phase 0) includes built-in multi-stage verification as its primary quality control. mDeBERTa filtering is an additional downstream gate that catches samples the built-in verifier missed.

- [ ] Run mDeBERTa-v3 multilingual NLI filtering on Kaggle
- [ ] Compute filtering statistics (pass rate, contradiction score distribution)
- [ ] Review a random sample of removed examples to check for false positives
- [ ] Document overlap between built-in verification failures and mDeBERTa rejections
- [ ] Document a quality report summarizing what was kept, what was removed, and why

**Definition of Done:** Filtered dataset exists with documented statistics and a quality report. The report explains how much value the mDeBERTa gate adds beyond the built-in verifier. No downstream work starts until this is complete.

**Quality Gate:** Quality report reviewed. False positive rate on reviewed samples below 5%.

---

## Phase 2: Human Validation (Weeks 2-3) — CRITICAL

This is the most important phase. Everything else depends on having trustworthy human labels.

### Recruitment and Setup
- [ ] Select 300 stratified samples (150 original, 150 perturbed) across all five perturbation types
- [ ] Recruit 3 Bengali-native annotators with finance domain familiarity
- [ ] Create annotation guidelines with clear definitions of `original`, `perturbed`, and `not sure`
- [ ] Define the annotation task: claim-centric `original` / `perturbed` / `not sure` labels plus free-text justification and required confidence

### Pilot Study
- [ ] Run a pilot study on 30 samples with all 3 annotators
- [ ] Review pilot results and refine guidelines based on disagreements
- [ ] Conduct a calibration meeting with annotators to align on edge cases

### Full Annotation
- [ ] Annotate the remaining 270 samples
- [ ] Measure inter-annotator agreement using Fleiss' kappa
- [ ] If kappa < 0.6, revise guidelines and re-annotate a subset
- [ ] Resolve all disagreements through discussion or majority vote
- [ ] Finalize annotations with adjudicated labels

### Analysis
- [ ] Report Fleiss' kappa in the paper
- [ ] Compare human labels against mDeBERTa filter labels to calibrate the automated pipeline
- [ ] Identify which perturbation types humans find hardest to detect

**Definition of Done:** 300 samples annotated by 3 people. Fleiss' kappa reported. Disagreements resolved. Human labels available as gold standard.

**Quality Gate:** Fleiss' kappa >= 0.6. Annotation guidelines documented. Adjudicated labels finalized.

---

## Phase 3: Benchmark Construction (Week 4)

Turn the filtered, human-validated dataset into a proper benchmark with metadata and splits.

- [ ] Add rich metadata to all samples (source, article length, domain, date range)
- [ ] Assign difficulty levels (Easy / Medium / Hard) based on human annotation agreement
- [ ] Assign reasoning labels (numerical, policy, temporal, entity, causal) to each perturbation
- [ ] Include claim planning metadata (importance_score, editability_score, diversity_bonus, target_span, etc.)
- [ ] Include generation model info and regeneration attempts for each sample
- [ ] Include verification result JSON (claim integrity, surface quality, semantic quality scores)
- [ ] Create train / dev / test splits (80/10/10, stratified by perturbation type)
- [ ] Validate splits for data leakage (check original_id overlap between splits)
- [ ] Document the dataset schema and all field definitions

**Definition of Done:** Dataset with metadata, difficulty levels, reasoning labels, claim planning provenance, and validated splits. Schema documented.

**Quality Gate:** No data leakage between splits. All metadata fields populated. Schema review complete.

---

## Phase 4: Model Evaluation (Weeks 5-7)

Train, fine-tune, and evaluate all model families. This is the core experimental work.

### Classical Baselines
- [ ] Train TF-IDF + Logistic Regression
- [ ] Train FastText classifier
- [ ] Train SVM with TF-IDF features

### Bengali PLMs
- [ ] Fine-tune BanglaBERT (csebuetnlp/banglabert)
- [ ] Fine-tune Bengali Electra

### Multilingual PLMs
- [ ] Fine-tune mBERT
- [ ] Fine-tune XLM-RoBERTa-base
- [ ] Fine-tune ModernBERT

### Open-Weight LLMs
- [ ] Fine-tune Llama-3-8B with LoRA
- [ ] Fine-tune Qwen-2.5-7B with LoRA
- [ ] Fine-tune Gemma-2-9B with LoRA
- [ ] Fine-tune Mistral-7B with LoRA

### Closed LLMs (Zero-Shot Only)
- [ ] Evaluate GPT-4o zero-shot
- [ ] Evaluate Gemini-1.5-Pro zero-shot
- [ ] Evaluate Claude-3.5-Sonnet zero-shot

### Metrics
- [ ] Record accuracy and macro F1 for all models
- [ ] Record per-perturbation F1 (numerical fact change, policy reversal, entity replacement, temporal shift, causal inversion)
- [ ] Record precision and recall per class
- [ ] Generate confusion matrices for top models
- [ ] Run McNemar's test for pairwise model comparisons

**Definition of Done:** All models evaluated. Metrics recorded in a single results table. Confusion matrices generated.

**Quality Gate:** All models converge (no training failures). Results table complete with all metrics. Statistical significance tests run.

---

## Phase 5: Real-World Validation (Week 7-8)

Test whether synthetic perturbation performance predicts real-world detection.

- [ ] Collect 200-500 real Bengali financial misinformation examples
- [ ] Sources: Facebook posts, fact-checker archives, Bangladesh Bank notices, DSE disclosures
- [ ] Label all real examples with the same claim-centric scheme (`original` / `perturbed` / `not sure`)
- [ ] Test all models on the real-world set
- [ ] Compare synthetic vs real performance across models
- [ ] Analyze whether models that perform well on synthetic perturbations also perform well on real data

**Definition of Done:** Real-world test set collected and labeled. All models evaluated on it. Comparison analysis complete.

**Quality Gate:** At least 200 real examples collected. All examples labeled. Comparison analysis written.

---

## Phase 6: Error Analysis (Week 8-9)

Understand what models get wrong and why.

- [ ] Identify numerical failures (models missing digit perturbations)
- [ ] Identify entity failures (models not recognizing swapped company names)
- [ ] Identify causal failures (models missing reversed cause-effect)
- [ ] Identify policy reversal failures (models missing directional flips)
- [ ] Identify temporal failures (models missing time shifts)
- [ ] Generate feature analysis across error types
- [ ] Create an error taxonomy categorizing failure modes
- [ ] Analyze whether error patterns on synthetic data match real-world failures

**Definition of Done:** Error taxonomy documented. Feature analysis complete. Synthetic-to-real error comparison done.

**Quality Gate:** Error taxonomy reviewed. At least 200 error samples analyzed. Taxonomy covers all perturbation types.

---

## Phase 7: Ablations (Week 9)

Run controlled experiments to isolate what matters.

- [ ] Perturbation-wise analysis (which perturbation types are hardest overall)
- [ ] Difficulty-wise analysis (how does performance vary by difficulty level)
- [ ] Reasoning-wise analysis (which reasoning types are most challenging)
- [ ] Statistical significance tests across all ablation conditions

**Definition of Done:** Ablation results documented with significance tests.

**Quality Gate:** All ablation conditions tested. Results consistent with main findings.

---

## Phase 8: Visualizations (Week 9-10)

Create publication-ready figures.

- [ ] Confusion matrices for top 3 models
- [ ] Per-category performance bar charts
- [ ] Error distribution plots across perturbation types
- [ ] Model comparison radar chart or heatmap
- [ ] Synthetic vs real performance scatter plot

**Definition of Done:** All figures generated in publication format (PDF/SVG).

**Quality Gate:** Figures reviewed for clarity. All labels correct. Consistent color scheme across figures.

---

## Phase 9: Dataset Release (Week 10)

Prepare everything for public release.

- [ ] Create GitHub repository with clean structure
- [ ] Write Hugging Face dataset card with schema, splits, and usage instructions
- [ ] Archive dataset on Zenodo for DOI
- [ ] Package evaluation scripts with clear documentation
- [ ] Create a leaderboard template for future comparisons
- [ ] Write README with reproducibility instructions (hardware, seeds, dependencies)

**Definition of Done:** Dataset and code publicly available. Reproducibility instructions complete.

**Quality Gate:** Repository builds cleanly. Dataset loads without errors. Evaluation scripts run end-to-end.

---

## Phase 10: Paper Writing (Weeks 7-10, parallel)

Write the paper. Most sections can start once Phase 4 results are available.

- [ ] Introduction and Related Work
- [ ] FinFact-BD dataset description
- [ ] Dataset Construction methodology
- [ ] Human Validation section with Fleiss' kappa
- [ ] Benchmark Models description
- [ ] Experimental Results with full metrics table
- [ ] Error Analysis with taxonomy
- [ ] Limitations (synthetic nature, domain specificity, language coverage)
- [ ] Ethics statement
- [ ] Conclusion and future work

**Definition of Done:** Complete draft ready for internal review.

**Quality Gate:** All sections written. All claims supported by results. No overclaiming.

---

## Phase 11: Revision (Week 11)

Incorporate feedback and polish.

- [ ] Circulate draft to internal reviewers (1-2 colleagues)
- [ ] Collect and address all feedback
- [ ] Revise based on reviewer comments
- [ ] Finalize reproducibility artifacts (code, data, instructions)
- [ ] Proofread entire paper for grammar and consistency

**Definition of Done:** Revised draft incorporating all feedback. Final proofread complete.

**Quality Gate:** All reviewer comments addressed. No outstanding issues.

---

## Phase 12: Submission (Week 12)

Submit and prepare for the workshop.

- [ ] Format paper to FinNLP / COLING template
- [ ] Verify page limit (6-8 pages)
- [ ] Ensure anonymous for review if double-blind
- [ ] Prepare supplementary materials
- [ ] Submit to FinNLP Workshop
- [ ] Upload preprint to arXiv if appropriate

**Definition of Done:** Paper submitted. Preprint uploaded if applicable.

**Quality Gate:** Submission confirmed. All artifacts (code, data, paper) publicly available or ready to release.

---

## Quality Gates Summary

Quality gates are checkpoints between phases. Do not advance until the gate passes.

| Gate | Between Phases | Criterion |
|------|---------------|-----------|
| G0 | Phase 0 → 1 | Multi-stage verification passes for accepted samples. Regeneration failure rate below 15%. Full provenance exported. |
| G1 | Phase 1 → 2 | Quality report reviewed. False positive rate < 5% on reviewed samples. |
| G2 | Phase 2 → 3 | Fleiss' kappa >= 0.6. Human labels finalized. |
| G3 | Phase 3 → 4 | No data leakage. All metadata populated. Schema documented. |
| G4 | Phase 4 → 5 | All models evaluated. Results table complete. Convergence verified. |
| G5 | Phase 5 → 6 | At least 200 real examples. All labeled. Comparison analysis done. |
| G6 | Phase 6 → 7 | Error taxonomy reviewed. 200+ error samples analyzed. |
| G7 | Phase 7 → 8 | All ablation conditions tested. Significance tests complete. |
| G8 | Phase 8 → 9 | Figures reviewed. Labels correct. Consistent styling. |
| G9 | Phase 9 → 10 | Dataset loads. Scripts run. README complete. |
| G10 | Phase 10 → 11 | Draft complete. All claims supported. No overclaiming. |
| G11 | Phase 11 → 12 | All feedback addressed. Final proofread done. |
| G12 | Post-submission | Artifacts released. Submission confirmed. |

---

## Anti-Slop Rules

These rules prevent common pitfalls in benchmark papers.

- **Do not claim "first" as the main contribution.** The contribution is the benchmark itself and what it reveals about model behavior.
- **Do not emphasize dataset size.** 10K samples with human validation beats 100K without it.
- **Do not overclaim novelty.** Be specific about what is new: the perturbation types, the human validation protocol, the real-world test set.
- **Do not skip the real-world validation.** Synthetic benchmarks without ecological validity are weak. The comparison between synthetic and real performance is a key contribution.
- **Do not ignore limitations.** Acknowledge that perturbations are synthetic, domain is specific, and language coverage is limited to Bengali.
- **Do not claim the generation model is novel.** The novelty is in the controlled planning and verification pipeline, not the generator itself.

---

*Checklist created: July 2026*  
*Status: Ready for execution*
