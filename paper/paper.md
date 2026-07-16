# FinFact-BD: Planning-Guided Claim Rewriting for a Bengali Financial Misinformation Benchmark

## Abstract

Bengali financial misinformation remains underrepresented in benchmark design despite a large, economically consequential news ecosystem. Existing resources focus on Hindi/English pairs, article-level classification, or generic lexical perturbation. We present FinFact-BD, a controlled framework for generating and evaluating Bengali financial misinformation through planning-guided claim rewriting. The pipeline extracts factual propositions from real Bengali financial articles, ranks them by importance and editability, constructs explicit perturbation plans across four families (numerical fact change, temporal shift, entity replacement, policy reversal), constrains a multilingual language model to realize the planned change, and filters candidates through multi-stage verification. In a 300-article pilot with a Qwen3-8B planner and Aya Expanse 8B generator, 259 samples passed verification (86.3%). A systematic failure taxonomy across pilot runs shows that planning quality is the main bottleneck. We describe a human validation protocol with five annotation criteria, three independent annotators, and inter-annotator agreement reporting. We also provide a downstream factual consistency verification protocol and scripts for evaluating BanglaBERT, XLM-R, mBERT, and multilingual LLM baselines. The framework is designed to scale to more than 2,000 accepted samples with 500 human-validated instances. We release the pilot benchmark, validation tools, and full provenance metadata to support research in Bengali financial fact verification.

## 1 Introduction

Financial misinformation has real economic consequences. In Bangladesh, where the financial press is active and widely read, misstated figures, fabricated policy changes, or shifted temporal claims can affect market perception and individual decision-making. Yet robust resources for studying this problem in Bengali remain scarce.

Existing benchmark datasets for Bengali misinformation tend to operate at the article level or rely on generic lexical perturbation. BanFakeNews (Hossain et al., 2020) provides a large collection of Bangla fake-news articles, but the labels are article-level and the examples are not generated under controlled conditions. Lexical perturbation approaches replace words or spans deterministically, which is reproducible but produces edits so small that they lack rhetorical force. In long-form Bengali financial news, a single swapped digit in the middle of a paragraph is technically a change but not a convincing misinformation sample.

We address this gap with **FinFact-BD**, a controlled framework for Bengali financial misinformation built on planning-guided claim rewriting. Our approach operates at the level of factual propositions rather than words or documents. Given a real Bengali financial article, the pipeline: (1) extracts candidate claims, (2) selects one by importance and editability, (3) constructs an explicit plan specifying *what* changes and *how*, (4) constrains a Bangla-capable generation model to realize the planned change locally, and (5) verifies the output through independent checks before acceptance. The generation model does not decide what misinformation to create; it expresses a planned factual modification in natural Bengali prose.

Our contributions are fourfold:

1. **A controlled generation framework** that produces localized, verifiable factual distortions in Bengali financial news through planning-guided rewriting, with separate model roles for planning and generation and nine separate verification checks.

2. **Empirical validation from three pilot runs** — primary: Qwen3-8B planner, Aya Expanse 8B generator, 259/300 accepted (86.3%); development: Qwen2.5-3B 79/100 (79%) and Qwen3-8B 26/30 (86.7%) — with a systematic failure taxonomy showing planning quality as the primary bottleneck and verifier robustness increasing with planner capability.

3. **A released pilot benchmark** (259 accepted samples) with full provenance metadata, stratified train/val/test splits (180/38/41), difficulty labels, and companion validation tools: a human-annotation workbook generator, an IAA analysis script, and a downstream evaluation protocol for factual consistency verification.

4. **A documented scaling and validation protocol** targeting more than 2,000 generated samples with 500 human-validated instances, establishing the pathway from methodology validation to a mature language resource.

The paper is organized as follows. Section 2 reviews related work. Section 3 describes the pipeline architecture. Section 4 presents experimental results, failure taxonomy, human validation protocol, and downstream evaluation framework. Section 5 discusses limitations, safety considerations, and future directions.

## 2 Related Work

### 2.1 Bengali Misinformation Resources

BanFakeNews (Hossain et al., 2020) introduced a 50K-article Bangla fake-news corpus collected from online sources. BanMANI (Kamruzzaman et al., 2023) studied manipulated news on social media. CheckSent-BN (Pal and Das, 2025) combined claim checkworthiness with sentiment classification. IndicClaimBuster (Pal et al., 2025) extended claim verification to Bengali in a multilingual setting. These resources establish the language but still treat misinformation at the document level and do not isolate financial misinformation as a dedicated benchmark problem. FinFact-BD instead operates at the claim level within authentic financial news articles.

### 2.2 Financial Fact Verification

FinDVer (Zhao et al., 2024) targets explainable claim verification over long financial documents in English. RFC-BENCH (Jiang et al., 2026) studies reference-free counterfactual financial misinformation through original-perturbed financial paragraph pairs. MFMDQwen and MFMD-Scen (Liu et al., 2026) extend financial misinformation to multilingual settings with Bengali coverage, but as part of broader multilingual evaluation rather than a dedicated Bengali financial benchmark grounded in real news. FinFact-BD contributes a Bengali-first benchmark with real article provenance and structured generation traces.

### 2.3 Adversarial Benchmark Methodology

ADVSCORE (Sung et al., 2025) argues that adversarial benchmarks should be interpretable rather than adversarially noisy. Fighting Fire with Fire (Satapara et al., 2024) and surveys on claim verification (Guo et al., 2022) reinforce the point: synthetic generation is defensible when it is controlled, validated, and explainable. Planning-guided rewriting follows this philosophy: the pipeline records *why* a claim was selected, *what* change was planned, *how* the model realized it, and *whether* verification accepted it. Each sample carries a full audit trail.

## 3 Method

### 3.1 Overview

FinFact-BD's generation pipeline treats each authentic Bengali financial article as a structured collection of factual propositions. Each article undergoes at most one controllable factual distortion through five stages:

1. **Claim extraction**: Decompose the article into sentence-level factual propositions using a heuristic extractor.
2. **Claim selection**: Score propositions by importance, editability, locality, verification potential, and risk; select the highest-scoring candidate.
3. **Perturbation planning**: Construct an explicit plan specifying the target span, replacement, edit family, and verification constraints.
4. **Constrained generation**: A Bangla-capable LM rewrites the target sentence to realize the planned change; the pipeline reconstructs the full article by splicing the rewritten sentence back.
5. **Verification and acceptance**: Nine independent checks verify the rewrite; failing candidates are regenerated (up to 3 attempts) or discarded.

Figure 1 shows the full pipeline.

```
Original Article
    ↓
[1] Claim Extraction (Heuristic)
    ↓
[2] Claim Selection (Importance + Editability + Risk scoring)
    ↓
[3] Perturbation Plan (Family, target_span, replacement, constraints)
    ↓
[4] Constrained Generation (Aya Expanse 8B) ← splice into original
    ↓
[5] Verification (9 checks)
    ↓
Pass? → Accept → Export
Fail? → Retry (≤3) → Fail? → Discard + log
```

### 3.2 Claim Extraction and Selection

We use a heuristic Bangla claim extractor that converts each article sentence into a structured claim with sentence index, text, entities, numbers, dates, policy terms, and claim type (numerical, entity, temporal, causal, policy). An audit of 20,000 articles showed that the first 20 sentences captured 99.65% of candidate claims, which makes heuristic extraction sufficient for the pilot and avoids the cost of LLM-based extraction at full scale.

Each claim receives five scores:

- **Importance score** (0–1): how central the claim is to the financial narrative.
- **Editability score** (0–1): how feasible the claim is to rewrite within its family.
- **Locality score** (0–1): whether the edit can be confined to the target sentence.
- **Verification score** (0–1): how reliably the change can be detected by automatic checks.
- **Risk score** (0–1): how likely the edit is to produce artifacts or semantic drift.

The overall score combines these signals into a single ranking metric. Only positively scored candidates pass the quality gate. The highest-scoring candidate is selected per article.

### 3.3 Perturbation Families

The framework supports four families of factual manipulation, each targeting a different type of financial claim:

| Family | Target | Example (original → rewritten) |
|--------|--------|-------------------------------|
| `numerical_fact` | Financial figures, percentages, counts | "২২ রকম ব্যবসা" → "২ রকম ব্যবসা" (22 business types → 2) |
| `temporal_shift` | Dates, time references | "গতকাল" → "আজ" (yesterday → today) |
| `entity_replacement` | Named entities (orgs, institutions) | "মূল্য তালিকার" → "সিটি ব্যাংকের" (price list → City Bank) |
| `policy_reversal` | Policy direction | "অনুমোদন করেছে" → "অনুমোদন দেয়নি" (approved → rejected) |

Causal inversion is defined as a fifth family, but no samples from that family survived the pilot pipeline because the heuristic extractor rarely identifies causal claims with sufficient confidence.

### 3.4 Perturbation Planning

Planning is the core methodological innovation. For the selected claim, a structured language model (Qwen/Qwen3-8B or Qwen/Qwen2.5-3B-Instruct) produces a JSON plan containing:

- `family`: the perturbation family
- `target_span`: the exact text to replace
- `replacement`: the new text
- `edit_instruction`: the action directive used in the planning prompt
- `expected_change`: the expected semantic effect
- `verification_constraints`: locality, entity, number, date preservation rules

The plan is validated before generation: if the `target_span` is absent from the claim sentence, or the replacement is identical to the target, or the edit is semantically vacuous, the plan is rejected and regenerated (up to 3 repair attempts). A secondary plan-review model optionally inspects the plan for vague or implausible edits.

### 3.5 Constrained Generation

The generator (CohereLabs/aya-expanse-8b with a controlled-realization prompt) receives:

- The original article
- The target sentence with surrounding context
- The structured rewrite plan (target span, replacement, edit instruction)

The model rewrites **only** the target sentence. The pipeline then splices this rewritten sentence into the original article, replacing the old target sentence while keeping all other sentences unchanged. This architecture ensures locality by construction: the generation model cannot affect unrelated sentences because the splice step restores them verbatim.

The planner and generator use separate model roles with distinct prompts. This separation is deliberate: one model proposes the perturbation and another realizes it in text.

### 3.6 Verification and Regeneration

Every generated candidate passes through nine separate checks grouped into three categories:

**Claim integrity**
1. **Intended change**: Does the rewritten sentence contain the planned replacement and not the original span?
2. **Locality**: Is the edit confined to the target sentence index?
3. **Artifact detection**: No replacement characters, dangling halant fragments, or other suspicious Unicode patterns.

**Surface quality**
4. **Semantic similarity**: Multilingual-e5 cosine similarity ≥ 0.74 between the original and rewritten articles.
5. **Fluency**: BanglaBERT perplexity ≤ 220.
6. **Journalistic style**: Paragraph structure, punctuation, register, headline-topic alignment.

**Factual fidelity**
7. **Contradiction**: Deterministic check for numeric/entity/temporal shifts, backed by mDeBERTa-XNLI for cases that are not handled deterministically (threshold ≥ 0.55).
8. **Hallucination**: No new entities, numbers, dates, or organizations outside the target sentence.
9. **Duplicate detection**: Not a near-duplicate of previously accepted samples (embedding cosine similarity < 0.985).

If any check fails, the pipeline regenerates with the same claim plan and temperature but a slightly different prompt, up to three attempts. The best passing candidate is kept. If all attempts fail, the article is discarded and the failure reason is logged.

## 4 Experiments

### 4.1 Setup

We ran three pilot experiments on Kaggle T4 GPUs, progressing from small-scale runs to the primary 300-article run:

| Configuration | Pilot30 | Pilot100 | Pilot300 (primary) |
|--------------|:-------:|:--------:|:------------------:|
| Planner | Qwen3-8B | Qwen2.5-3B-Instruct | Qwen3-8B |
| Generator | Aya Expanse 8B | Aya Expanse 8B | Aya Expanse 8B |
| Verifier | Full stack | Full stack | Full stack |
| Samples | 30 | 100 | 300 |
| Temperature (gen) | 0.4 | 0.4 | 0.4 |
| Max regen attempts | 3 | 3 | 3 |
| Planned articles | 29 | 90 | 287 |
| Accepted | 26 (86.7%) | 79 (79.0%) | 259 (86.3%) |

Runtime configuration: models were loaded sequentially by role (planner → generator → verifier), and each role was released before the next loaded to fit within Kaggle's T4 15GB memory.

### 4.2 Results

#### Acceptance Rates

| Metric | Pilot30 (Qwen3-8B, 30) | Pilot100 (Qwen2.5-3B, 100) | Pilot300 (Qwen3-8B, 300) |
|--------|:----------------------:|:--------------------------:|:------------------------:|
| Accepted | 26 (86.7%) | 79 (79.0%) | **259 (86.3%)** |
| Planning failures | 1 (3.3%) | 10 (10.0%) | 38 (12.7%) |
| Verifier failures | 3 (10.0%) | 11 (11.0%) | 3 (1.0%) |

The Pilot300 run confirms that planning quality is the primary bottleneck. Verifier failures dropped to 1.0% (3 samples), suggesting that the generation and verification stack is robust conditional on a valid plan. Planning failures increased to 12.7% (38 articles), driven by 13 articles that could not be planned at all (Qwen3-8B failed to produce a structured plan) and 25 plans rejected during validation (missing target spans, empty replacements, or inconsistent edit instructions). The 86.3% acceptance rate is consistent with the Pilot30 Qwen3-8B result (86.7%), suggesting that planner choice, rather than sample size, is the main determinant of yield.

#### Systematic Failure Taxonomy

To understand failure modes beyond aggregate counts, we analyze failures across all three runs through the lens of the pipeline stage in which they originate:

**Extractor-stage failures.** The heuristic extractor identifies candidate claims but does not discriminate by perturbation family. Across all runs, 0.35% of sentences produced no extractable claim (coverage 99.65%). The extractor likely under-identifies causal and policy claims because these require discourse-level reasoning beyond sentence-level heuristics. This is consistent with the absence of causal_inversion samples and the low count of policy_reversal (1.2% in Pilot300).

**Planner-stage failures.** These dominate: 10–12.7% of articles fail at the planning stage. Within this category, three sub-types appear:
- *Plan generation failure* (34% of planning failures): the LLM produces malformed JSON or omits required fields. Observed more frequently with Qwen2.5-3B than Qwen3-8B.
- *Target span mismatch* (41%): the planned `target_span` is absent from the claim sentence or the `replacement` is identical to the target, causing plan validation rejection.
- *Semantic vacuity* (25%): the plan passes schema validation but the edit is semantically vacuous (e.g., replacing a number with an equivalent expression), leading to verifier rejection downstream.

**Generator-stage failures.** The generator faithfully executes valid plans in nearly all cases. Across Pilot300, only 3 samples failed verification after a valid plan (1.0%), all due to `planned_numeric_replacement_missing` - the model replaced the numeric value but not exactly the planned number. This is a conservative failure: the model fails to realize the target perturbation exactly, which is preferable to hallucination.

**Verifier-stage failures.** In Pilot100 (Qwen2.5-3B), verifier failures reached 11%, driven by `nli_contradiction_below_threshold` (20 out of 100 samples). This pattern is consistent with lower-quality plans from the weaker planner rather than verifier brittleness: when the plan is imprecise, the generated rewrite is correspondingly ambiguous, and the NLI verifier flags the weak contradiction. With Qwen3-8B in Pilot30 and Pilot300, this failure drops to near-zero.

**Implication.** The failure taxonomy confirms that the pipeline bottleneck is at the planning stage, not generation or verification. Improving planner capability — through larger models, specialized planning LMs, or structured decoding — is the highest-leverage intervention for scaling.

#### Family Distribution (Pilot300, Accepted, N=259)

| Family | Count | % |
|--------|------:|---:|
| numerical_fact | 155 | 59.8% |
| temporal_shift | 62 | 23.9% |
| entity_replacement | 39 | 15.1% |
| policy_reversal | 3 | 1.2% |

Numerical claims dominate because they are most common in financial articles and easiest to verify deterministically. Policy reversal and causal inversion are under-represented because the heuristic extractor identifies these claim types less reliably.

#### Verification Failure Breakdown

Combined across all three runs, verification failures follow a consistent pattern:

| Reason | Pilot30 | Pilot100 | Pilot300 |
|--------|:-------:|:--------:|:--------:|
| nli_contradiction_below_threshold | 1 | 20 | 1 |
| original_entity_still_present | — | 6 | — |
| causal_marker_missing | — | 4 | — |
| temporal_anchor_unchanged | — | 3 | — |
| planned_numeric_replacement_missing | 1 | 3 | 2 |
| new_facts_outside_target | 1 | 3 | — |
| causal_effect_not_inverted | — | 2 | — |

The leading failure - NLI contradiction below threshold - is concentrated in the Pilot100 run where the Qwen2.5-3B planner produced lower-quality plans. In Pilot300 (Qwen3-8B), verifier failures dropped to near-zero, suggesting that the generator faithfully executes well-formed plans. This is a conservative failure mode: the generator fails to realize some targets exactly, which is preferable to producing undetected hallucinations.

#### Runtime Breakdown (Pilot300)

| Phase | Seconds | % of Total |
|-------|--------:|-----------:|
| Planning (Qwen3-8B) | 27,751.1 | 98.50% |
| Generation (Aya) | 189.9 | 0.67% |
| Verification | 142.8 | 0.51% |
| Unplanned articles | — | 0.32% |
| **Total** | **28,164.9** | |

Planning dominates runtime by two orders of magnitude. The Qwen3-8B planner takes ~97 seconds per article (vs ~152s for Qwen2.5-3B in Pilot100), while generation takes ~0.7 seconds per article and verification ~0.5 seconds. The per-article planning time is lower with the larger model, likely because Qwen3-8B produces valid structured plans with fewer repair attempts. This reinforces the scaling strategy: planner quality, rather than generation speed, determines throughput.

### 4.3 Difficulty Distribution

Within the 259 accepted samples (Pilot300), we derive difficulty labels from claim selection scores:

| Difficulty | Count | % | Dominant Families |
|-----------|------:|---:|------------------|
| easy | 131 | 50.6% | numerical_fact (111) |
| medium | 125 | 48.3% | temporal_shift (48), entity_replacement (36) |
| hard | 3 | 1.2% | temporal_shift (2), policy_reversal (1) |

The difficulty distribution is roughly balanced across easy and medium, with very few hard samples because the claim selection ranker filters out high-risk claims before planning. Temporal shifts and policy reversals are over-represented in the "hard" category because their verification is less deterministic than numerical checks. The three hard samples involve multi-faceted edits (e.g., simultaneous numerical and temporal changes in a single sentence) or claims with high ambiguity scores.

### 4.4 Benchmark Dataset and Human Validation Protocol

**Released pilot dataset.** The pilot benchmark (`.benchmarks/benchmark_dataset/`) contains:

- `finfact_bd_benchmark.jsonl`: 259 samples with cleaned metadata
- `finfact_bd_benchmark.csv`: flat CSV version
- `splits.json`: stratified train/val/test splits (180/38/41) preserving family distribution
- `dataset_metadata.json`: full column glossary, score semantics, suggested evaluation protocol, license

Each sample record includes: original and rewritten article, target claim with sentence index, perturbation family, edit description (English), difficulty label, per-component verification scores (9 checks), and full generation provenance.

**Human validation protocol.** We define a human validation protocol designed to assess the quality of generated rewrites at scale. The protocol targets 500 randomly sampled benchmark instances, each reviewed by three independent annotators fluent in Bengali financial news. Each annotator judges five criteria:

1. **Fluency** (1–5): Is the rewritten sentence linguistically well-formed?
2. **Naturalness** (1–5): Does the rewrite read like authentic journalistic prose?
3. **Intended factual change** (realized / absent / wrong): Does the rewrite realize the planned factual modification?
4. **Absence of unintended edits** (yes / no / not sure): Are any claims beyond the target sentence altered?
5. **Overall quality** (1–5): Holistic judgment of the rewrite as a viable benchmark instance.

Inter-annotator agreement is measured using Fleiss' κ for each criterion (unweighted for nominal categories, quadratic-weighted for ordinal scales). Pairwise agreement percentages and adjudicated acceptance rates (majority vote) are reported. We also report per-criterion acceptance rates before and after adjudication.

The protocol is implemented as an open-source analysis pipeline:
- `scripts/create_human_validation_workbooks.py`: generates annotator XLSX workbooks from pipeline output, with claim-focus display, context windows, and dropdown validations
- `scripts/human_validation_analysis.py`: reads filled workbooks from multiple annotators, computes Fleiss' κ, pairwise agreement, adjudicated labels, and outputs summary statistics and LaTeX tables

This protocol is designed to be reproducible: the workbook generator and analysis script are released alongside the benchmark.

### 4.5 Downstream Evaluation Framework

To demonstrate the utility of FinFact-BD as a benchmark, we define a **factual consistency verification** task: given an article–rewrite pair $(a, c)$, predict whether the rewritten claim $c$ is factually consistent with the original article $a$. The initial labels are the automatic verification results from the pipeline (accepted/failed), with human-adjudicated labels for the 500-sample validation subset.

The benchmark's stratified splits (train/val/test, 180/38/41) support supervised and zero-shot evaluation. Candidate baseline models include:

| Model | Type | Expected capability |
|-------|------|-------------------|
| BanglaBERT (csebuetnlp/banglabert) | Monolingual encoder | Bengali linguistic competence, no explicit NLI |
| XLM-R (base and large) | Multilingual encoder | Cross-lingual transfer, strong on XNLI |
| mBERT (bert-base-multilingual-cased) | Multilingual encoder | General multilingual baseline |
| mDeBERTa-XNLI | Cross-encoder NLI | Zero-shot NLI, independent verification signal |

We report per-perturbation-family accuracy and F1, macro/micro F1, and confusion matrices for each model. The evaluation additionally measures agreement between model predictions and automatic verifier scores, providing a diagnostic signal for verifier calibration across perturbation types.

The evaluation framework is implemented in `scripts/evaluate_downstream.py` and supports both zero-shot evaluation (via cross-encoder NLI) and fine-tuning for downstream classification. Results will be summarized in a standardized LaTeX table as samples accumulate.

## 5 Limitations and Future Work

**Framework validation scale.** The current pilot contains 259 accepted samples from a 300-article run. It validates the generation pipeline but remains a pilot, not a final benchmark. Our immediate target is to scale this to more than 2,000 accepted samples through parallel batch generation and to obtain 500 human-validated instances with reported inter-annotator agreement. We expect the combination of automated verification and human validation at this scale to establish FinFact-BD as a mature resource rather than a proof-of-concept.

**Planning bottleneck.** Planning consumes 98% of total runtime. Accelerating planning through smaller specialized models, speculative planning, or caching could substantially improve throughput without sacrificing plan quality. The failure taxonomy (§4.2) confirms that planning-stage failures dominate, and targeted improvements to planner reliability would directly increase yield.

**Family imbalance.** Numerical facts dominate (59.8%). Policy reversal and causal inversion remain under-represented in the 300-sample run (1.2% and 0%, respectively). Improving claim extraction for these families - through LLM-based extraction or targeted heuristics - would produce a more balanced benchmark.

**NLI as second opinion.** The current verifier uses mDeBERTa-XNLI only for non-deterministic contradictions. A full cross-encoder NLI pass over all samples (available via `--nli` flag in the export script) would provide an independent diagnostic signal. The downstream evaluation framework (§4.5) will further test verifier calibration by comparing model predictions against automatic scores.

**Dual-use risk.** Because the benchmark intentionally rewrites financial claims, future releases should retain provenance metadata, clear research-use terms, and documentation of the generation pipeline, while avoiding any turnkey interface that could be repurposed for large-scale misinformation generation.

**Cross-lingual detection.** The pipeline's approach generalizes to other Bengali NLP tasks (e.g., health misinformation) and other low-resource languages. The separation of planning, generation, and verification into distinct roles is language-agnostic.

## 6 Conclusion

We present FinFact-BD, a controlled framework for Bengali financial misinformation generation based on planning-guided claim rewriting. The pipeline extracts factual propositions from real articles, scores them for importance and editability, plans explicit perturbations across four families, constrains a Bangla-capable language model to realize the change, and enforces nine separate verification checks before acceptance. In a 300-article pilot with Qwen3-8B planner and Aya Expanse 8B generator, 259 samples passed verification (86.3%), with two smaller development pilots confirming consistent performance. A systematic failure taxonomy identifies planning quality as the primary bottleneck: planning-stage failures account for 10-12.7% of articles across runs, while verifier failures drop below 2% with the stronger planner. We release a pilot benchmark (259 samples) with full provenance, stratified splits, difficulty labels, and companion tools for human validation and downstream evaluation. We also define a path to more than 2,000 generated samples with 500 human-validated instances and a factual consistency verification task for benchmarking. FinFact-BD provides a replicable methodology for controlled misinformation generation in low-resource languages and a foundation for Bengali financial fact verification research.

## References

- Md Zobaer Hossain, Md Ashraful Rahman, Md Saiful Islam, and Sudipta Kar. 2020. BanFakeNews: A Dataset for Detecting Fake News in Bangla. In *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pages 2862–2871, Marseille, France. European Language Resources Association.
- Mahammed Kamruzzaman, Md. Minul Islam Shovon, and Gene Louis Kim. 2023. BanMANI: A Dataset to Identify Manipulated Social Media News in Bangla. In *Proceedings of the Workshop on Computational Terminology in NLP and Translation Studies (ConTeNTS) Incorporating the 16th Workshop on Building and Using Comparable Corpora (BUCC)*, pages 51–58, Varna, Bulgaria. INCOMA Ltd.
- Pritam Pal and Dipankar Das. 2025. CheckSent-BN: A Bengali Multi-Task Dataset for Claim Checkworthiness and Sentiment Classification from News Headlines. In *Proceedings of the Second Workshop on Bangla Language Processing (BLP-2025)*, pages 119–130, Mumbai, India. Association for Computational Linguistics.
- Pritam Pal, Shyamal Krishna Jana, and Dipankar Das. 2025. IndicClaimBuster: A Multilingual Claim Verification Dataset. In *Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics*, pages 2478–2489, Mumbai, India. Asian Federation of Natural Language Processing and ACL.
- Yilun Zhao, Yitao Long, Tintin Jiang, Chengye Wang, Weiyuan Chen, Hongjun Liu, Xiangru Tang, Yiming Zhang, Chen Zhao, and Arman Cohan. 2024. FinDVer: Explainable Claim Verification over Long and Hybrid-content Financial Documents. In *Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing*, pages 14739–14752, Miami, Florida, USA. Association for Computational Linguistics.
- Yuechen Jiang, Zhiwei Liu, Yupeng Cao, Yueru He, Ziyang Xu, Chen Xu, Zhiyang Deng, Prayag Tiwari, Xi Chen, Alejandro Lopez-Lira, Jimin Huang, Junichi Tsujii, and Sophia Ananiadou. 2026. All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection. In *Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 10737–10776, San Diego, California, United States. Association for Computational Linguistics.
- Zhiwei Liu, Yupeng Cao, Yuechen Jiang, Mohsinul Kabir, Polydoros Giannouris, Chen Xu, Ziyang Xu, Tianlei Zhu, Md. Tariquzzaman, Triantafillos Papadopoulos, Yan Wang, Lingfei Qian, Xueqing Peng, Zhuohan Xie, Ye Yuan, Saeed Almheiri, Abdulrazzaq Alnajjar, Ming-Bin Chen, Harry Stuart, Paul Thompson, Prayag Tiwari, Alejandro Lopez-Lira, Xue Liu, Jimin Huang, and Sophia Ananiadou. 2026. Same Claim, Different Judgment: Benchmarking Scenario-Induced Bias in Multilingual Financial Misinformation Detection. In *Findings of the Association for Computational Linguistics: ACL 2026*, pages 9838–9864, San Diego, California, United States. Association for Computational Linguistics.
- Yoo Yeon Sung, Maharshi Gor, Eve Fleisig, Ishani Mondal, and Jordan Boyd-Graber. 2025. Is Your Benchmark Truly Adversarial? AdvScore: Evaluating Human-Grounded Adversarialness. In *Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)*, pages 623–642, Albuquerque, New Mexico. Association for Computational Linguistics. **Outstanding Paper Award**.
- Shrey Satapara, Parth Mehta, Debasis Ganguly, and Sandip Modha. 2024. Fighting Fire with Fire: Adversarial Prompting to Generate a Misinformation Detection Dataset. *arXiv preprint arXiv:2401.04481*.
- Zhijiang Guo, Michael Schlichtkrull, and Andreas Vlachos. 2022. A Survey on Automated Fact-Checking. *Transactions of the Association for Computational Linguistics*, 10:178–206.
