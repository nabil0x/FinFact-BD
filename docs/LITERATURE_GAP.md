# Literature Gaps: Research Questions in Bengali Financial Misinformation

This document frames open research questions in Bengali financial misinformation detection. Rather than claiming novelty for its own sake, it asks: what do we not yet know, and why does it matter?

For a paper-ready summary of the closest related work, see [docs/RELATED_WORK.md](RELATED_WORK.md).

---

## Research Questions

### RQ1: Can controlled claim rewriting with Bangla generation models produce realistic Bengali financial misinformation?

**The question:** Can controlled claim rewriting with Bangla generation models produce realistic Bengali financial misinformation that is both human-legible and benchmark-quality?

**What we know:** Bangla generation models like `csebuetnlp/banglat5` and `Vacaspati/BanglaByT5` exist for seq2seq tasks. Controlled rewriting methodology exists in English for benchmark construction. But nobody has combined claim-level planning with constrained Bangla generation to produce financial misinformation samples that are both human-legible and suitable for benchmark evaluation.

**Why it matters:** If multilingual models fail here, it suggests language-specific pretraining or fine-tuning is necessary. If they succeed, it opens the door to zero-shot cross-lingual transfer from English financial NLP.

---

### RQ2: Which perturbation types are hardest?

**The question:** Financial misinformation takes many forms: numerical changes, policy reversals, entity replacement, temporal shifts, and causal inversions. Which of these are hardest for current models to detect in Bengali?

**What we know:** Claim-rewriting studies exist for English (RFC-BENCH). But nobody has isolated Bengali financial-news claim-rewriting families or compared difficulty across rewriting strategies in Bengali.

**Why it matters:** Knowing which perturbation types are hardest tells us where models are weakest. It also helps prioritize which types of misinformation to focus on in real-world detection systems.

---

### RQ3: Does financial terminology increase difficulty?

**The question:** Bengali financial text contains domain-specific terms like DSE (Dhaka Stock Exchange), bKash, Nagad, remittance terminology. Does the presence of these terms make misinformation harder to detect, or does domain-specific vocabulary actually help models?

**What we know:** General Bengali NLP benchmarks exist (BENI v2, BengaliBERT evaluations). Financial NLP benchmarks exist in English and Chinese. But nobody has studied whether financial domain-specific terminology creates a difficulty spike in Bengali misinformation detection.

**Why it matters:** If domain terminology increases difficulty, it suggests models need domain-specific pretraining or vocabulary. If it decreases difficulty, it suggests financial text has regularities that models can exploit.

---

### RQ4: Does numerical misinformation fool LLMs more than PLMs?

**The question:** Numerical misinformation (changing 500 to 1000, or halving a revenue figure) is a common financial manipulation. Do larger language models handle numerical reasoning better than smaller pretrained language models, or does the problem persist across scales?

**What we know:** LLMs have been evaluated on numerical reasoning in English. Some studies show LLMs struggle with precise numerical claims. But nobody has compared LLM vs. PLM performance on Bengali numerical misinformation specifically.

**Why it matters:** Numerical misinformation causes direct financial harm. If scale doesn't help, we need specialized numerical reasoning modules. If scale helps, we know where to focus engineering effort.

---

### RQ5: Which reasoning skills are weakest?

**The question:** Detecting financial misinformation requires multiple reasoning skills: comparing quantities, tracking policy direction, verifying entity consistency, tracking time, and following causal chains. Which of these skills are weakest in current models when applied to Bengali financial text?

**What we know:** Reasoning benchmarks exist in English (Chain-of-Thought, numerical reasoning, causal inference). Bengali NLP has focused on classification tasks without decomposing reasoning skills. But nobody has analyzed which specific reasoning skills are weakest for Bengali financial misinformation.

**Why it matters:** Fine-grained skill analysis tells us whether models lack knowledge, lack reasoning ability, or lack both. This guides future work toward the right interventions.

---

## What Existing Work Covers

### Financial Misinformation Detection

Several recent papers address financial misinformation, but their focus is still different from Bengali financial news perturbation:

- **FinDVer** (EMNLP 2024) benchmarks explainable claim verification over long, hybrid financial documents with expert annotations and reasoning explanations. It is finance-specific, but it is document-grounded claim verification rather than synthetic Bengali news perturbation.
- **RFC-BENCH** (ACL 2026) provides a reference-free benchmark for counterfactual financial misinformation detection in English, with paired original-perturbed paragraphs and four manipulation families. It is the closest methodological precursor to FinFact-BD.
- **MFMDQwen / MFMDBench** (2026) extend financial misinformation evaluation to English, Chinese, Greek, and Bengali. However, the benchmark is multilingual and task-driven rather than Bangladesh-specific and perturbation-wise.
- **MFMD-Scen** (2026) studies scenario-induced bias in multilingual financial misinformation detection, including Bengali. It is valuable for bias analysis, but it does not study Bengali financial-news perturbation families.

### Bengali NLP

Bengali NLP has made significant progress, but misinformation detection remains understudied:

- **BanFakeNews** (2020) introduced a large Bangla fake-news corpus for general fake-news detection.
- **BanMANI** (2023) labeled manipulated Bangla social-media news relative to reference articles.
- **CheckSent-BN** (2025) targets claim checkworthiness and sentiment classification in Bengali headlines.
- **IndicClaimBuster** (2025) extends claim verification to Bengali within a multilingual claim-verification benchmark.
- **BENI v2** (2025) is a large Bengali news corpus with 38,107 economy articles. It provides the source material for our dataset but contains no misinformation labels.

These datasets establish Bengali resources for news and verification, but they do not target Bengali financial misinformation with controlled perturbations.

### Methodology

Adversarial-benchmark research increasingly emphasizes human-grounded validation and explicit quality checks:

- **ADVSCORE** argues that a benchmark should be measured not only by model difficulty but also by how well it aligns with human performance.
- **Fighting Fire with Fire** uses NLI-style consistency checks to judge synthetic real-news and fake-news generation.
- **Adversarial Attacks Against Automated Fact-Checking: A Survey** summarizes how many attacks rely on systematic perturbations, but also notes the lack of a universal evaluation framework.

These methodological papers support the need for human validation, explicit contradiction checks, and leakage-free benchmark construction.

### What these works share

All existing work falls into one of three categories:
1. Financial NLP in English (mature but not transferable to Bengali)
2. Bengali NLP in general (strong language models but no financial misinformation)
3. Multilingual NLP (broad coverage but no deep Bengali financial analysis)

None of them sit at the intersection: Bengali + financial + misinformation + claim-rewriting analysis + reasoning skills.

---

## What Nobody Has Done

These are specific, documented gaps in the literature. Each is a research question worth answering.

| Gap | Existing Work | What's Missing |
|-----|---------------|----------------|
| Systematic evaluation of multilingual models on Bengali financial text | Multilingual models exist; Bengali NLP exists; financial NLP exists | No study tests multilingual models specifically on Bengali financial misinformation |
| Perturbation-wise difficulty analysis for Bengali | English perturbation studies (RFC-BENCH) | No equivalent analysis for Bengali, where script and morphology differ |
| Financial domain-specific difficulty analysis in Bengali | General Bengali NLP benchmarks | No study measures whether financial terminology creates a difficulty spike |
| Numerical misinformation comparison across model scales | LLM evaluation in English | No comparison of LLM vs. PLM on Bengali numerical reasoning |
| Reasoning skill decomposition for Bengali financial misinformation | English reasoning benchmarks | No analysis of which reasoning skills are weakest in Bengali |
| Controlled generation for Bengali financial misinformation | Bangla generation models exist (banglat5, BanglaByT5); controlled rewriting methodology exists in English | No study uses Bangla generation models within a planning+verification pipeline for benchmark construction |

**Note:** MFMDQwen claims Bengali coverage, but it does not provide:
- Bangladesh-specific financial manipulation types (DSE, bKash, Nagad)
- Perturbation-type breakdown
- Domain-specific evaluation
- Reasoning skill analysis
- Human validation

These are not just missing features. They are open research questions that require controlled experiments to answer.

---

## How We Address These Gaps

FinFact-BD is designed as a controlled experimental setup, not just a dataset release. Each research question maps to specific experimental components:

| Research Question | Experimental Component | What We Evaluate |
|-------------------|----------------------|------------------|
| RQ1: Generation model quality | Evaluate claim-guided Bangla rewriting with banglat5, BanglaByT5 | Human-legibility, benchmark quality, verification pass rate |
| RQ2: Hardest claim-rewriting families | Stratified evaluation across 5 claim-rewriting families | Per-family F1 scores, difficulty ranking |
| RQ3: Financial terminology difficulty | Compare performance on financial vs. general Bengali text | Accuracy gap, error analysis |
| RQ4: Numerical misinformation across scales | Compare PLMs (110M-395M) vs. LLMs (7B-8B) on numerical fact-change examples | Numerical accuracy, scale vs. performance |
| RQ5: Weakest reasoning skills | Decompose errors into reasoning skill categories | Skill-level accuracy, failure patterns |

**Dataset construction:** 20,000 samples from BENI v2, with 5 claim-rewriting families generated through a controlled pipeline: claim extraction → selection → planning → Bangla generation → multi-stage verification. Human validation on 300 samples, and a held-out real-world test set.

**Model evaluation:** Classical baselines (TF-IDF, FastText, SVM), Bengali PLMs (BanglaBERT), multilingual PLMs (XLM-R, mBERT, mDeBERTa), open-weight LLMs (Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B), and closed LLMs (GPT-4o, Gemini-1.5-Pro, Claude-3.5-Sonnet).

**Quality filtering:** Two-stage pipeline (controlled rewriting + mDeBERTa NLI) with human validation to calibrate filter accuracy.

---

## Contribution Claims

These are specific, defensible claims. Each is verifiable through the released dataset and code.

1. **First dedicated benchmark for Bengali financial misinformation.** FinFact-BD provides 20,000 labeled samples across 5 perturbation types, sourced from real Bengali financial news. No existing dataset covers Bengali financial misinformation with this scope and structure.

2. **First human-validated synthetic dataset with rich metadata.** 300 samples annotated by 3 Bengali-native speakers with finance domain familiarity. Each sample includes perturbation type, source article reference, and metadata. This enables both automated evaluation and human-judgment calibration.

3. **First systematic evaluation of multilingual models on Bengali financial text.** We evaluate XLM-R, mBERT, and mDeBERTa-v3 on Bengali financial misinformation detection. Previous work evaluated these models on general Bengali tasks or English financial tasks, but not this intersection.

4. **First perturbation-wise difficulty analysis in Bengali.** We measure detection accuracy for each of 5 perturbation types and identify which are hardest. No previous study has done this for Bengali, where script complexity and morphology may create different difficulty patterns than English.

5. **First reasoning skill analysis for Bengali financial misinformation.** We decompose model errors into reasoning categories (numerical, policy, entity, temporal, causal) and identify which skills are weakest. No previous study has applied this decomposition to Bengali financial text.

6. **Controlled generation pipeline for Bengali financial misinformation.** The pipeline combines claim-level planning with constrained Bangla generation and multi-stage verification. This is not free-form generation — it is a structured process that produces auditable, verifiable misinformation samples.

---

*Analysis Date: July 2026*
*Status: Gap verified, experimental design mapped to research questions*
