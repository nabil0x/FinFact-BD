# Novelty Assessment: FinFact-BD Benchmark

To the best of our knowledge, FinFact-BD is the first comprehensive benchmark for Bengali financial misinformation that combines planning-guided Bangla claim rewriting, human validation, and multi-model evaluation. This document assesses what is novel about the work, what is not, and how to defend the claims against reasonable criticism. It refers to the frozen `FinFact-BD-v1.0` release regenerated on `2026-07-14`.

---

## What Is Novel

### 1. Benchmark Scope

FinFact-BD is the first benchmark that pairs planning-guided Bangla claim rewriting with real-world article data specifically for Bengali financial misinformation. Existing Bengali NLP datasets cover news classification, sentiment analysis, and named entity recognition. None target the financial misinformation detection task with this combination of controlled rewriting strategies and domain-specific source material from the Bangladeshi financial ecosystem.

The dataset draws from BENI v2, a corpus of Bengali economy articles, and applies five fact-aware rewriting families (numerical fact change, policy reversal, entity replacement, temporal shift, causal inversion) to create labeled real/fake pairs. Each rewriting family targets a distinct financial proposition type, making the benchmark useful for diagnosing model weaknesses, not just measuring accuracy.

The generation approach uses a planning-guided claim rewriting pipeline: claims are extracted from source articles, scored for suitability, converted into explicit perturbation plans, realized by a constrained multilingual generation model, and accepted only after independent verification. The novelty lies in the controlled system around the generator, not in the generator itself. The implementation uses separate model roles: Qwen3-8B for structured extraction and planning, Aya Expanse 8B for local Bangla rewrite realization, multilingual-e5-large for similarity, mDeBERTa-XNLI for contradiction, and BanglaBERT for language-quality verification.

### 2. Human Validation

FinFact-BD includes the first human-validated dataset for Bengali financial text with explicit three-annotator agreement reporting. The validation protocol covers 300 stratified samples with free-text justifications, enabling analysis of where human judgment diverges from model predictions and where annotators disagree with each other.

Most existing Bengali NLP datasets use single-annotator labels or do not report inter-annotator agreement. The three-annotator design with Fleiss' kappa reporting provides a reliability signal that is absent from comparable resources.

### 3. Rich Metadata

Each instance in FinFact-BD carries structured metadata that goes beyond the label: perturbation type, source article reference, article length, domain tag, and date range. This metadata enables researchers to slice evaluation by perturbation difficulty, analyze performance across financial subdomains, and study temporal effects.

No existing Bengali financial text dataset provides this level of annotation granularity. The perturbation-type labels in particular allow fine-grained error analysis that binary original/fake datasets cannot support.

### 4. Model Evaluation

FinFact-BD includes the first systematic comparison across the full model hierarchy for Bengali financial text: classical baselines (TF-IDF, FastText, SVM), Bengali-specific PLMs (BanglaBERT, Bengali Electra), multilingual PLMs (mBERT, XLM-RoBERTa, ModernBERT), open-weight LLMs with LoRA fine-tuning (Llama-3, Qwen-2.5, Gemma-2, Mistral), and closed LLMs evaluated zero-shot (GPT-4o, Gemini-1.5-Pro, Claude-3.5-Sonnet).

Previous Bengali NLP work typically evaluates one or two model families. The breadth of this comparison, spanning models from 110M parameters to 100B+, provides a reference point for what is achievable at each compute tier.

### 5. Error Analysis

FinFact-BD includes a structured error analysis that categorizes model failures by perturbation type (numerical fact change, policy reversal, entity replacement, temporal shift, causal inversion) and tests whether synthetic perturbation failures predict real-world failures. The inclusion of a held-out real-world test set of actual Bengali financial misinformation makes this analysis possible.

Most benchmark papers report aggregate metrics. The failure-mode decomposition in FinFact-BD tells researchers which specific linguistic capabilities are missing, not just that performance is low.

### 6. Controlled Generation Pipeline

FinFact-BD uses a multi-stage controlled generation pipeline, not free-form generation or token-level rule replacement. The pipeline is: claim extraction, claim selection, rewrite planning, constrained Bangla generation, independent verification, acceptance, and regeneration when needed. Each stage has a defined role.

The planner decides what change should happen. The generation model (Aya Expanse 8B by default) is constrained to realize that change by rewriting only the target sentence or paragraph, never the full article. The verifier then decides whether the output is accepted, using criteria across three dimensions: claim integrity (did the intended claim change, and only that claim?), surface quality (is the Bangla fluent and journalistic?), and semantic quality (is the rewrite a believable financial news item that contradicts the original where intended?).

This is not free-form generation. It is a controlled, auditable process where every accepted sample has a traceable provenance: the source claim, the selection score, the rewrite plan, the generator model, the verification result, and the number of regeneration attempts. The pipeline constrains the generator at every step, producing targeted misinformation for benchmark construction rather than arbitrary text. The model never has the final say; independent verification governs acceptance.

---

## What Is NOT Novel

Honest assessment of what already exists.

### Bengali NLP Datasets Exist

Bengali news classification, sentiment analysis, and fake news detection datasets are established. BanglaNews, Bengali Sentiment Corpus, and related resources provide coverage of general Bengali text classification. FinFact-BD does not claim to be the first Bengali NLP dataset. It claims to be the first that targets financial misinformation specifically.

### Financial Misinformation Detection Exists

English-language financial fact-checking and misinformation detection are active research areas. RFC-BENCH (ACL 2026), FinFact, and related work provide benchmarks and methods for English financial text. FinFact-BD builds on this tradition. The perturbation methodology draws from RFC-BENCH. The contribution is language and domain coverage, not a fundamentally new detection paradigm.

### Synthetic Data Generation Exists

Synthetic data generation for NLP benchmarks is well-established. Perturbation-based approaches, back-translation, and LLM-based generation are all common techniques. FinFact-BD uses synthetic perturbation, not a novel generation method. The contribution is the combination of perturbation with human validation and structured metadata, not the generation approach itself.

### Model Evaluation Exists

Systematic model evaluation across architectures is standard practice in benchmark papers. FinFact-BD follows established evaluation protocols. The contribution is applying this evaluation to a language and domain combination that has not been studied, not inventing new evaluation methodology.

### Bangla Generation Models Exist

The Bangla generation model itself is not novel. Aya Expanse 8B, Qwen3-8B, multilingual-e5-large, mDeBERTa-XNLI, and BanglaBERT are existing pretrained models. The novelty is in how we assign them single responsibilities and constrain the generator's output within a planning and verification pipeline. Using off-the-shelf models inside a controlled, auditable system is a different contribution than the models themselves.

---

## Defensibility Assessment

### How to Defend Against "But X Already Exists"

The strongest defense is specificity. When a reviewer says "Bengali fake news detection already exists," the response is: "Yes, but Bengali financial misinformation detection with synthetic perturbation, human validation, and multi-model evaluation does not." Each qualifier matters. Removing any one of them weakens the claim. Together, they define a contribution that has not been made.

The key is to never claim novelty in isolation. Every novelty claim should be scoped to the intersection of dimensions:

- Bengali + financial + misinformation + benchmark (not just "Bengali NLP")
- Synthetic perturbation + human validation + structured metadata (not just "synthetic data")
- Classical + PLM + LLM evaluation (not just "model comparison")

### How to Position Against Related Work

Frame FinFact-BD as filling a gap in the landscape, not replacing existing work. FinDVer and RFC-BENCH do excellent work for English financial verification. MFMDQwen and MFMD-Scen extend coverage to Bengali within multilingual financial settings. FinFact-BD complements these by providing Bengali-specific news perturbation with validation rigor that neither targets.

The comparison table in Section 6 makes this concrete. Reviewers can see exactly where FinFact-BD stands relative to alternatives.

### How to Handle Reviewer Pushback

Common reviewer objections and suggested responses:

**"The perturbation methodology is not novel."**
Correct. The perturbation methodology is adapted from RFC-BENCH. The novelty is in the application domain, the human validation protocol, and the structured metadata that enables fine-grained error analysis. Methodological novelty and contribution novelty are different things.

**"20K samples is small compared to English benchmarks."**
FinFact-BD is sized for the available Bengali financial text corpus, not for parity with English benchmarks. The 10,000 original articles are sampled from BENI v2 and balanced with 10,000 perturbations. Scaling further would require additional corpus development, which is outside the scope of this work.

**"Synthetic perturbations don't reflect real misinformation."**
This is why FinFact-BD includes a real-world test set. The benchmark explicitly tests whether synthetic perturbation failures predict real-world failures (RQ4). Acknowledging this limitation and designing the benchmark to address it is stronger than ignoring it.

**"Why not more perturbation types?"**
Five perturbation types cover the primary linguistic failure modes identified in the financial misinformation literature. Adding more would increase annotation cost without proportional diagnostic value. The current set is sufficient to answer the research questions.

**"Bangla generation models already exist."**
Yes, but we are not claiming novelty for the generator. The novelty is in the controlled system that constrains the generator to realize a pre-specified factual modification and accepts outputs only after independent verification. Using an existing model inside a planning-guided, verifiable, provenance-rich pipeline is a valid contribution.

---

## Novelty vs Contribution

These are different things. Conflating them weakens both claims.

### Novelty: What Is New

The specific combination of Bengali financial domain, synthetic perturbation with five corruption types, three-annotator human validation, structured metadata, and full-model-hierarchy evaluation has not been published before. Each component exists elsewhere. The assembly does not.

### Contribution: What Is Useful

FinFact-BD gives researchers a tool for diagnosing model weaknesses on Bengali financial text. The perturbation-type labels enable targeted improvement. The real-world test set validates ecological validity. The model comparison provides a baseline for future work. The dataset and code are released for reproducibility.

### Impact: What Matters

Bengali is spoken by over 200 million people. Bangladesh has a growing digital financial ecosystem with active misinformation problems (pump-and-dump schemes on the DSE, bKash/Nagad fraud, remittance scams). A benchmark that enables systematic study of financial misinformation detection in Bengali serves both the NLP research community and the practical need for automated fact-checking in an under-served language.

---

## Comparison Table

| Feature | FinFact-BD | RFC-BENCH | MFMDQwen | MFMD-Scen |
|---------|------------|-----------|----------|------------|
| **Language** | Bengali | English | Multilingual | Multilingual |
| **Domain** | Financial misinformation | Financial text | Financial text | Financial claims |
| **Size** | 20K (10K pairs) | ~10K | Large | Varies |
| **Generation Approach** | Claim-guided Bangla rewriting with planning + verification | Token replacement | N/A | N/A |
| **Human Validation** | 3 annotators, Fleiss' kappa | Limited | Not reported | Not reported |
| **Perturbation Metadata** | 5 types, structured | Types reported | N/A | N/A |
| **Difficulty Levels** | Yes (via metadata) | No | No | No |
| **Real-World Test Set** | Yes (separate) | No | No | No |
| **Model Evaluation Hierarchy** | Classical to LLM | PLMs only | LLM-focused | LLM-focused |
| **Error Analysis** | By perturbation type | Aggregate | Aggregate | Aggregate |
| **Code Released** | Yes | Yes | Varies | Varies |

**Where FinFact-BD stands out:** human validation with agreement reporting, structured perturbation metadata, real-world test set, and the full model hierarchy from classical baselines through closed LLMs. No existing benchmark covers all of these for Bengali financial text.

---

*Assessment prepared July 2026. Intended for COLING 2026 FinNLP Workshop submission.*
