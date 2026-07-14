# Research Philosophy: FinFact-BD Benchmark

> "The goal is not to create another dataset. The goal is to build a benchmark that future researchers will trust, reproduce, and build upon."

---

## Core Principles

### 1. Benchmark > Dataset

We build a benchmark, not just a dataset. A dataset is a collection of examples. A benchmark is a controlled experimental setup with research questions, baselines, evaluation protocols, and known failure modes. Every design decision should serve the goal of enabling rigorous, reproducible comparisons across models and methods.

### 2. Research Questions > "We built X"

The paper should not open with "We built FinFact-BD." It should open with the questions we are answering. The benchmark exists to serve those questions, not the other way around. If we cannot articulate what a benchmark teaches us, we should not publish it.

### 3. Human validation is mandatory

No dataset is complete without human validation. Synthetic perturbations, rule-based filters, and model-based quality checks are tools for scale, not substitutes for human judgment. We validate with domain-expert annotators and report inter-annotator agreement. Anything less is insufficient for a published benchmark.

### 4. Error analysis is a contribution

Understanding failures is as important as accuracy. A benchmark where every model scores 99% teaches us nothing. A benchmark where we can say "models fail on numerical misinformation because they do not attend to digit tokens" is genuinely useful. We treat error analysis as a first-class contribution, not an appendix.

### 5. Reproducibility is non-negotiable

Every experiment must be reproducible. Every result must be traceable from raw data to final number. Every preprocessing step must be documented. Every random seed must be logged. If a reviewer cannot reproduce a table from our paper, we have failed.

---

## Research Questions

The benchmark is organized around five research questions. These questions drive every design decision, from dataset construction to evaluation protocol.

**RQ1: How well do multilingual models detect Bengali financial misinformation?**

This establishes the current state of the art. We evaluate classical models, Bengali-specific PLMs, multilingual PLMs, and open-weight LLMs on the same benchmark. The question is not "which model is best" in the abstract, but "how much room for improvement exists" for each model family.

**RQ2: Which perturbation types are hardest?**

Each perturbation type targets a distinct linguistic failure mode. Numerical fact change tests digit attention. Policy reversal tests direction tracking in financial actions. Temporal shift tests time awareness. Causal inversion tests discourse parsing. Entity replacement tests real-world grounding. Breaking down difficulty by perturbation type reveals where models actually struggle.

**RQ3: Does financial terminology increase difficulty?**

We compare model performance on general-domain Bengali text versus financial-domain text. If a model that performs well on general misinformation detection fails on financial misinformation, the domain gap matters. This question directly informs whether domain-specific pretraining or fine-tuning is necessary.

**RQ4: Does numerical misinformation fool LLMs more than PLMs?**

Numerical claims require exact token attention. PLMs with small context windows may struggle differently than LLMs with larger context. We compare performance specifically on the numerical fact-change subset across model scales. This question has direct implications for model selection in real-world deployment.

**RQ5: Which reasoning skills are weakest?**

Beyond per-perturbation breakdowns, we cluster errors by the reasoning skill they require: comparing quantities, tracking policy direction, reasoning about time, recognizing entities, and understanding causality. This tells the community which capabilities need targeted improvement.

---

## Evaluation Philosophy

### Model hierarchy

We evaluate models in a deliberate hierarchy that mirrors the field's progression:

| Tier | Models | Purpose |
|------|--------|---------|
| Classical | TF-IDF + LR, FastText, SVM | Establishes floor performance |
| Bengali PLM | BanglaBERT, Bengali Electra | Tests language-specific pretraining |
| Multilingual PLM | mBERT, XLM-R, ModernBERT | Tests cross-lingual transfer |
| Open-weight LLM | Llama-3-8B, Qwen-2.5-7B, Gemma-2-9B, Mistral-7B | Tests scale with fine-tuning |
| Closed LLM | GPT-4o, Gemini-1.5-Pro, Claude-3.5-Sonnet | Tests zero-shot generalization |

This hierarchy lets us attribute performance differences to specific factors: language-specific pretraining, model scale, training paradigm, or architectural choices.

### Metrics

We report accuracy, precision, recall, and macro F1 as primary metrics. We also report per-perturbation-type F1 to reveal where models succeed and fail. Confusion matrices are reported for every model. We do not cherry-pick metrics that make our results look better.

### Statistical significance

We report p-values for all pairwise model comparisons using McNemar's test. If the difference between two models is not statistically significant, we say so. Effect sizes matter more than raw numbers.

### Ablation studies

We run ablation studies to understand what drives performance. What happens when we remove the quality filter? When we change the perturbation threshold? When we reduce training data? Ablations tell us which design decisions matter and which are cosmetic.

---

## Dataset Philosophy

### Synthetic data is a tool, not the end goal

We use planning-guided claim rewriting because it is reproducible, auditable, and explainable. Every factual change has a known cause. This is a methodological choice, not a claim that synthetic data is superior to real data. Controlled rewrites let us manage difficulty and measure specific capabilities while keeping the resulting misinformation human-legible.

The objective is not to fabricate synthetic news articles. The framework generates controlled misinformation by rewriting exactly one factual claim inside an authentic Bangla financial news article while preserving the surrounding journalistic context. The model is not responsible for deciding what misinformation to create. It realizes a pre-specified perturbation plan, and independent verification governs acceptance.

### Real-world validation is essential

A benchmark built entirely on synthetic data has limited ecological validity. We include a held-out test set of real Bengali financial misinformation to check whether failures on synthetic perturbations predict failures on real misinformation. If they do not, the benchmark's utility is limited.

### Rich metadata enables deeper analysis

Every instance carries metadata: source article, perturbation type, perturbation parameters, article length, domain, and date range. This metadata lets researchers slice results in ways we did not anticipate. A benchmark without metadata is a flat list of examples.

### Difficulty levels help understand model limits

Not all misinformation is equally hard to detect. We annotate difficulty based on perturbation magnitude and linguistic complexity. Reporting performance at different difficulty levels reveals where model capability breaks down, rather than hiding it behind aggregate accuracy.

---

## Paper Writing Philosophy

### Frame around research questions, not "we built X"

The paper structure follows the research questions, not the construction pipeline. We do not write "Section 2: Dataset Construction, Section 3: Experiments." We write "Section 2: Research Questions, Section 3: Methodology for Answering Them, Section 4: Results and Analysis."

### Use "to the best of our knowledge" instead of "first"

The word "first" is a factual claim that requires exhaustive literature review. "To the best of our knowledge" conveys the same information with appropriate epistemic humility. It is also harder for reviewers to challenge.

### Error analysis as major contribution

The error analysis section should be at least as long as the results section. We cluster errors, explain patterns, identify root causes, and suggest directions for improvement. This is what makes the benchmark useful beyond a leaderboard.

### Limitations section is mandatory

We describe what the benchmark cannot do. Which types of misinformation are not covered? Which languages are excluded? Which failure modes are not tested? A limitations section is not a weakness. It is a sign that we understand our contribution's scope.

### No marketing language

Words like "state-of-the-art," "unprecedented," "revolutionary," and "groundbreaking" do not appear in the paper. Results speak for themselves. If our numbers are good, the numbers will convince readers. Adjectives will not.

---

## Quality Standards

### Human validation

All human annotation follows a structured protocol. We report Fleiss' kappa for inter-annotator agreement. A kappa below 0.6 triggers protocol revision and re-annotation. We do not publish datasets with poor agreement and hope nobody notices.

### Statistical significance

No performance comparison is reported without a significance test. McNemar's test for pairwise comparisons. Confidence intervals where appropriate. We do not claim Model A is better than Model B unless the difference is statistically significant at p < 0.05.

### Reproducibility

All code, data, and experiment configurations are released. Every result in the paper can be reproduced by running the provided scripts. We log random seeds, hardware specifications, and software versions. If a reviewer cannot reproduce a result, we fix the issue before publication.

### Documentation

Every design decision is justified in the paper or in supplementary materials. Why these five perturbation types? Why this quality filter threshold? Why this train/test split? If we cannot explain a decision, we revisit it.

---

## Evaluation Philosophy (Detailed)

A benchmark is only useful if models fail in interesting ways. High accuracy on every perturbation type means the benchmark is too easy. Low accuracy everywhere means it is too hard or poorly constructed. The sweet spot is where some perturbation types are easy and others are hard, revealing specific model capabilities and weaknesses.

We ask three questions about every result:

1. **Where do models fail?** Which perturbation types, which difficulty levels, which article types?
2. **Why do they fail?** Is it a tokenization issue? A context window limitation? A domain knowledge gap?
3. **Can humans solve those cases?** If humans also struggle, the examples may be ambiguous. If humans succeed where models fail, the gap is meaningful.

---

## Human Validation Philosophy

Human annotation exists to answer questions machines cannot. We do not use human annotation merely to "verify" synthetic labels. We use it to answer questions like:

- Does this perturbation look realistic to a Bengali reader?
- Could this misinformation spread on social media?
- Would an investor be misled by this?
- Is the original article clearly reliable?

These questions measure ecological validity, not just label correctness. A perfectly generated perturbation that looks obviously fake to a human reader has limited value as a misinformation benchmark.

---

## Error Analysis Philosophy

Accuracy is not the end. It is the beginning of analysis. Every error is information:

- **False positives** reveal ambiguity in the original articles or over-sensitivity in the model.
- **False negatives** reveal model weaknesses, specific perturbation types that bypass detection, or gaps in the model's financial knowledge.
- **Confused pairs** reveal which perturbation types are linguistically similar and which are distinct.

We cluster errors by perturbation type, difficulty level, article length, and reasoning skill. We visualize error distributions. We identify the ten most informative error cases and analyze them in detail. This level of analysis is what separates a benchmark paper from a dataset announcement.

---

## Engineering Philosophy

Research code should be readable, modular, versioned, tested, and documented. Code communicates ideas, not cleverness. Every function has a docstring. Every script has a README. Every experiment has a configuration file. The codebase is a research artifact, not just an implementation detail.

---

## Ethical Principles

This dataset studies misinformation. It does not promote misinformation. The purpose is defensive: building systems that detect and flag misleading financial content. We acknowledge potential risks of misuse and describe mitigation strategies in the paper. We do not generate content designed to deceive real people.

---

## Open Science

We release everything: dataset, code, evaluation scripts, generation pipeline, documentation, trained baselines, and experiment configurations. Open science increases impact. A benchmark locked behind a download form or a paywall is a benchmark that will not be used.

---

## Long-Term Vision

FinFact-BD should become a standard benchmark for Bengali financial misinformation research. It should enable work on misinformation detection, adversarial robustness, multilingual transfer learning, financial fact verification, and trustworthy financial language models.

A benchmark earns trust not because it is large, but because the community believes its methodology, trusts its quality, and can reproduce its results.

---

## Guiding Questions

Before making any decision, ask:

1. Does this improve scientific rigor?
2. Is this reproducible?
3. Can another researcher understand this?
4. Does this make the benchmark more realistic?
5. Would a reviewer trust this methodology?
6. Will this still make sense in five years?
7. Does this answer an actual research question?
8. Are we measuring what we claim to measure?

If the answer to any question is "no," reconsider the design.

---

*Last updated: July 14, 2026*
