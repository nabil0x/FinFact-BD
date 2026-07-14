# FinFact-BD: A Benchmark for Bengali Financial Misinformation Detection

Frozen release: `FinFact-BD-v1.0`, regenerated on `2026-07-14`.

## Abstract

Financial misinformation in Bengali is still underrepresented in benchmark design, even though the underlying news ecosystem is large, economically consequential, and full of claims that are easy to distort in subtle ways. FinFact-BD addresses that gap with a controlled benchmark built from 10,000 real Bengali financial articles sampled from BENI v2 and 10,000 planning-guided claim rewrites generated from the same source pool. The generation process operates at the level of propositions rather than words, and it is organized around five misinformation families: numerical fact change, policy reversal, entity replacement, temporal shift, and causal inversion. That choice matters. It keeps the benchmark grounded in financial meaning rather than lexical noise.

The generation pipeline does not rely on microscopic span replacement. It first extracts candidate propositions with a structured reasoning model, then selects them through a deterministic policy that combines importance, editability, and diversity, drafts an explicit perturbation plan, and then constrains a multilingual Bangla-capable generation model to realize the planned change in natural prose while keeping the rest of the article stable. The model does not decide what misinformation to create; it only expresses a planned factual modification. Independent verification is the acceptance gate: failed candidates are regenerated up to three times before acceptance or discard. A separate human-validation pack covers 300 stratified samples, annotated by three Bengali-native annotators in a claim-centric workflow that focuses attention on the changed claim and its local context instead of forcing annotators to search entire articles for a hidden edit. The final release is split by original article identifier to prevent leakage across train, validation, and test partitions. Rich metadata records the rewrite trace, confidence, and provenance for each sample.

The benchmark is designed to make a narrow point clearly: Bengali financial misinformation is not just a low-resource classification task, and it is not well served by generic lexical perturbation. It is a claim-level reasoning problem, and the benchmark should look like one.

## Related Work

FinFact-BD sits between three lines of prior work.

The first is Bengali misinformation and claim-focused NLP. BanFakeNews introduced a large Bangla fake-news corpus; BanMANI studied manipulated social-media news in Bangla; CheckSent-BN combined claim checkworthiness with sentiment classification; and IndicClaimBuster extended claim verification to Bengali in a multilingual setting. These resources establish the language, but they do not isolate Bengali financial misinformation as a benchmark problem.

The second is financial fact verification and financial misinformation detection. FinDVer focuses on explainable claim verification over long, hybrid financial documents, while RFC-BENCH studies reference-free counterfactual financial misinformation in English through original-perturbed financial paragraphs. MFMDQwen/MFMDBench and MFMD-Scen extend financial misinformation work into multilingual settings and include Bengali coverage, but they do so as part of broader multilingual evaluation rather than a Bengali finance benchmark built from real news and fact-aware rewrites.

The third is adversarial benchmark methodology. ADVSCORE argues that a benchmark should be adversarial in a way humans can actually understand, not merely in a way that weakens a model. Fighting Fire with Fire and related survey work on adversarial attacks and claim verification reinforce the same point: synthetic generation is only defensible when it is controlled, validated, and interpretable. FinFact-BD follows that logic. It is not trying to invent a new detection paradigm. It is trying to provide a Bengali financial benchmark that is controlled enough to trust and realistic enough to matter.

## Why Planning-Guided Claim Rewriting

Symbolic token replacement is reproducible, but in long-form Bengali financial news it often produces edits that are too small to carry rhetorical force. A benchmark built from such microscopic changes is technically reproducible but visually unconvincing. It asks annotators to detect the edit by inspection, which is not the same thing as reading a realistic piece of misinformation.

The revised pipeline keeps the control but changes the realization step. We still decide what should change, and we still constrain the change to a specific factual proposition. But instead of swapping tokens directly, we extract candidate claims from the article, score them by importance, editability, and diversity, build a structured rewrite plan, and then constrain Aya Expanse 8B to rewrite only the targeted sentence or local paragraph. Qwen3-8B handles extraction and planning because those stages require structured reasoning, while Aya handles Bangla realization because that stage requires multilingual prose quality. The generation model is one component in a constrained generation system, not a full-article inventor and not the source of the misinformation decision. The planner determines the target claim, perturbation family, intended factual change, locality constraints, and verification requirements before generation begins. Independent verification then decides whether the output is accepted. If verification rejects a candidate, the pipeline regenerates up to three times, keeping a passing sample or discarding the article entirely. The goal is simple: keep the change explicit enough for humans to see, but structured enough for the benchmark to remain measurable. We therefore treat the article as a claim-bearing object rather than a bag of editable words.

```text
Original article
      ↓
Claim / proposition extraction
      ↓
Claim selection (importance + editability + diversity scoring)
      ↓
Explicit perturbation plan
      ↓
Constrained Bangla generation model (realizes planned local rewrite)
      ↓
Independent verification and acceptance gate
      ↓
Regeneration loop (up to 3 attempts)
      ↓
Human validation pack
      ↓
FinFact-BD release
```

## Method Outline

### 1. Source corpus and sampling

We sample 10,000 Bengali financial articles from BENI v2. The source pool is filtered for economic relevance and then stratified so the benchmark is not dominated by one source, one year, or one topic.

### 2. Claim extraction and claim planning

Each article is decomposed into sentence-level financial propositions. The generator looks for claim-bearing spans such as numbers, policy actions, entities, dates, causal connectors, and attribution cues. Those propositions are not edited arbitrarily. Each candidate claim receives three component scores: `importance_score` (how central the claim is to the article), `editability_score` (how feasible the claim is to rewrite within its rewrite family), and `diversity_bonus` (a penalty reduction for claims that diversify the family distribution across the dataset). The highest-scoring claim is selected, subject to family balance and duplicate avoidance, and then turned into a structured plan that specifies which factual claim will change, why that claim was chosen, and which local context must remain stable.

### 3. Role-specific model allocation

The implementation deliberately assigns different model families to different cognitive roles. Qwen3-8B extracts factual claims and emits structured rewrite plans. Aya Expanse 8B performs the constrained local Bangla rewrite. multilingual-e5-large measures article-level semantic similarity, mDeBERTa-XNLI checks contradiction between the original and rewritten target claim, and BanglaBERT provides an ELECTRA discriminator language-quality signal. Deterministic checks enforce locality and hallucination constraints.

### 4. Constrained Bangla generation model rewriting

The generation model receives the original article together with the structured perturbation plan and rewrites only the affected sentence or a single local paragraph. The prompt shape is narrow: it specifies the target claim, the desired factual change, and constraints on style and scope. The model is not allowed to invent a new article and does not decide what misinformation to create. Its job is narrower: express the planned factual change in natural Bangla while preserving journalistic style and keeping the surrounding article coherent. The planner and the generator are separate stages and use separate model roles.

### 5. Planning-guided rewrite families

The benchmark still uses five families of factual manipulation:

- Numerical fact change
- Policy reversal
- Entity replacement
- Temporal shift
- Causal inversion

These families are intentionally narrow. They map onto common forms of financial misinformation and keep the benchmark explainable. Under the revised pipeline, they operate as generation constraints rather than literal string-replacement rules.

### 6. Local planning and difficulty control

Each generated sample is limited to one planned factual change. The planner records the selected claim, rewrite family, edit scope, expected change, and constraints that must hold after generation. Difficulty is therefore controlled by the selected claim, perturbation family, edit visibility, and human-validation outcomes rather than by composing multiple hidden edits inside the same article.

### 7. Multi-stage verification

Generated samples pass through independent verification before acceptance. The verifier is not a final cleanup step; it governs whether the generated article can enter the dataset. The module has specific automatic checks:

1. **Claim integrity** checks whether the intended proposition changed and whether any extra proposition changed. Automatic checks: claim-level before/after comparison (comparing the plan's `target_span` against the generated output) and span-locality check (verifying the edit is confined to the planned sentence or paragraph).
2. **Surface quality** checks whether the rewrite is fluent Bangla, preserves journalistic style, and is not still microscopic. Automatic checks: fluency score (language model perplexity or a trained fluency classifier) and style preservation check (comparing token distributions and sentence length between original and rewritten spans).
3. **Semantic quality** checks whether the sample remains a believable Bangla financial news item, whether the contradiction score supports the intended label, and whether there are duplicate or near-duplicate artifacts. Automatic checks: semantic similarity (embedding cosine distance), contradiction score (NLI-based), and duplicate detection (exact and near-duplicate hash matching against the growing output set).

An NLI-based scorer can be used as an additional signal, but it is not the sole judge of quality. We also check whether exactly one planned proposition changed and whether the changed span matches the plan.

### 8. Regeneration loop

When verification rejects a candidate, the pipeline does not simply discard the article. It regenerates the rewrite up to three attempts, each time presenting the same claim plan to the generator with slightly varied prompting. The best passing candidate is kept. If all three attempts fail verification, the article is dropped from the output set and the failure reason is logged. This loop balances quality against coverage: it gives the generator multiple chances to produce a valid rewrite while capping computational cost.

### 9. Human validation

Human validation is deliberately narrower than generation. We do not ask annotators to comb through long articles in search of a hidden edit; that would turn the task into a memory test rather than a judgment task. Instead, the validation pack contains 300 samples, split evenly between original and perturbed articles and stratified across rewrite families. Each item is presented through an XLSX-based interface with a headline, a focused claim window, nearby context, and the full article text in the same sheet. Annotators label the sample as `original`, `perturbed`, or `not sure`, assign a confidence level, and add a short justification. A separate full-articles sheet remains available for easier scanning, but the center of gravity stays on the claim itself.

This choice is not cosmetic. Long financial articles often contain enough surrounding material to make a tiny edit hard to spot, even when the edit matters a great deal for the proposition being expressed. The validation protocol therefore measures what we actually care about: whether a human reader can recognize the sample as a clean article, a perturbed one, or an ambiguous case when only the relevant claim window is shown.

### 10. Release protocol

The frozen release is versioned as `FinFact-BD-v1.0` and regenerated on `2026-07-14`. Train, validation, and test partitions are assigned at the original-article level to prevent leakage across split boundaries. Each row preserves provenance and perturbation metadata so downstream analysis can trace what changed, where it changed, and how strong the change was.

### 11. Export and provenance

The pipeline exports three files:

- `finfact_bd_rewritten_raw.csv`: all generated samples before filtering, including failures and regeneration attempts.
- `finfact_bd_rewritten_filtered.csv`: samples that passed the multi-stage verification and are ready for human validation.
- `metadata.json`: full provenance records for every sample.

Each row carries complete traceability back to its origin: the source article, the target claim, the selection scores (`importance_score`, `editability_score`, `diversity_bonus`), the generator model used, the structured rewrite plan, the verification result, and the number of regeneration attempts. This provenance makes it possible to audit the pipeline, reproduce any individual sample, and analyze which claims were selected and why.

## Core Contribution

The claim of the paper is deliberately modest, and stronger for it: FinFact-BD provides a leakage-free, human-validated, proposition-level benchmark for Bengali financial misinformation. Its value is not that it creates a new task out of thin air. Its value is that it makes the task legible.

## What This Is Not

FinFact-BD is not:

- Free-form fake news generation. Every rewrite is constrained to a single planned factual proposition.
- An unconstrained paraphraser. The generator receives a structured plan and must stay within its scope.
- A token-replacement script. The realization step uses a sequence-to-sequence model that produces fluent, contextual rewrites.
- A classifier. The pipeline generates data; classification is the downstream task.
- Article-level rewriting without a plan. The planner decides what changes before the generator touches the text.

The generation model is one component in a constrained generation system. The scientific contribution is the integration of explicit planning, constrained realization, independent verification, regeneration, and full provenance for realistic localized factual distortion.
