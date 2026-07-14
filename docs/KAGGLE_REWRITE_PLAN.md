# Kaggle Controlled Claim Rewriting Plan

Frozen release target: `FinFact-BD-v1.0`.

## Goal

Move FinFact-BD generation to a Kaggle-native pipeline that produces claim-guided Bangla rewrites instead of microscopic symbolic edits. The notebook should do the full generation pass on Kaggle: load the source articles, select candidate claims, plan the rewrite, execute the change with a Bangla generation model, verify the output, and export accepted samples.

## Why this change

The previous symbolic edit strategy was controlled and reproducible, but many edits were too small to be easy for humans to notice in long Bengali financial articles. The new approach keeps the control but changes the realization step: we still decide what claim should change, but a generator rewrites the affected sentence or paragraph so the misinformation is more human-legible.

This is a controlled semantic editing pipeline, not free-form generation. The planner decides the claim, the rewriter realizes the change, and the verifier decides whether the result is acceptable.

## Recommended model path

Start with a small seq2seq Bangla generator on Kaggle:

- `csebuetnlp/banglat5`
- `Vacaspati/BanglaByT5`

Use the smaller, cleaner model first. If the rewrite quality is too subtle or too generic, move to a stronger instruction model later, but keep it constrained to the selected claim. The generation model is for rewriting only, not for free-form article invention.

## Kaggle notebook stages

### 1. Environment setup

- Install `transformers`, `torch`, `sentencepiece`, `accelerate`, `datasets`, `pandas`, `zstandard`, and any Bangla normalization dependency required by the model card.
- Detect Kaggle paths:
  - `/kaggle/input`
  - `/kaggle/working`
- Load the source CSV or compressed CSV from the Kaggle dataset mount.

### 2. Claim extraction

- Split each article into sentences.
- Identify candidate propositions:
  - numeric claims
  - policy claims
  - entity claims
  - temporal claims
  - causal claims

### 3. Claim selection

- Score each extracted claim before generation.
- Suggested selection features:
  - centrality
  - financial importance
  - editability
  - diversity
  - factuality
  - rewrite feasibility
- Store the component scores as `importance_score`, `editability_score`, and `diversity_bonus`.
- Combine them into a selection policy score.
- Prefer the highest-scoring claim, subject to family balance, rewrite feasibility, and duplicate avoidance.

### 4. Claim planning

- Assign a rewrite family.
- Build a planner output before text generation.
- Store the plan as structured metadata:
  - `sample_id`
  - `original_id`
  - `rewrite_family`
  - `target_sentence_index`
  - `target_span`
  - `desired_change`
  - `difficulty`
  - `importance_score`
  - `editability_score`
  - `diversity_bonus`
  - `expected_scope`
  - `expected_changed_claim`

### 5. Controlled rewriting

- Send the original article and the rewrite plan to the rewriter.
- Rewrite only the target sentence or a single local paragraph.
- Never rewrite the full article.
- Preserve:
  - article style
  - surrounding context
  - topic
  - overall length as much as possible
- Keep the rewriter separate from the planner, even if both stages use the same base model.

### 6. Multi-stage verification

The verifier runs after generation and before acceptance as a staged module.

Stage 1: claim integrity

- Did the intended claim change?
- Did any extra claim change?
- Can the changed span be localized to the planned sentence or paragraph?

Stage 2: surface quality

- Is the rewrite fluent Bangla?
- Does it preserve journalistic style?
- Is it too close to the source to be visually trivial?

Stage 3: semantic quality

- Does the rewrite remain a believable Bangla financial news item?
- Does the contradiction score support the intended label?
- Are there duplicate or near-duplicate artifacts?

Suggested automatic checks:

- claim-level before/after comparison
- span-locality check
- semantic similarity
- contradiction score
- fluency score
- duplicate detection

### 7. Regeneration loop

- If verification fails, regenerate up to 3 attempts.
- Keep the best passing candidate.
- If all attempts fail, drop the sample and log the failure reason.

### 8. Export

Write out:

- `finfact_bd_rewritten_raw.csv`
- `finfact_bd_rewritten_filtered.csv`
- `metadata.json`

Keep provenance fields so every row can be traced back to:

- source article
- target claim
- claim selection score
- component scores
- generator model
- rewrite plan
- verification result
- regeneration attempts

## Suggested notebook layout

```text
1. Install / imports
2. Load data
3. Build claim plans
4. Load generation model
5. Generate rewrites
6. Verify and filter
7. Regenerate failed cases
8. Save outputs
9. Build human validation pack
```

## Prompt shape

If we use an instruction model, keep the prompt narrow:

```text
Rewrite only the targeted claim in this Bangla financial article.
Rules:
- Preserve journalistic style.
- Change exactly one factual proposition.
- Keep the rest of the article coherent.
- Do not add unrelated facts.
- Return only the rewritten article.
```

## What this is not

- Not free-form fake news generation
- Not an unconstrained paraphraser
- Not a token-replacement script
- Not a classifier
- Not article-level rewriting without a plan

The generator is an executor inside a larger planning and verification pipeline.

## Acceptance criterion

The plan is good enough if the generated samples are:

- visibly different to a human reader
- still fluent Bangla news text
- controlled enough to analyze by perturbation family
- locally coherent around exactly one intended factual change
- reproducible on Kaggle
