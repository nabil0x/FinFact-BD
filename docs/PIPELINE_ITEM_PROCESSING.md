# FinFact-BD Pipeline Item Processing

This document explains how one source article is processed end-to-end by the active FinFact-BD rewrite pipeline.

The pipeline works on a single item at a time, where an item is one authentic Bangla financial article. The article is transformed into a claim-level rewrite candidate, passed through planning, rewritten locally, verified, and either accepted or rejected. Accepted samples are exported with full provenance and checkpointed so interrupted runs can resume safely.

## 1. What The Pipeline Receives

The input row is an article record, usually loaded from `data/finfact_bd/finfact_bd_originals.csv` or an equivalent CSV export. Each record contains:

- `article_id`
- `headline`
- `text`
- any available metadata

In code, this becomes an `Article` object from `src/generation/metadata.py`:

```python
Article(
    article_id,
    headline,
    text,
    metadata,
)
```

The pipeline never rewrites the whole article from scratch. It selects one factual claim inside the article, changes that claim, and then reconstructs the full article by splicing the rewritten sentence back into the original text.

## 2. Stage-by-Stage Walkthrough

The active orchestration lives in `src/generation/pipeline.py` under `PlanningGuidedRewritePipeline`.

### Stage 1: Article Loading

The run starts in `scripts/run_rewrite_pipeline.py`, which loads the YAML config and passes it into the pipeline.

The pipeline then:

1. Reads the input article file.
2. Applies the requested sample limit and offset, if present.
3. Restores any previous `checkpoint.json` state so a partially completed run can resume.
4. Loads any `planned_articles.jsonl` cache so deterministic planning work is not repeated.

This stage is about run management, not rewriting yet. The goal is to prepare a clean list of pending articles.

### Stage 2: Sentence Splitting

Each article text is split into sentence spans using the shared sentence segmentation helpers in `src/generation/utils.py`.

Each span stores:

- sentence index
- character start offset
- character end offset
- sentence text

This matters because later stages need to know exactly which sentence can be changed and where that sentence sits inside the full article.

### Stage 3: Claim Extraction

The extractor converts sentences into factual claim candidates.

Default behavior:

- `HeuristicClaimExtractor` is used in the active configuration.
- `LLMClaimExtractor` is available for the optional LLM extraction profile.

The extractor inspects each sentence and records:

- `sentence_index`
- `sentence`
- `claim_text`
- `claim_type`
- `entities`
- `numbers`
- `policies`
- `dates`
- `confidence`

### What the extractor is looking for

The heuristic extractor classifies a sentence into one of the supported claim types:

- `numerical`
- `entity`
- `temporal`
- `causal`
- `policy`
- `other`

It relies on sentence-level features such as:

- numeric expressions
- named entities
- policy-related terms
- dates and temporal phrases
- causal terms
- financial language cues

Claims below the minimum confidence threshold are dropped. This keeps weak or noisy sentences out of the planning stage.

### Stage 4: Claim Ranking

The claim ranker scores each extracted claim and applies a quality gate.

In `src/generation/claim_selection.py`, each claim receives:

- `importance_score`
- `editability_score`
- `verification_score`
- `locality_score`
- `risk_score`
- `overall_score`

The ranker keeps only claims that satisfy the minimum overall score and maximum risk score configured in the YAML.

The practical effect is:

- strong claims survive
- weak or risky claims are dropped
- the pipeline only plans rewrites on claims that can likely be changed locally and verified cleanly

If no claim passes the gate, the article is rejected at planning time and a failure record is written.

### Stage 5: Rewrite Planning

The selected ranked claim is converted into a structured `RewritePlan`.

The plan contains:

- `family`
- `target_claim`
- `edit_instruction`
- `edit_scope`
- `expected_change`
- `verification_constraints`
- `target_span`
- `replacement`

Supported families in the active codebase:

- `numerical_fact`
- `entity_replacement`
- `temporal_shift`
- `policy_reversal`
- `causal_inversion`

The planner can be heuristic or LLM-based depending on configuration. In the main pipeline, the LLM planner generates a JSON plan and then the plan is validated before rewrite.

### How the plan is chosen

The planner decides:

- which span inside the sentence should change
- what the new span should be
- what kind of factual distortion is being created
- what must remain unchanged

The planner does not produce the final article. It only produces a structured instruction for the generator.

### Plan validation

Before generation starts, the pipeline checks that the plan is executable.

Typical plan checks include:

- target span is present
- replacement is not empty
- target span is not identical to replacement
- target sentence index exists in the article
- the plan is locally realizable in the source article

If the plan fails validation, the pipeline can ask for a repaired plan or try a fallback family depending on the planner configuration.

### Stage 6: Constrained Local Generation

The generator realizes the plan in fluent Bangla.

In the active pipeline, `RewriteGenerator` is responsible for this step.

There are two generation paths:

1. `controlled_rewrite` for families that can be realized by exact local replacement
2. prompt-based generation when exact control is not sufficient

For the common families, the pipeline often performs exact replacement directly inside the target sentence.

### Controlled rewrite behavior

For `numerical_fact`, `temporal_shift`, `entity_replacement`, and `policy_reversal`, the generator may:

- take the original article
- locate the target sentence
- replace only the target span
- preserve the rest of the article unchanged

`entity_replacement` is special because linked mentions can appear in more than one sentence. In that case, the generator may replace linked occurrences more broadly when the verifier allows that behavior.

### Output of generation

The generator returns a `GeneratedRewrite` object containing:

- `rewritten_article`
- the prompt used
- generation parameters such as model name, revision, temperature, seed, attempt count, and max tokens

If the rewritten article is unchanged or contains artifacts, the generation attempt is rejected and the pipeline moves to regeneration.

### Stage 7: Verification

Verification decides whether a generated rewrite is acceptable.

The verifier stack is modular and typically reports:

- `intended_change`
- `locality`
- `text_quality_artifacts`
- `semantic_similarity`
- `contradiction`
- `journalistic_style`
- `hallucination`
- `duplicate_detection`

Each verifier returns:

- score
- pass/fail flag
- reason
- optional details

### What each verifier checks

- `intended_change`: did the planned change actually happen?
- `locality`: did only the target sentence change?
- `text_quality_artifacts`: are there broken fragments, invalid characters, or suspicious truncation artifacts?
- `semantic_similarity`: is the full article still close enough to the source?
- `contradiction`: does the rewrite actually contradict the original claim in the intended way?
- `journalistic_style`: does the text still read like plausible journalistic Bangla?
- `hallucination`: did the rewrite introduce extra facts outside the target sentence?
- `duplicate_detection`: is this rewrite too similar to a previously accepted sample?

If any verifier fails, the sample is not accepted.

### Stage 8: Regeneration

Rejected samples are not silently relaxed into acceptance.

Instead, the `RegenerationController` tries again up to the configured maximum number of attempts.

Across attempts it may:

- slightly increase temperature
- retry generation with the same plan
- re-run verification on the new candidate
- split verification batches if CUDA memory pressure appears

The controller records each attempt in an `AttemptRecord` so the full failure history is preserved.

If all attempts fail, the article is discarded and a failure entry is stored in the checkpoint.

### Stage 9: Acceptance and Export

When a sample passes verification, it is stored as a `SampleRecord`.

The exported row contains:

- `sample_id`
- `article_id`
- `headline`
- `original_article`
- `rewritten_article`
- `selected_claim`
- `claim_index`
- `claim_type`
- `perturbation_family`
- `rewrite_plan`
- `generator_model`
- `model_revision`
- `prompt_version`
- `temperature`
- `seed`
- `verification_scores`
- `regeneration_attempts`
- `timestamp`

The exporter writes:

- `finfact_bd_rewritten.jsonl`
- `finfact_bd_rewritten.csv`
- `metadata.json`
- `human_validation.xlsx` when enabled

## 3. What Happens To A Single Article

The short version is:

1. The article is loaded.
2. The article is split into sentences.
3. Candidate claims are extracted.
4. Claims are ranked and filtered.
5. One claim is selected.
6. A rewrite plan is produced.
7. The target sentence is rewritten locally.
8. Verification checks the rewrite.
9. If it fails, generation is retried.
10. If it passes, the sample is exported.

The article is either:

- accepted as a benchmark sample
- rejected with a logged reason

No sample is accepted without passing verification.

## 4. Typical Failure Points

This is useful when reading logs or audit outputs.

### No ranked claim

The extractor found claims, but none passed the quality gate.

Common causes:

- weak claim confidence
- high risk
- low locality
- low editability

### Planning failure

The planner produced a plan that could not be validated.

Common causes:

- missing target span
- identical target and replacement
- wrong claim family for the claim type
- malformed JSON from the planner

### Generation failure

The generator could not produce a valid localized rewrite.

Common causes:

- empty rewrite
- sentence artifacts
- output that does not contain the planned replacement

### Verification failure

The rewrite was generated, but at least one verifier rejected it.

Common causes:

- extra sentence changed
- original claim still present
- semantic similarity too low
- NLI contradiction score below threshold
- new facts outside the target sentence

## 5. Why The Pipeline Is Structured This Way

The pipeline intentionally separates cognitive roles:

- extraction finds candidate claims
- ranking chooses the strongest candidate
- planning decides the factual distortion
- generation realizes the edit
- verification decides acceptance

This separation is important because it keeps the benchmark auditable. Every accepted sample can be traced back to:

- the source article
- the selected claim
- the rewrite plan
- the generated output
- the verifier decisions
- the regeneration history

## 6. Where To Look In The Code

Key files:

- `src/generation/pipeline.py`
- `src/generation/claim_extraction.py`
- `src/generation/claim_selection.py`
- `src/generation/perturbation_planner.py`
- `src/generation/rewrite_generator.py`
- `src/generation/verifier.py`
- `src/generation/regeneration.py`
- `src/generation/exporter.py`
- `src/generation/metadata.py`
- `scripts/run_rewrite_pipeline.py`

If you want to trace one sample in detail, start with the exported JSONL row and work backward through:

- `selected_claim`
- `rewrite_plan`
- `verification_scores`
- `regeneration_attempts`

That is the shortest path from the exported benchmark sample back to the original article.
