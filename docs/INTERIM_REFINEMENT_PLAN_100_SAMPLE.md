# Interim Refinement Plan After the 180-Article Kaggle Pilot

## Material Passport
| Field | Value |
|---|---|
| Project | FinFact-BD |
| Artifact | Interim refinement plan |
| Stage | Post-run analysis of 180-article Kaggle diagnostic pilot |
| Verification status | ANALYZED from Kaggle logs, not locally reproduced |
| Date | 2026-07-15 |
| Evidence source | User-provided Kaggle outputs from `data/generated/rewrite_generation_pilot180` |

## Executive Verdict
The 180-article pilot is a successful diagnostic run, but not a production-readiness pass.

The infrastructure works: all 180 articles were planned, exports were produced, checkpointing worked, GPU memory stayed stable, and no OOM recovery was needed. The scientific quality gates still need refinement before human validation or scale-up.

| Area | Verdict |
|---|---|
| End-to-end execution | Pass |
| Planning checkpointing | Pass |
| Export integrity | Pass |
| GPU memory stability | Pass |
| Acceptance rate | Weak but diagnostically useful |
| Intended-change verification | Needs correction |
| Locality preservation | Needs correction |
| NLI calibration | Needs manual audit |
| Human-validation workbook | Needs correction before annotation |
| Production readiness | Not ready |

## Run Summary
| Metric | Value |
|---|---:|
| Input articles | 180 |
| Planned articles | 180 |
| Accepted samples | 81 |
| Failed samples | 99 |
| Acceptance rate | 45.0% |
| Average attempts per article | 2.3222 |
| Average attempts for accepted samples | 1.4938 |
| First-attempt acceptance among accepted samples | 66.67% |
| Accepted after regeneration | 27 |
| Total runtime | 9,621.04 seconds, about 2h 40m |
| Articles per minute | 1.1225 |
| Accepted samples per minute | 0.5051 |
| Planning time | 2,887.94 seconds |
| Generation time | 6,733.10 seconds |
| Verification time | 148.44 seconds |
| Peak GPU memory | 8,836 MB |
| Peak CPU RAM | 2.8276 GB |
| OOM recoveries | 0 |

Interpretation: generation and regeneration dominate runtime. Verification is not the bottleneck. The 45% acceptance rate is low, but usable for diagnosing fixable failure modes.

## Pipeline Health
| Component | Finding |
|---|---|
| Planning | Completed for all 180 articles |
| Generation | Completed with no crash-level failures |
| Verification | 418 candidate attempts verified |
| Checkpointing | `planned_articles.jsonl` contains 180 rows |
| Memory | No OOM recovery, peak GPU under 9 GB |
| Export | CSV, JSONL, metadata, checkpoint, workbook all produced |

The infrastructure is no longer the main concern. The core concerns are generator behavior, verifier calibration, and human-evaluation validity.

## Failure Distribution
All 99 failed articles failed with `no_passing_rewrite`.

| Verifier reason | Count | Approx. share of 418 attempts | Interpretation |
|---|---:|---:|---|
| `nli_contradiction_below_threshold` | 237 | 56.7% | Dominant rejection source |
| `unexpected_sentence_changes:*` | approx. 154 | 36.8% | Major locality failure source |

These counts are not mutually exclusive. Many failed attempts have both NLI and locality failures.

## Critical Finding 1: Intended-Change Verification Is Too Weak
The current intended-change check accepts a target sentence as changed when the string differs. That is not enough.

Example:

| Field | Text |
|---|---|
| Original | `বিশ্বের ১শ’টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।` |
| Rewritten | `বিশ্বের ১০০টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।` |
| Intended-change score | 1.0 |
| NLI contradiction | approx. 0.041 |

This is not misinformation. It is mostly orthographic normalization: `১শ’টির বেশি` and `১০০টির বেশি` mean essentially the same thing.

Implication: the NLI verifier correctly rejected this, but the intended-change verifier incorrectly treated it as a successful factual change. This is a P0 issue because the framework claims controlled factual distortion, not superficial string edits.

Required refinement:

- Intended-change verification must be plan-aware.
- Numerical changes must verify changed normalized numeric value.
- Policy reversal must verify changed polarity or direction.
- Entity replacement must verify a different valid entity appears.
- Temporal shift must verify a changed date or time anchor.
- Causal inversion must verify changed causal relation.

## Critical Finding 2: NLI Failures Mix True Rejections and Possible False Negatives
The NLI gate rejects many candidates, but the failure examples show mixed causes.

| Case | Interpretation |
|---|---|
| `potrika_000007323` | Correct rejection: surface normalization, not contradiction |
| `potrika_000274505` | Correct rejection: vague/incoherent policy edits |
| `potrika_000001960` | Correct rejection: later attempts effectively unchanged |
| `potrika_000296792` | Possible threshold-edge case: attempt 3 changes `৫৬` to `৫০`, NLI 0.5195 vs 0.55 threshold |
| `potrika_000434741` | Possible NLI weakness on subtle tax-rate direction changes |

Do not lower the contradiction threshold globally yet. First separate true generation failures from verifier false negatives.

Required audit:

- Label at least 50 NLI-rejected target pairs.
- Label at least 30 NLI-accepted target pairs.
- Stratify by perturbation family.
- Record NLI score distributions by family.
- Tune thresholds only on a held-out development subset.

## Critical Finding 3: Locality Failures Are Real
The failure audit contains broad locality failures such as:

```text
unexpected_sentence_changes:[1, 2, 3, 4, 5, 6]
unexpected_sentence_changes:[1, 2, ..., 34]
unexpected_sentence_changes:[4, 5, ..., 44]
```

Observed patterns:

| Pattern | Example behavior |
|---|---|
| Context leakage | Generated sentence includes following sentence content |
| Truncated Bangla | Output contains `�` |
| Sentence fusion | Target sentence absorbs the next article sentence |
| Repetition | `শাই ওয়ে শাই ওয়েনহাই` |
| Incomplete morphology | `অনুষ্` |

Implication: this is not just a verifier issue. The generator/localizer is allowing multi-sentence or corrupted output to be spliced into the article, which threatens the core locality claim.

Required refinement:

- Reject raw generator output containing `�`.
- Reject output containing more than one sentence boundary unless salvaging is explicitly safe.
- Extract the first valid rewritten sentence, not an index-aligned generated span.
- Enforce final sentence terminator when the original target sentence had one.
- Add a post-splice sanity check before expensive verification.

## Critical Finding 4: Fluency and Style Checks Are Too Permissive
Malformed outputs often receive fluency scores around 0.95-0.97 and journalistic style 1.0.

| Malformation | Observed example |
|---|---|
| Replacement character | `�` |
| Broken word | `অনুষ্` |
| Non-standard token | `ন঵াজনক` |
| Wrong/garbled currency | `টালা খর্চ` |
| Duplicated name fragment | `শাই ওয়ে শাই ওয়েনহাই` |

Required refinement:

- Add explicit artifact detection for replacement characters, repeated fragments, dangling stems, mixed-script corruption, and malformed currency expressions.
- Add a lightweight Bangla text-quality verifier before NLI.
- Treat artifact detection as a hard fail.

## Critical Finding 5: Planner and Claim Selection Still Select Bad Targets
Some failures originate before generation.

| Article | Problem |
|---|---|
| `potrika_000002090` | Rice mill inauguration planned as policy reversal with `কর` -> `বিল`, nonsensical |
| `potrika_000001960` | Smartphone bundle offer gets policy-style target `কর` -> `ট্যাক্স`, unrelated to the claim |
| `potrika_000007323` | Numeric plan changes surface form without changing factual value |

Implication: the pipeline is not secretly rule-based, but the heuristic extractor/ranker can feed poor claims into the planner. The planner can return formally valid JSON that is scientifically invalid.

Required refinement:

- Add plan validity checks before generation.
- Reject plans where `target_span` is not semantically central to the claim.
- Reject numeric plans where replacement is value-equivalent to the source.
- Reject policy plans where the source sentence lacks real policy/action polarity.
- Penalize promotional product announcements, event reports, and non-financial institutional news.

## Human Validation Readiness
The workbook is generated, but it is not ready for actual annotation.

Known issue: the workbook likely uses the original selected claim as `claim_focus` rather than the rewritten target claim. If true, annotators are not judging the generated misinformation claim.

Required refinement before any human study:

- Display the rewritten target sentence as `claim_focus`.
- Add `original_target_claim` to a hidden or separate audit sheet.
- Keep visible context from the rewritten article.
- Do not expose perturbation family, verification scores, or planner output to annotators.
- Add fields for plausibility, fluency/style, unrelated-context preservation, and hallucination.

Minimum human pilot after fixes:

| Item | Target |
|---|---|
| Accepted samples | 30-50 |
| Annotators | At least 2 |
| Primary label | `original`, `rewritten`, `not sure` |
| Secondary ratings | plausibility, fluency/style, hallucination, context preservation |
| Agreement | Cohen's kappa or Krippendorff's alpha |

## Accepted Refinement Priorities
### P0: Fix Human Validation Workbook
Required changes:

- Extract rewritten target sentence from `rewritten_article`.
- Use rewritten target sentence as annotator-facing `claim_focus`.
- Include original target claim only in provenance/audit fields.
- Add human-quality columns beyond `label`, `confidence`, and `justification`.

Reason: human validation is invalid if annotators inspect the wrong target claim.

### P0: Make Intended-Change Verification Plan-Aware
Required changes:

- Numerical verifier: compare normalized numeric values.
- Policy verifier: check polarity/direction markers.
- Entity verifier: verify target entity changed to a different entity.
- Temporal verifier: verify date/time anchor changed.
- Causal verifier: require explicit changed causal relation.

Reason: string-level target change is not evidence of factual misinformation.

### P0: Harden Sentence-Only Localizer
Required changes:

- Reject raw output containing `�`.
- Reject or sanitize repeated context echoes.
- Select the first valid generated sentence, not `generated_spans[target_index]`.
- Enforce target sentence terminator.
- Add unit tests for sentence fusion and local context echo.

Reason: broad `unexpected_sentence_changes` failures directly undermine locality.

### P1: Add Plan Validity Gate
Required checks:

- `target_span` must occur in the selected claim.
- `replacement` must not be value-equivalent to the target span.
- Family must match claim content.
- Policy reversal requires actual policy/action polarity.
- Numerical fact requires at least one numeric value and a changed normalized value.

Reason: generation should not start from scientifically invalid plans.

### P1: Calibrate NLI on a Manual Audit Set
Required audit:

- 50 NLI-rejected pairs.
- 30 NLI-accepted pairs.
- Stratify by family.
- Label whether intended factual change occurred.
- Estimate false-positive and false-negative behavior.

Reason: the current NLI threshold may reject valid Bangla financial edits, but the examples also show many true rejections.

### P1: Strengthen Artifact and Fluency Checks
Required checks:

- Replacement-character detection.
- Broken-token and dangling-stem heuristics.
- Repetition detection beyond simple repeated characters.
- Currency and numeric unit sanity checks.
- Optional Bangla language-quality classifier later.

Reason: malformed Bangla currently passes fluency/style gates.

### P2: Improve Claim Selection
Required changes:

- Penalize long, multi-clause target sentences more strongly.
- Penalize target sentences that contain multiple reported facts.
- Penalize quote-heavy or attribution-heavy sentences.
- Penalize event-announcement sentences unless financial factual content is central.
- Add family diversity only after reliability improves.

Reason: some failures originate before generation.

### P2: Pin Reproducibility Metadata
Required changes:

- Resolve exact Hugging Face commit hashes during preflight.
- Store model commit hashes in `metadata.json`.
- Store package versions, CUDA version, GPU type, config hash, prompt version, and git commit hash.

Reason: `revision: main` is not sufficient for an EMNLP-grade reproduction package.

## Decision Gates Before Next Scale-Up
Do not start a 1k stress test until these gates pass:

| Gate | Required evidence |
|---|---|
| Workbook correctness | Annotator-facing claim is rewritten target claim |
| Intended-change validity | Plan-aware verifier rejects surface-only edits |
| Locality stability | Broad `unexpected_sentence_changes` substantially reduced |
| Artifact filtering | Outputs with `�` or broken Bangla hard-fail |
| NLI calibration | Manual audit completed and threshold decision documented |
| Plan validity | Invalid plans rejected before Aya generation |

Suggested next sequence:

```text
Patch P0 issues
  -> 5-sample smoke
  -> 20-sample pilot
  -> 100-sample rerun
  -> failure audit
  -> human validation pilot
  -> 1k stress test
```

Do not jump from the current 180-run to 20k production.

## Updated Research Interpretation
The current system is genuinely more than a rule-based perturbation engine. It has structured planning, LLM sentence realization, regeneration, independent verification, and provenance.

The scientific claim should remain conservative:

```text
Planning-guided controlled claim rewriting under active validation.
```

It should not yet be described as:

```text
Validated large-scale Bangla financial misinformation generation benchmark.
```

## Current Readiness Assessment
| Scale | Status |
|---|---|
| Smoke test | Ready |
| 20-sample pilot | Ready |
| 100-sample rerun after P0 fixes | Recommended |
| Human validation pilot | Blocked until workbook fix |
| 1k stress test | Blocked until locality and intended-change fixes |
| 20k production run | Not ready |

## Final Interim Verdict
The 180-article pilot should be treated as a successful diagnostic run and a failed production-readiness run. The infrastructure works. The research pipeline still needs P0 refinement before the accepted samples can be trusted for human validation or used as evidence in an EMNLP paper.
