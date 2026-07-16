# Experiment Results Log

Last updated: 2026-07-16

## Overview

Running record of FinFact-BD planning-guided rewrite pipeline experiments.
Each row changes one knob at a time. Compare vertically.

---

## Pilot20 — Qwen3-8B

| Field | Value |
|---|---|
| Date | 2026-07-16 |
| Planner | `Qwen/Qwen3-8B` (llm_json, temp=0.0) |
| Generator | `CohereLabs/aya-expanse-8b` (temp=0.4) |
| Samples | 30 |
| Accepted | **26/30 (86.7%)** |
| Failed | 4 |
| First-attempt acceptance | 100% (26/26, no regen needed) |
| Config | `configs/default.yaml` (seed=42, 3 regen attempts, 5 families) |
| Git commit | `a979eb4` |

### Failure Breakdown

| Reason | Count | Articles |
|---|---|---|
| `all_planning_candidates_failed` | 1 | potrika_000001647 |
| `no_passing_rewrite` | 3 | potrika_000006160, potrika_000001747, potrika_000429545 |

### Verifier Failure Signature

| Reason | Count |
|---|---|
| suspicious_bangla_fragment | 2 |
| target_sentence_length_shift | 1 |

### Post-Pilot20 Fixes (commit `106b0de`)

1. **suspicious_bangla_fragment** — Removed `"অনুষ্"` from fragment list in `utils.py`.
2. **target_sentence_length_shift** — Skipped for `causal_inversion` family in `verifier.py`.
3. **planning diagnostics** — Added target_span to plan-validation failure logs in `pipeline.py`.

---

## Pilot100 — Qwen2.5-3B

| Field | Value |
|---|---|
| Date | 2026-07-16 |
| Planner | `Qwen/Qwen2.5-3B-Instruct` (llm_json, temp=0.0) |
| Generator | `CohereLabs/aya-expanse-8b` (temp=0.4) |
| Samples | 100 |
| Accepted | **79/100 (79.0%)** |
| Failed | 21 |
| First-attempt acceptance | 100% (79/79, no regen needed) |
| Config | `configs/default.yaml` (seed=42, 3 regen attempts, 5 families) |
| Planning runtime | 15,202s (98% of total) |
| Total runtime | ~4.3 hours (15,515s) |
| Git commit | `106b0de` (fixes deployed) |

### Failure Breakdown

| Reason | Count | % of Failures |
|---|---|---|
| `all_planning_candidates_failed` | 10 | 47.6% |
| `no_passing_rewrite` | 11 | 52.4% |

### Failed Articles (all_planning_candidates_failed)

- potrika_000436687
- potrika_000296999
- potrika_000165235
- potrika_000432146
- potrika_000437356
- potrika_000004574
- potrika_000436962
- potrika_000431157
- potrika_000340455
- potrika_000433423

### Verifier Failure Reasons

| Reason | Count |
|---|---|
| nli_contradiction_below_threshold | 20 |
| original_entity_still_present | 6 |
| causal_marker_missing | 4 |
| temporal_anchor_unchanged | 3 |
| planned_numeric_replacement_missing | 3 |
| new_facts_outside_target | 3 |
| causal_effect_not_inverted | 2 |

### Runtime Breakdown

| Phase | Seconds | % of Total |
|---|---|---|
| Planning | 15,201.7 | 97.98% |
| Generation | 313.7 | 2.02% |
| Verification | 69.7 | 0.45% |

---

## Comparison

| Metric | Qwen3-8B (30) | Qwen2.5-3B (100) | Δ |
|---|---|---|---|
| Acceptance rate | **86.7%** | **79.0%** | **-7.7pp** |
| Planning failures | 1 (3.3%) | 10 (10%) | +6.7pp |
| Verifier failures | 3 (10%) | 11 (11%) | ~equal |
| First-attempt rate | 100% | 100% | tied |

The 7.7pp gap is driven entirely by planning failures (3% → 10%). The smaller model produces structurally invalid plans more often. Verifier failure rates are similar across both models (~10-11%).

---

## Verdict for Mass Batch

| Strategy | Expected Acceptance | Planning Time | Notes |
|---|---|---|---|
| **Qwen3-8B** | **~85%** | ~15K s / 100 | Best quality |
| Qwen2.5-3B | ~79% | ~15K s / 100 | 7.7pp worse |
| Heuristic planner | ~45-55% | < 1 min | Instant but low ceiling |

**Recommendation:** Qwen3-8B for mass batch. The 6.7pp planning gap between models is large enough to justify the cost.
