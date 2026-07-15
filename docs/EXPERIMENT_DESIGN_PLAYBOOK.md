# Experiment Design Playbook

Last updated: 2026-07-16 (refined: failure signatures aligned to code, verifier stack documented)

## Purpose

This document is the operating plan for FinFact-BD experiments.
Use it when you want to:

- run a clean 10-sample or 20-sample pilot,
- compare prompt revisions,
- switch planner or generator models,
- test plan-review or repair changes,
- keep runs comparable instead of mixing too many changes at once.

The goal is to keep the experiment surface small:

- freeze the baseline controls,
- change one meaningful knob at a time,
- inspect failures before scaling,
- record the exact command and output directory for every run.

## What Should Stay Fixed

These should stay fixed within one experiment series unless the point of the run is to change them.

| Control | Recommended default | Why it should stay fixed |
|---|---|---|
| Input file | `data/finfact_bd/finfact_bd_originals.csv` | Avoids data drift across runs |
| Seed | `42` | Keeps article sampling and stochastic generation comparable |
| Output naming | `data/generated/rewrite_generation_<run_name>` | Keeps artifacts and audits easy to find |
| Verification thresholds | Current `configs/rewrite_pipeline.yaml` values (see Verifier Stack section below) | Prevents silent quality drift |
| Human validation workbook | Enabled | Keeps annotation output consistent |
| Family priority | `numerical_fact`, `causal_inversion`, `entity_replacement`, `temporal_shift`, `policy_reversal` | Keeps planning comparable across runs |
| Claim family definitions | The rulebook in `docs/rewrite_rule_instructions.md` | Prevents prompt drift from becoming data drift |
| Inspection flow | `inspect -> metrics -> failure-audit` | Keeps post-run analysis consistent |

## What You Can Change

Change only one of these per run series unless you are explicitly running a combined ablation.

| Knob | Typical values | What it tests |
|---|---|---|
| Planner model | `qwen25-3b`, `qwen3-8b` | Planning quality, speed, cost |
| Generator model | Current `Aya` baseline or a smaller generator you choose | Bangla realization quality and runtime |
| Extractor mode | Heuristic vs LLM extractor | Claim selection quality |
| Plan repair attempts | `1` or `2` | How robust the planner is to malformed JSON |
| Plan review attempts | `0` or `1` | Whether the reviewer improves plan quality or adds noise |
| Family examples | Rulebook examples, more examples, sharper examples | Whether the planner learns the intended rewrite style |
| Family-specific rules | Numerical, entity, temporal, policy, causal | Whether a family needs tighter constraints |
| Verifier thresholds | Only after a labeled audit | Whether the gate is too strict or too weak |
| Sample count | `5`, `10`, `20`, `100`, `1000` | Whether a change survives beyond a toy run |

## Recommended Experiment Ladder

Use this ladder instead of jumping directly to large runs.

### 1. Smoke

Use `5` samples when you only want to verify plumbing.

Purpose:

- check model loading,
- check prompt formatting,
- check checkpoint writing,
- check workbook export.

Pass condition:

- no crash,
- output files exist,
- logs show the expected model names.

### 2. Pilot-10

Use `10` samples when you are testing prompt or rule changes.

Purpose:

- compare rule examples,
- compare reviewer prompts,
- check whether a planner change reduces bad plans.

Pass condition:

- accepted samples are not obviously worse than the previous baseline,
- failure reasons become cleaner,
- no new catastrophic exception class appears.

### 3. Pilot-20

Use `20` samples when you want a more stable signal than a 10-sample run.

Purpose:

- compare planner presets,
- compare reviewer settings,
- check runtime overhead.

Pass condition:

- acceptance rate is stable enough to compare,
- failure audit shows the expected dominant reasons,
- no major increase in broken Bangla or multi-sentence leakage.

### 4. Stress-1000

Use this only after the smaller runs are stable.

Purpose:

- throughput,
- memory behavior,
- failure rate at scale,
- checkpoint recovery.

Pass condition:

- no OOM recovery cascade,
- checkpoints remain consistent,
- failure modes are already known from the smaller pilots.

## Model Selection Rules

The current pipeline has separate model roles.

Important rule:

- `--planner-preset qwen25-3b` changes the planner only.
- It does not change the generator.
- If you still see an 8B download, that is usually the generator, not the planner.

### Planner-only smaller model

Use this when you want to test lighter planning without changing realization.

```bash
python scripts/kaggle_run.py pilot \
  --planner-preset qwen25-3b \
  --num-samples 10 \
  --output-dir data/generated/rewrite_generation_pilot10_qwen25_3b
```

### Planner preset plus explicit generator override

Use this when you want to change the generator explicitly as well.

```bash
python scripts/kaggle_run.py pilot \
  --planner-preset qwen25-3b \
  --generator-model <smaller-or-alternate-generator-model> \
  --num-samples 10 \
  --output-dir data/generated/rewrite_generation_pilot10_small_stack
```

If you later introduce a smaller generator, put that model name in `--generator-model`.

### Wrapper script behavior

`scripts/kaggle_pilot20.sh` forwards extra arguments.

So this is valid:

```bash
scripts/kaggle_pilot20.sh --planner-preset qwen25-3b
```

### How to verify the loaded model

Check the log line written by `kaggle_run.py`:

```text
Using resolved config ... overrides=planner=Qwen/Qwen2.5-3B-Instruct
```

That is the strongest quick check that the override actually took effect.

## Run Protocol

Use the same order every time.

1. Run preflight for the model stack you plan to use.
2. Run a smoke or pilot generation job.
3. Inspect the output directory.
4. Run metrics.
5. Run failure audit.
6. Read the first few accepted examples and the first few failures.
7. Decide whether the next change should be prompt, model, verifier, or schema.

Suggested command sequence:

```bash
python scripts/kaggle_run.py preflight --stage load --planner-preset qwen25-3b
python scripts/kaggle_run.py pilot --planner-preset qwen25-3b --num-samples 10 --output-dir data/generated/rewrite_generation_pilot10_qwen25_3b
python scripts/kaggle_run.py inspect --output-dir data/generated/rewrite_generation_pilot10_qwen25_3b
python scripts/kaggle_run.py metrics --output-dir data/generated/rewrite_generation_pilot10_qwen25_3b --log logs/rewrite_pilot.log --write
python scripts/kaggle_run.py failure-audit --output-dir data/generated/rewrite_generation_pilot10_qwen25_3b --write
```

## What To Record After Every Run

Keep a small run sheet with the following fields:

- date,
- git commit,
- exact command,
- output directory,
- fixed controls,
- changed knob,
- accepted count,
- failed count,
- dominant failure reasons,
- one-sentence interpretation,
- next change.

This is more useful than keeping only the final metrics JSON.

## Verifier Stack

The pipeline runs these verifiers in order. Each returns a score, pass/fail flag, and reason string.

1. `intended_change` — deterministic check that the planned edit actually happened in the target sentence
2. `locality` — ensures only the target sentence changed (entity replacement allows linked-mention updates)
3. `text_quality_artifacts` — detects repeated Bangla fragments, dangling halants, replacement characters, and target sentence length drift
4. `semantic_similarity` — multilingual-e5 embedding cosine similarity against the original article
5. `contradiction` — mDeBERTa-XNLI contradiction score on the target sentence pair (with deterministic overrides for numeric and temporal families)
6. `fluency` — BanglaBERT ELECTRA perplexity on the rewritten article
7. `journalistic_style` — paragraph structure, punctuation, repeated characters, non-news register, headline topic drift
8. `hallucination` — new entities, numbers, dates, or organizations outside the target sentence
9. `duplicate_detection` — corpus-level embedding cosine similarity against previously accepted rewrites

Thresholds live in `configs/rewrite_pipeline.yaml` under `verification`.

## Failure-To-Fix Map

Use this map when the failure audit is noisy. The signatures below are the most actionable verifier reasons. The full set of reason strings is larger — run `failure-audit` and inspect `verifier_reasons` for the complete list.

| Failure signature | What it usually means | Next fix |
|---|---|---|
| `numeric_value_unchanged` | The rewrite changed wording, not the number | Use larger numeric contrast or reject the plan |
| `planned_numeric_replacement_missing` | Planner lost the numeric target | Tighten the planner prompt and repair path |
| `numeric_scale_change_too_weak` | The numeric change was too small to be meaningful | Require a larger scale jump in the planner prompt |
| `entity_same_role_replacement` | Entity swap was same-class or peer (e.g. bank-to-bank) | Expand the entity role map and reject peer swaps |
| `planned_entity_replacement_missing` | Planner specified a replacement that did not appear in the output | Tighten the controlled-realization splice logic |
| `original_entity_still_present` | The original entity was not removed from the target sentence | Enforce exact-span replacement before LLM fallback |
| `policy_direction_not_reversed` | The policy direction terms did not flip | Sharpen the policy reversal prompt examples |
| `causal_marker_missing` | The rewritten sentence lost the causal connector | Require causal markers in the rewrite prompt |
| `causal_effect_not_inverted` | The cause-effect relationship was not contradicted | Flip the effect, not just the sentence order |
| `temporal_anchor_unchanged` | The date or time expression was not shifted | Use exact-span replacement for temporal targets |
| `planned_temporal_replacement_missing` | Planner specified a time shift that did not appear | Tighten the temporal splice logic |
| `target_sentence_length_shift` | Rewrite drifted into a longer or fused sentence | Tighten the rewrite prompt and splice logic |
| `unexpected_sentence_changes` | The rewrite touched extra sentences | Enforce single-sentence output and hard locality checks |
| `repeated_bangla_fragment` | Generator produced broken Bangla or truncation | Add artifact filters before verifier calls |
| `nli_contradiction_below_threshold` | The factual flip was too weak | Make the plan more extreme or more semantically opposite |
| `new_facts_outside_target` | The generator hallucinated outside the target span | Restrict the rewrite to the target span only |
| `embedding_similarity_below_threshold` | The rewritten article drifted too far from the original | Tighten the locality of the rewrite |
| `perplexity_above_threshold` | The Bangla output is ungrammatical or noisy | Simplify the rewrite or increase temperature |
| `near_duplicate_in_corpus` | The rewrite is too similar to a previously accepted sample | Increase diversity or filter near-duplicates earlier |

## Recommended Current Baseline For Iteration

For prompt and reviewer work, use this as the default baseline:

- planner preset: `qwen25-3b`,
- generator: current Aya baseline,
- plan repair attempts: `2`,
- plan review attempts: `1`,
- seed: `42`,
- sample count: `10`,
- output dir: `data/generated/rewrite_generation_<name>`.

That setup is usually the best balance between speed and signal while the prompt and review logic are still changing.

## Next-Run Decision Rule

After each run, choose only one next action:

- update the rewrite rulebook,
- update the planner prompt,
- update the reviewer prompt,
- update the verifier thresholds,
- update the model profile.

Do not change all of them together unless you are deliberately doing a combined ablation.
