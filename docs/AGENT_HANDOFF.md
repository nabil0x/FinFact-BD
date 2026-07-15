# Agent Handoff: FinFact-BD Rewrite Pipeline

Last updated: 2026-07-15

## Current State

The repository is on `main`. Recent important commits before the plan-review work:

```text
205f180 Align numeric rewrite validation with rulebook
36b72b5 Enrich rewrite family planning examples
3b20776 Improve rule-guided rewrite quality gates
```

The latest completed 10-sample run should be treated as a prompt/rule-example plus verifier-alignment experiment. The current plan-review changes add one reviewer pass before generation.

## What Changed Recently

### Rulebook and Planning

- Added `docs/rewrite_rule_instructions.md` as the human-editable rewrite rulebook.
- Updated planning priority to:
  1. `numerical_fact`
  2. `causal_inversion`
  3. `entity_replacement`
  4. `temporal_shift`
  5. `policy_reversal`
- Expanded `FAMILY_RULES` in `src/generation/prompts.py` with stronger good/bad examples.
- Bumped prompt version to `planning-guided-v6-plan-review`.

### Rewrite Quality Controls

- Numerical rewrites now require meaningful scale contradiction.
- Date/fiscal-year edits are routed away from `numerical_fact` toward `temporal_shift`.
- Entity replacements reject same-role peer swaps.
- Causal rewrites are expected to flip economic logic, not merely swap clauses.
- Text quality artifacts such as broken Bangla fragments and replacement characters are rejected.

### Entity Lexicon

- Expanded entity coverage in `src/generation/utils.py`.
- Current exact entity count: about `349`.
- Current role groups: `14`.
- Added coverage for banks, regulators, ministries, state corporations, insurers, company groups, and research/training bodies.

## Validation Status

Latest local validation before push:

```bash
pytest -q
```

Result:

```text
35 passed
```

## Current Kaggle Experiment

The intended current run is a 10-sample pilot after committing the plan-review changes.

Correct command:

```bash
python scripts/kaggle_run.py pilot \
  --config configs/rewrite_pipeline.yaml \
  --output-dir data/generated/rewrite_generation_pilot10_plan_review \
  --num-samples 10 \
  --seed 42 \
  --log-level INFO
```

Important: `kaggle_run.py` does not have a `generate` command. Use `pilot`, `smoke`, `stress1k`, `full`, or `resume`.

After the run finishes:

```bash
python scripts/kaggle_run.py inspect \
  --output-dir data/generated/rewrite_generation_pilot10_plan_review
```

```bash
python scripts/kaggle_run.py metrics \
  --output-dir data/generated/rewrite_generation_pilot10_plan_review \
  --log logs/rewrite_pilot.log \
  --write
```

```bash
python scripts/kaggle_run.py failure-audit \
  --output-dir data/generated/rewrite_generation_pilot10_plan_review \
  --write
```

Do not run `metrics` or `failure-audit` before generation completes because the output directory and checkpoint will not exist.

## Compare Against Previous Run

Previous baseline output:

```text
data/generated/rewrite_generation_pilot10_human
```

Next plan-review output:

```text
data/generated/rewrite_generation_pilot10_plan_review
```

Compare:

- accepted count
- failure reasons
- `planned_numeric_replacement_missing`
- `numeric_value_unchanged`
- `nli_contradiction_below_threshold`
- `repeated_bangla_fragment`
- quality of accepted samples in `human_validation.xlsx`

## Expected Result Pattern

The latest prompt examples should improve plan quality. Expected changes:

- fewer weak numerical changes
- fewer same-role entity swaps
- fewer value-equivalent numeric plans
- better human-readable target/replacement choices

Acceptance rate may improve only moderately on 10 samples. If accepted count stays low but failure reasons become cleaner, that is still useful progress.

## Latest Completed 10-Sample Result

Output directory:

```text
data/generated/rewrite_generation_pilot10_rule_examples
```

Summary:

```text
input_articles: 10
accepted: 5
failed: 5
acceptance_rate: 0.50
planning_seconds: 287.5959
generation_seconds: 148.7275
total_runtime_seconds: 436.3234
average_attempts_per_article: 1.75
first_attempt_acceptance_rate: 1.0
oom_recoveries: 0
peak_gpu_memory_mb: 6020
```

This is an improvement over the earlier 10-sample human pilot, which accepted 3/10.

Accepted examples include stronger numerical contradictions:

- `৫০ লাখ টাকার` -> `৫ কোটি টাকার`
- `৫৬ টাকার` -> `৫ কোটি টাকার`
- `১ হাজার ৩১৬ মার্কিন ডলার` -> `১ কোটি ৩১৬ মার্কিন ডলার`

Main failures:

- `potrika_000009175`: planner exception, same-role entity replacement.
- `potrika_000006739`: planner exception, numerical replacement not a significant scale contradiction.
- `potrika_000007323`: NLI contradiction remained too low for `১শ’টির বেশি` -> `১০০ শতাংশ`.
- `potrika_000274505`: policy reversal was semantically contradictory, but generated sentence was too long/awkward and failed `target_sentence_length_shift`.
- `potrika_000005717`: deterministic numeric rewrite was falsely flagged for `suspicious_bangla_fragment` because the unchanged source contains `কনসালটেন্সি`, which includes the suspicious substring `কনসা`.

Interpretation:

- Prompt examples improved planning quality and acceptance rate.
- Remaining issues are mostly verifier/planner alignment:
  - planner should use high-contrast numeric contradictions intentionally rather than weak/value-equivalent changes,
  - policy reversals need controlled phrase replacement when target/replacement are known,
  - artifact detector should compare changed target sentence only or avoid overbroad substrings,
  - same-role entity exception is working as intended but should trigger fallback planning rather than immediate article failure.

## Post-Run Fixes Implemented Locally

The following fixes were implemented after the 5/10 pilot result and should be committed before the next Kaggle run:

- Added numeric contradiction validation with user-approved high-contrast rules:
  - count facts should remain count facts, e.g. `১শ’টির বেশি` -> `২শ’টির বেশি` or `৫শ’টির বেশি`;
  - allows bare price -> crore/lakh when it is an intentional scale contradiction, e.g. `৫৬ টাকার` -> `৫ কোটি টাকার`;
  - rejects count -> percentage, e.g. `১শ’টির বেশি` -> `১০০ শতাংশ`;
  - still rejects incoherent money -> percentage changes, e.g. `৫৬ টাকার` -> `১০০ শতাংশ`;
  - still rejects weak/value-equivalent numerical changes.
- Added deterministic numeric contradiction override after NLI scoring:
  - NLI is still called for lifecycle/timing observability;
  - strong numeric contradictions can pass even when Bangla NLI under-scores the contradiction.
- Added controlled exact replacement for `policy_reversal`, so clean phrase plans do not go through Aya and drift into long awkward sentences.
- Narrowed the artifact detector:
  - valid `কনসালটেন্সি` no longer triggers `suspicious_bangla_fragment`;
  - isolated/truncated `কনসা` is still rejected.
- Added LLM planner validation repair:
  - if the first JSON plan is syntactically valid but fails validation, Qwen gets one repair prompt with the validation error.
- Added optional LLM plan review:
  - a valid plan is reviewed as `pass` or `repair` before generation;
  - reviewer repairs are validated deterministically before use;
  - malformed reviewer output falls back to the already-valid plan instead of failing the article.
  - current configs set `plan_review_attempts: 1`; set it to `0` for faster large runs.
- After the 8/10 plan-review pilot, tightened count handling:
  - `১শ’টির বেশি` -> `১০০ শতাংশ` is now rejected;
  - count rewrites should use count expressions such as `২শ’টির বেশি` or `৫শ’টির বেশি`;
  - scaled percentage phrases such as `১ কোটি শতাংশ` are rejected as incoherent;
  - validation repair now gets two attempts and explicitly avoids same-role entities and count percentages.

Validation after these local fixes:

```text
35 passed
```

## Do Not Change Mid-Run

If a Kaggle run is already active, do not change prompts, verifier rules, or configs until the run finishes. Otherwise the run becomes hard to interpret.

## Likely Next Engineering Step

After reviewing the 10-sample result, the next high-value work is verifier alignment:

- Add deterministic policy-direction checks.
- Add stronger causal-effect inversion checks.
- Add family-specific NLI handling instead of globally lowering the NLI threshold.
- Keep the verifier conservative; do not make it globally permissive.

Relevant files:

- `src/generation/verification_rules.py`
- `src/generation/verifier.py`
- `src/generation/utils.py`
- `tests/test_refinement_quality_gates.py`

## Notes for Future Agents

- Preserve the staged model-loading architecture.
- Avoid changing generated outputs or deleting Kaggle output directories unless explicitly requested.
- Use `pytest -q` before committing.
- If adding new prompt rules, update `PROMPT_VERSION`.
- If changing verifier behavior, add focused regression tests for the new failure/acceptance rule.
