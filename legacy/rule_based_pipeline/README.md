# Legacy Rule-Based Pipeline Archive

This directory preserves the previous FinFact-BD symbolic perturbation system
for historical comparison only.

It contains:

- the monolithic rule-based perturbation generator
- regex/numeric/entity/policy substitution logic
- old BENI extraction utilities
- old rule/NLI filtering scripts and notebooks
- old tests for the symbolic pipeline
- old configuration files

The active package under `src/generation/` must not import anything from this
archive. New work should use the controlled claim-level LLM rewriting pipeline.

Manual historical run example:

```bash
PYTHONPATH=legacy/rule_based_pipeline/src \
  python legacy/rule_based_pipeline/src/generation/perturbation_pipeline.py
```

The archive may still require the original local data layout and dependencies
used by the old experiments.
