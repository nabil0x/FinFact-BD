# Dataset-Driven Pipeline Configuration

This note records the July 15, 2026 configuration audit for
`data/finfact_bd/finfact_bd_originals.csv`.

## Dataset Profile

- Rows: `10,000`
- Empty articles: `0`
- Duplicate article IDs: `0`
- Label distribution: `10,000` original articles
- Split distribution: `8,001` train, `999` validation, `1,000` test
- Median article length: `221` whitespace tokens, `14` sentences
- 90th percentile article length: `531` whitespace tokens, `39` sentences
- 95th percentile article length: `636` whitespace tokens, `48` sentences

The dataset is strongly front-loaded. Using the full 80-sentence heuristic pass,
selected claims had this position distribution:

- sentence `0`: `44.29%`
- within first `3` sentences: `75.82%`
- within first `5` sentences: `88.15%`
- within first `10` sentences: `94.12%`

## Heuristic Candidate Coverage

The heuristic extractor already finds claim candidates for nearly every article.

| max_sentences | min_confidence | coverage | avg claims/article | no-claim articles |
|---:|---:|---:|---:|---:|
| 5 | 0.35 | 0.9891 | 3.42 | 109 |
| 8 | 0.35 | 0.9927 | 4.49 | 73 |
| 10 | 0.35 | 0.9940 | 5.03 | 60 |
| 12 | 0.35 | 0.9947 | 5.51 | 53 |
| 15 | 0.35 | 0.9954 | 6.15 | 46 |
| 20 | 0.35 | 0.9965 | 6.98 | 35 |
| 30 | 0.35 | 0.9974 | 8.01 | 26 |
| 80 | 0.35 | 0.9977 | 8.94 | 23 |

The gain from scanning 80 sentences instead of 20 is only `12` additional
articles with a claim candidate out of `10,000`, while full LLM extraction is
orders of magnitude slower.

## Recommended Default

The default Kaggle/research-run configuration should use:

```yaml
claim_extraction:
  backend: "heuristic"
  min_confidence: 0.35
  min_sentence_chars: 18
  max_sentences: 20

planner:
  backend: "llm_json"
  max_new_tokens: 384
```

Rationale:

- Heuristic extraction gives `99.65%` article coverage at `max_sentences=20`.
- It removes the observed `~3-4 minutes/article` Qwen extraction bottleneck.
- Qwen remains in the pipeline for structured rewrite planning.
- The claim selection stage still ranks by importance, editability, locality,
  verification feasibility, and risk.
- `configs/rewrite_pipeline_llm_extraction.yaml` preserves the slower full LLM
  extraction profile for ablations or reviewer-requested comparisons.

## Expected Impact

The Kaggle log showed Qwen extraction plus planning taking roughly `35` minutes
for `9` articles. Most of that time was extraction. Switching to heuristic
extraction should reduce the Qwen phase to planning-only latency, typically tens
of seconds per article rather than minutes.
