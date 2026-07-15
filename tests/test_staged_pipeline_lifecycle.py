from __future__ import annotations

import csv

from src.generation.models import ModelBundle
from src.generation.pipeline import PlanningGuidedRewritePipeline


class ReleasableInstructionModel:
    model_name = "fake-qwen"
    model_revision = "test"

    def __init__(self) -> None:
        self.release_count = 0

    def generate_text(self, prompt: str, temperature: float, seed: int, max_new_tokens: int) -> str:
        if prompt.startswith("Extract factual claims"):
            return """
            {
              "claims": [
                {
                  "sentence_index": 0,
                  "sentence": "বাংলাদেশ ব্যাংক নীতিগত সুদের হার ১০ শতাংশ বাড়িয়েছে।",
                  "claim": "Bangladesh Bank raised policy interest rate to 10 percent",
                  "type": "numerical",
                  "entities": ["বাংলাদেশ ব্যাংক"],
                  "numbers": ["১০"],
                  "policies": ["সুদের হার"],
                  "dates": [],
                  "confidence": 0.95
                }
              ]
            }
            """
        return """
        {
          "family": "numerical_fact",
          "target_span": "১০ শতাংশ",
          "replacement": "৭ শতাংশ",
          "locality": "target_sentence",
          "edit_instruction": "Change only the selected rate from ১০ শতাংশ to ৭ শতাংশ.",
          "expected_change": "The interest-rate value changes from ১০ শতাংশ to ৭ শতাংশ.",
          "verification_constraints": {"preserve_all_other_sentences": true}
        }
        """

    def release(self) -> None:
        self.release_count += 1


class ReleasableGenerator:
    model_name = "fake-aya"
    model_revision = "test"

    def __init__(self) -> None:
        self.release_count = 0

    def generate_batch(self, prompts, temperatures, seeds, max_new_tokens):
        outputs = []
        for prompt in prompts:
            article = prompt.rsplit("Original complete article:", 1)[-1]
            article = article.rsplit("Complete rewritten article:", 1)[0].strip()
            outputs.append(article.replace("১০ শতাংশ", "৭ শতাংশ", 1))
        return outputs

    def release(self) -> None:
        self.release_count += 1


class StableEmbedder:
    model_name = "fake-e5"

    def __init__(self) -> None:
        self.release_count = 0
        self.encode_batch_sizes = []

    def encode(self, texts):
        self.encode_batch_sizes.append(len(texts))
        return [[1.0, float(len(text)), 0.5] for text in texts]

    def release(self) -> None:
        self.release_count += 1


class PassingNLI:
    model_name = "fake-nli"

    def __init__(self) -> None:
        self.release_count = 0
        self.batch_calls = 0

    def contradiction_score(self, premise: str, hypothesis: str) -> float:
        return 0.9 if premise != hypothesis else 0.0

    def contradiction_scores(self, premises, hypotheses):
        self.batch_calls += 1
        return [self.contradiction_score(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]

    def release(self) -> None:
        self.release_count += 1


class PassingFluency:
    model_name = "fake-banglabert"

    def __init__(self) -> None:
        self.release_count = 0
        self.batch_calls = 0

    def perplexity(self, text: str) -> float:
        return 25.0

    def perplexities(self, texts):
        self.batch_calls += 1
        return [self.perplexity(text) for text in texts]

    def release(self) -> None:
        self.release_count += 1


def test_pipeline_releases_role_models_once_per_staged_run(tmp_path):
    input_csv = tmp_path / "input.csv"
    with open(input_csv, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["article_id", "headline", "text"])
        writer.writeheader()
        for article_id in ("a1", "a2"):
            writer.writerow(
                {
                    "article_id": article_id,
                    "headline": "বাংলাদেশ ব্যাংক সুদের হার বাড়িয়েছে",
                    "text": "বাংলাদেশ ব্যাংক নীতিগত সুদের হার ১০ শতাংশ বাড়িয়েছে। বাজার স্থিতিশীল আছে।",
                }
            )

    instruction = ReleasableInstructionModel()
    generator = ReleasableGenerator()
    embedder = StableEmbedder()
    nli = PassingNLI()
    fluency = PassingFluency()
    bundle = ModelBundle(
        generator=generator,
        embedder=embedder,
        nli=nli,
        fluency=fluency,
        extractor=instruction,
        planner=instruction,
    )
    config = {
        "paths": {"input_csv": str(input_csv), "output_dir": str(tmp_path / "out")},
        "pipeline": {"seed": 7},
        "input": {"id_column": "article_id", "headline_column": "headline", "text_column": "text"},
        "claim_extraction": {"backend": "llm_json", "min_confidence": 0.1},
        "claim_ranking": {"min_overall_score": 0.1, "max_risk_score": 1.0},
        "planner": {"backend": "llm_json", "allowed_families": ["numerical_fact"]},
        "generation": {"temperature": 0.0, "max_new_tokens": 256},
        "regeneration": {"max_attempts": 1, "temperature_step": 0.1},
        "verification": {
            "semantic_similarity_min": 0.0,
            "contradiction_min": 0.5,
            "fluency_max_perplexity": 100.0,
            "duplicate_max_similarity": 1.1,
        },
        "human_validation": {"enabled": False},
    }

    result = PlanningGuidedRewritePipeline(config, model_bundle=bundle).run()

    assert len(result.samples) == 2
    assert instruction.release_count == 1
    assert generator.release_count == 1
    assert embedder.release_count == 1
    assert nli.release_count == 1
    assert fluency.release_count == 1
    assert 2 in embedder.encode_batch_sizes
    assert nli.batch_calls == 1
    assert fluency.batch_calls == 1
    assert all("৭ শতাংশ" in sample.rewritten_article for sample in result.samples)
