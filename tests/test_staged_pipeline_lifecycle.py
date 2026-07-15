from __future__ import annotations

import csv

from src.generation.metadata import (
    Article,
    Claim,
    PlannedArticle,
    RankedClaim,
    RewritePlan,
    VerificationReport,
    VerifierResult,
)
from src.generation.models import ModelBundle
from src.generation.planning_checkpoint import append_planned_article
from src.generation.pipeline import PlanningGuidedRewritePipeline
from src.generation.regeneration import RegenerationConfig, RegenerationController
from src.generation.rewrite_generator import RewriteGenerator


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


class ExplodingInstructionModel:
    model_name = "exploding-qwen"
    model_revision = "test"

    def generate_text(self, prompt: str, temperature: float, seed: int, max_new_tokens: int) -> str:
        raise AssertionError("planning checkpoint should skip instruction model calls")


def write_input_csv(path, article_ids=("a1", "a2")):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["article_id", "headline", "text"])
        writer.writeheader()
        for article_id in article_ids:
            writer.writerow(
                {
                    "article_id": article_id,
                    "headline": "বাংলাদেশ ব্যাংক সুদের হার বাড়িয়েছে",
                    "text": "বাংলাদেশ ব্যাংক নীতিগত সুদের হার ১০ শতাংশ বাড়িয়েছে। বাজার স্থিতিশীল আছে।",
                }
            )


def base_config(input_csv, output_dir):
    return {
        "paths": {"input_csv": str(input_csv), "output_dir": str(output_dir)},
        "pipeline": {"seed": 7},
        "input": {"id_column": "article_id", "headline_column": "headline", "text_column": "text"},
        "claim_extraction": {"backend": "llm_json", "min_confidence": 0.1},
        "claim_ranking": {"min_overall_score": 0.1, "max_risk_score": 1.0},
        "planner": {"backend": "llm_json", "allowed_families": ["numerical_fact"]},
        "generation": {"temperature": 0.0, "max_new_tokens": 256},
        "regeneration": {"max_attempts": 1, "temperature_step": 0.1},
        "verification": {
            "batch_size": 8,
            "semantic_similarity_min": 0.0,
            "contradiction_min": 0.5,
            "fluency_max_perplexity": 100.0,
            "duplicate_max_similarity": 1.1,
        },
        "human_validation": {"enabled": False},
    }


def planned_article(article_id="a1"):
    article = Article(
        article_id=article_id,
        headline="বাংলাদেশ ব্যাংক সুদের হার বাড়িয়েছে",
        text="বাংলাদেশ ব্যাংক নীতিগত সুদের হার ১০ শতাংশ বাড়িয়েছে। বাজার স্থিতিশীল আছে।",
    )
    claim = Claim(
        sentence_index=0,
        sentence="বাংলাদেশ ব্যাংক নীতিগত সুদের হার ১০ শতাংশ বাড়িয়েছে।",
        claim_type="numerical",
        entities=["বাংলাদেশ ব্যাংক"],
        numbers=["১০"],
        policies=["সুদের হার"],
        dates=[],
        confidence=0.95,
    )
    selected = RankedClaim(claim, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9)
    plan = RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Change only the selected rate from ১০ শতাংশ to ৭ শতাংশ.",
        edit_scope="target_sentence",
        expected_change="The interest-rate value changes from ১০ শতাংশ to ৭ শতাংশ.",
        verification_constraints={"preserve_all_other_sentences": True},
        target_span="১০ শতাংশ",
        replacement="৭ শতাংশ",
    )
    return PlannedArticle(article, selected, plan, sample_seed=123)


def test_pipeline_releases_role_models_once_per_staged_run(tmp_path):
    input_csv = tmp_path / "input.csv"
    write_input_csv(input_csv)

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
    config = base_config(input_csv, tmp_path / "out")

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
    assert (tmp_path / "out" / "planned_articles.jsonl").exists()
    assert "runtime" in result.stats


def test_pipeline_resumes_from_planning_checkpoint_without_instruction_model(tmp_path):
    input_csv = tmp_path / "input.csv"
    output_dir = tmp_path / "out"
    write_input_csv(input_csv, article_ids=("a1",))
    append_planned_article(output_dir / "planned_articles.jsonl", planned_article("a1"))
    bundle = ModelBundle(
        generator=ReleasableGenerator(),
        embedder=StableEmbedder(),
        nli=PassingNLI(),
        fluency=PassingFluency(),
        extractor=ExplodingInstructionModel(),
        planner=ExplodingInstructionModel(),
    )

    result = PlanningGuidedRewritePipeline(base_config(input_csv, output_dir), model_bundle=bundle).run()

    assert len(result.samples) == 1
    assert result.stats["runtime"]["counters"]["planned_checkpoint_hits"] == 1


class OomThenPassVerifier:
    def __init__(self) -> None:
        self.batch_sizes = []

    def verify_batch(self, articles, rewritten_articles, plans):
        self.batch_sizes.append(len(articles))
        if len(articles) > 1:
            raise RuntimeError("CUDA out of memory")
        return [VerificationReport([VerifierResult("test", 1.0, True, "passed")]) for _ in articles]


def test_regeneration_splits_verification_batch_on_cuda_oom():
    planned = [planned_article("a1"), planned_article("a2")]
    verifier = OomThenPassVerifier()
    controller = RegenerationController(
        RewriteGenerator(ReleasableGenerator()),
        verifier,
        RegenerationConfig(max_attempts=1, base_temperature=0.0),
    )

    outcomes = controller.run_batch_with_attempts(
        [item.article for item in planned],
        [item.plan for item in planned],
        [10, 20],
    )

    assert [outcome.result is not None for outcome in outcomes] == [True, True]
    assert verifier.batch_sizes == [2, 1, 1]
