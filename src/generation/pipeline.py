from __future__ import annotations

import csv
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.generation.claim_extraction import ClaimExtractor, build_claim_extractor
from src.generation.claim_selection import ClaimRanker, build_claim_ranker
from src.generation.exporter import DatasetExporter, HumanValidationWorkbookBuilder
from src.generation.metadata import Article, PlannedArticle, SampleRecord
from src.generation.models import ModelBundle, build_model_bundle
from src.generation.perturbation_planner import RewritePlanner, build_planner
from src.generation.planning_checkpoint import append_planned_article, load_planned_articles
from src.generation.regeneration import RegenerationConfig, RegenerationController, RegenerationOutcome
from src.generation.rewrite_generator import RewriteGenerator
from src.generation.runtime import RuntimeMetrics, timed
from src.generation.utils import read_json, stable_sample_id, utc_timestamp, write_json
from src.generation.verifier import CompositeVerifier, build_verifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineComponents:
    extractor: ClaimExtractor
    ranker: ClaimRanker
    planner: RewritePlanner
    regenerator: RegenerationController


@dataclass
class PipelineRunResult:
    samples: List[SampleRecord]
    failures: List[Dict[str, Any]]
    stats: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArticleProcessOutcome:
    sample: Optional[SampleRecord] = None
    failure: Optional[Dict[str, Any]] = None


class PlanningGuidedRewritePipeline:
    def __init__(
        self,
        config: Dict[str, Any],
        model_bundle: Optional[ModelBundle] = None,
        components: Optional[PipelineComponents] = None,
    ) -> None:
        self.config = config
        self.seed = int(config.get("pipeline", {}).get("seed", 42))
        self.rng = random.Random(self.seed)
        self.paths = config.get("paths", {})
        self.input_cfg = config.get("input", {})
        self.checkpoint_path = Path(self.paths.get("checkpoint", "data/generated/rewrite_checkpoint.json"))
        self.planning_checkpoint_path = self.checkpoint_path.with_name("planned_articles.jsonl")
        self.metrics = RuntimeMetrics()
        self.model_bundle = model_bundle or build_model_bundle(config["models"])
        self.components = components or self._build_components(self.model_bundle)

    def run(self, input_csv: Optional[str] = None, output_dir: Optional[str] = None, num_samples: Optional[int] = None) -> PipelineRunResult:
        input_path = Path(input_csv or self.paths.get("input_csv", ""))
        out_dir = Path(output_dir or self.paths.get("output_dir", "data/generated/rewrite_generation"))
        self.checkpoint_path = self._checkpoint_for_run(out_dir, output_dir is not None)
        self.planning_checkpoint_path = out_dir / "planned_articles.jsonl"
        logger.info("Using output_dir=%s checkpoint=%s", out_dir, self.checkpoint_path)
        articles = self._load_articles(input_path, num_samples)
        checkpoint = self._load_checkpoint()
        samples = [SampleRecord(**row) for row in checkpoint.get("samples", [])]
        failures = list(checkpoint.get("failures", []))
        processed = set(checkpoint.get("processed_ids", []))
        planned_checkpoint = load_planned_articles(self.planning_checkpoint_path)

        pending_articles = [article for article in articles if article.article_id not in processed]
        planned_articles: List[PlannedArticle] = []
        logger.info("Planning phase starting pending=%d", len(pending_articles))

        try:
            with timed(self.metrics, "planning"):
                for article in pending_articles:
                    cached = planned_checkpoint.get(article.article_id)
                    if cached is not None:
                        planned_articles.append(cached)
                        self.rng.randint(0, 2**31 - 1)
                        self.metrics.increment("planned_checkpoint_hits")
                        continue
                    try:
                        plan_outcome = self._plan_article(article)
                    except Exception as exc:
                        logger.exception("Article planning crashed: %s", article.article_id)
                        failures.append({"article_id": article.article_id, "reason": "exception", "stage": "planning", "error": str(exc)})
                        processed.add(article.article_id)
                        self._save_checkpoint(processed, samples, failures)
                        continue
                    if isinstance(plan_outcome, ArticleProcessOutcome):
                        failures.append(plan_outcome.failure or {"article_id": article.article_id, "reason": "unknown_planning_failure"})
                        processed.add(article.article_id)
                        self._save_checkpoint(processed, samples, failures)
                        continue
                    append_planned_article(self.planning_checkpoint_path, plan_outcome)
                    planned_articles.append(plan_outcome)
                    self.metrics.increment("planned_articles")
        finally:
            self._release_instruction_models()
        logger.info("Planning phase complete planned=%d failed=%d", len(planned_articles), len(failures))

        logger.info("Generation phase starting planned=%d", len(planned_articles))
        try:
            with timed(self.metrics, "generation"):
                batch_size = max(1, int(self.config.get("verification", {}).get("batch_size", 8)))
                for batch in self._chunks(planned_articles, batch_size):
                    active_batch = [planned for planned in batch if planned.article.article_id not in processed]
                    for planned_article, outcome in zip(active_batch, self._generate_article_batch(active_batch)):
                        article = planned_article.article
                        if outcome.sample is None:
                            failures.append(outcome.failure or {"article_id": article.article_id, "reason": "unknown_generation_failure"})
                        else:
                            samples.append(outcome.sample)
                        processed.add(article.article_id)
                        self._save_checkpoint(processed, samples, failures)
        finally:
            self._release_verifier_models()
            self._release_model(self.model_bundle.generator)
        logger.info("Generation phase complete accepted=%d failed=%d", len(samples), len(failures))

        stats = {
            "input_articles": len(articles),
            "accepted": len(samples),
            "failed": len(failures),
            "seed": self.seed,
            "runtime": self._runtime_metadata(),
        }
        DatasetExporter(out_dir).export(samples, stats)
        if self.config.get("human_validation", {}).get("enabled", True):
            workbook_name = self.config.get("human_validation", {}).get("workbook", "human_validation.xlsx")
            HumanValidationWorkbookBuilder(out_dir / workbook_name).build(samples)
        logger.info("Pipeline complete accepted=%d failed=%d", len(samples), len(failures))
        return PipelineRunResult(samples=samples, failures=failures, stats=stats)

    def _build_components(self, models: ModelBundle) -> PipelineComponents:
        generator = RewriteGenerator(
            model=models.generator,
            max_new_tokens=int(self.config.get("generation", {}).get("max_new_tokens", 768)),
        )
        verifier = build_verifier(models, self.config.get("verification", {}))
        regen_cfg = RegenerationConfig(
            max_attempts=int(self.config.get("regeneration", {}).get("max_attempts", 3)),
            base_temperature=float(self.config.get("generation", {}).get("temperature", 0.4)),
            temperature_step=float(self.config.get("regeneration", {}).get("temperature_step", 0.15)),
        )
        return PipelineComponents(
            extractor=build_claim_extractor(self.config.get("claim_extraction", {}), model=models.extractor),
            ranker=build_claim_ranker(self.config.get("claim_ranking", {})),
            planner=build_planner(self.config.get("planner", {}), model=models.planner),
            regenerator=RegenerationController(generator, verifier, regen_cfg),
        )

    def _plan_article(self, article: Article) -> PlannedArticle | ArticleProcessOutcome:
        claims = self.components.extractor.extract(article)
        selected = self.components.ranker.select(article, claims)
        if selected is None:
            logger.warning("No ranked claim passed quality gate for %s", article.article_id)
            return ArticleProcessOutcome(
                failure={
                    "article_id": article.article_id,
                    "reason": "no_ranked_claim",
                    "extracted_claims": [claim.to_dict() for claim in claims],
                }
            )
        plan = self.components.planner.create_plan(selected)
        sample_seed = self.rng.randint(0, 2**31 - 1)
        return PlannedArticle(article=article, selected=selected, plan=plan, sample_seed=sample_seed)

    def _generate_article(self, planned: PlannedArticle) -> ArticleProcessOutcome:
        return self._generate_article_batch([planned])[0]

    def _generate_article_batch(self, planned_articles: List[PlannedArticle]) -> List[ArticleProcessOutcome]:
        if not planned_articles:
            return []
        try:
            outcomes = self.components.regenerator.run_batch_with_attempts(
                [planned.article for planned in planned_articles],
                [planned.plan for planned in planned_articles],
                [planned.sample_seed for planned in planned_articles],
            )
        except Exception as exc:
            logger.exception("Generation batch crashed size=%d", len(planned_articles))
            return [self._generation_exception_outcome(planned, exc) for planned in planned_articles]
        return [self._outcome_from_regeneration(planned, outcome) for planned, outcome in zip(planned_articles, outcomes)]

    def _generation_exception_outcome(self, planned: PlannedArticle, exc: Exception) -> ArticleProcessOutcome:
        return ArticleProcessOutcome(
            failure={
                "article_id": planned.article.article_id,
                "reason": "exception",
                "stage": "generation",
                "error": str(exc),
                "selected_claim": planned.selected.claim.to_dict(),
                "rewrite_plan": planned.plan.to_dict(),
            }
        )

    def _outcome_from_regeneration(self, planned: PlannedArticle, outcome: RegenerationOutcome) -> ArticleProcessOutcome:
        article = planned.article
        selected = planned.selected
        plan = planned.plan
        sample_seed = planned.sample_seed
        if outcome.result is None:
            return ArticleProcessOutcome(
                failure={
                    "article_id": article.article_id,
                    "reason": "no_passing_rewrite",
                    "selected_claim": selected.claim.to_dict(),
                    "rewrite_plan": plan.to_dict(),
                    "attempts": [attempt.__dict__ for attempt in outcome.attempts],
                }
            )
        result = outcome.result
        params = result.generation.params
        sample_id = stable_sample_id(article.article_id, selected.claim.sentence_index, plan.family, sample_seed)
        return ArticleProcessOutcome(
            sample=SampleRecord(
                sample_id=sample_id,
                article_id=article.article_id,
                headline=article.headline,
                original_article=article.text,
                rewritten_article=result.generation.rewritten_article,
                selected_claim=selected.claim.to_dict(),
                claim_index=selected.claim.sentence_index,
                claim_type=selected.claim.claim_type,
                perturbation_family=plan.family,
                rewrite_plan=plan.to_dict(),
                generator_model=params.model_name,
                model_revision=params.model_revision,
                prompt_version=params.prompt_version,
                temperature=params.temperature,
                seed=params.seed,
                verification_scores=result.verification.to_dict(),
                regeneration_attempts=len(result.attempts),
                timestamp=utc_timestamp(),
            )
        )

    def _chunks(self, planned_articles: List[PlannedArticle], size: int) -> List[List[PlannedArticle]]:
        return [planned_articles[index : index + size] for index in range(0, len(planned_articles), size)]

    def _load_articles(self, path: Path, num_samples: Optional[int]) -> List[Article]:
        if not path.exists():
            raise FileNotFoundError(f"Input CSV not found: {path}")
        text_col = self.input_cfg.get("text_column", "text")
        id_col = self.input_cfg.get("id_column", "article_id")
        headline_col = self.input_cfg.get("headline_column", "headline")
        articles: List[Article] = []
        with open(path, "r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                text = (row.get(text_col) or "").strip()
                if not text:
                    continue
                article_id = str(row.get(id_col) or row.get("id") or len(articles))
                articles.append(
                    Article(
                        article_id=article_id,
                        headline=(row.get(headline_col) or "").strip(),
                        text=text,
                        metadata={key: value for key, value in row.items() if key not in {text_col, headline_col}},
                    )
                )
                if num_samples is not None and len(articles) >= num_samples:
                    break
        logger.info("Loaded %d articles from %s", len(articles), path)
        return articles

    def _checkpoint_for_run(self, out_dir: Path, output_dir_overridden: bool) -> Path:
        if output_dir_overridden:
            return out_dir / "checkpoint.json"
        configured = self.paths.get("checkpoint")
        return Path(configured) if configured else out_dir / "checkpoint.json"

    def _release_instruction_models(self) -> None:
        seen: set[int] = set()
        for model in (self.model_bundle.extractor, self.model_bundle.planner):
            if model is not None and id(model) not in seen:
                seen.add(id(model))
                self._release_model(model)

    def _release_verifier_models(self) -> None:
        seen: set[int] = set()
        for model in (self.model_bundle.embedder, self.model_bundle.nli, self.model_bundle.fluency):
            if id(model) not in seen:
                seen.add(id(model))
                self._release_model(model)

    def _release_model(self, model: object) -> None:
        release = getattr(model, "release", None)
        if callable(release):
            release()

    def _runtime_metadata(self) -> Dict[str, Any]:
        verifier_timing = self.components.regenerator.verifier.timing_summary()
        return {
            **self.metrics.to_dict(),
            "verification": verifier_timing,
            "planned_checkpoint": str(self.planning_checkpoint_path),
        }

    def _load_checkpoint(self) -> Dict[str, Any]:
        if not self.checkpoint_path.exists():
            return {"processed_ids": [], "samples": [], "failures": []}
        data = read_json(self.checkpoint_path)
        logger.info("Loaded checkpoint %s with %d processed ids", self.checkpoint_path, len(data.get("processed_ids", [])))
        return data

    def _save_checkpoint(self, processed: set[str], samples: List[SampleRecord], failures: List[Dict[str, Any]]) -> None:
        write_json(
            self.checkpoint_path,
            {
                "processed_ids": sorted(processed),
                "samples": [sample.to_dict() for sample in samples],
                "failures": failures,
            },
        )
