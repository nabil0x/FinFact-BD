from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.generation.metadata import Article, AttemptRecord, GeneratedRewrite, RewritePlan, VerificationReport
from src.generation.rewrite_generator import RewriteGenerator
from src.generation.runtime import clear_cuda_cache, is_cuda_oom
from src.generation.verifier import CompositeVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegenerationResult:
    generation: GeneratedRewrite
    verification: VerificationReport
    attempts: List[AttemptRecord]


@dataclass(frozen=True)
class RegenerationOutcome:
    result: Optional[RegenerationResult]
    attempts: List[AttemptRecord]


MAX_TEMPERATURE = 0.85
DETERMINISTIC_VERIFIER_NAMES = frozenset({
    "intended_change",
    "locality",
    "text_quality_artifacts",
    "journalistic_style",
    "hallucination",
})


@dataclass(frozen=True)
class RegenerationConfig:
    max_attempts: int = 3
    base_temperature: float = 0.4
    temperature_step: float = 0.15


@dataclass(frozen=True)
class RegenerationController:
    generator: RewriteGenerator
    verifier: CompositeVerifier
    config: RegenerationConfig = RegenerationConfig()

    def run(self, article: Article, plan: RewritePlan, seed: int) -> Optional[RegenerationResult]:
        return self.run_with_attempts(article, plan, seed).result

    def run_with_attempts(self, article: Article, plan: RewritePlan, seed: int) -> RegenerationOutcome:
        return self.run_batch_with_attempts([article], [plan], [seed])[0]

    def run_batch_with_attempts(
        self,
        articles: List[Article],
        plans: List[RewritePlan],
        seeds: List[int],
    ) -> List[RegenerationOutcome]:
        if not (len(articles) == len(plans) == len(seeds)):
            raise ValueError("Regeneration batch inputs must have identical lengths")
        attempts_by_index: List[List[AttemptRecord]] = [[] for _ in articles]
        results: List[Optional[RegenerationResult]] = [None] * len(articles)
        pending = list(range(len(articles)))
        for attempt in range(1, self.config.max_attempts + 1):
            if not pending:
                break
            raw_temp = self.config.base_temperature + self.config.temperature_step * (attempt - 1)
            temperature = min(raw_temp, MAX_TEMPERATURE)
            if temperature != raw_temp:
                logger.debug(
                    "Temperature capped from %.4f to %.4f for attempt %d",
                    raw_temp,
                    temperature,
                    attempt,
                )
            prev_failure_reasons: List[str] = []
            for idx in pending:
                prev_attempts = attempts_by_index[idx]
                if prev_attempts:
                    prev = prev_attempts[-1]
                    if prev.verification:
                        prev_failure_reasons.extend(prev.verification.get("reasons", []))

            if attempt > 1:
                det_fail = any(
                    any(r.startswith(pre) for pre in ("intended_change", "locality", "text_quality", "journalistic_style", "hallucination", "new_facts"))
                    for r in prev_failure_reasons
                )
                nondet_fail = any(
                    any(r.startswith(pre) for pre in ("embedding", "nli", "perplexity", "duplicate"))
                    for r in prev_failure_reasons
                )
                if det_fail and not nondet_fail:
                    prev_temp_raw = self.config.base_temperature + self.config.temperature_step * (attempt - 2)
                    reverted_temp = min(prev_temp_raw, MAX_TEMPERATURE)
                    logger.debug(
                        "Deterministic-only failure; resetting temperature from %.4f to %.4f",
                        temperature,
                        reverted_temp,
                    )
                    temperature = reverted_temp

            generated: List[tuple[int, GeneratedRewrite]] = []
            for idx in pending:
                try:
                    generation = self.generator.rewrite(
                        article=articles[idx],
                        plan=plans[idx],
                        temperature=temperature,
                        seed=seeds[idx] + attempt,
                        attempt=attempt,
                    )
                except Exception as exc:
                    logger.exception("Regeneration attempt failed article=%s attempt=%d", articles[idx].article_id, attempt)
                    attempts_by_index[idx].append(
                        AttemptRecord(
                            attempt=attempt,
                            temperature=temperature,
                            prompt_version="unavailable",
                            rewritten_article=None,
                            verification=None,
                            error=str(exc),
                        )
                    )
                    continue
                generated.append((idx, generation))
            if not generated:
                continue

            try:
                reports = self._verify_generated_adaptive(generated, articles, plans)
            except Exception as exc:
                logger.exception("Batch verification failed attempt=%d", attempt)
                for idx, generation in generated:
                    attempts_by_index[idx].append(
                        AttemptRecord(
                            attempt=attempt,
                            temperature=temperature,
                            prompt_version=generation.params.prompt_version,
                            rewritten_article=generation.rewritten_article,
                            verification=None,
                            error=str(exc),
                        )
                    )
                continue

            next_pending: List[int] = []
            for (idx, generation), report in zip(generated, reports):
                attempts_by_index[idx].append(
                    AttemptRecord(
                        attempt=attempt,
                        temperature=temperature,
                        prompt_version=generation.params.prompt_version,
                        rewritten_article=generation.rewritten_article,
                        verification=report.to_dict(),
                    )
                )
                if report.passed:
                    logger.info("Accepted rewrite article=%s attempt=%d", articles[idx].article_id, attempt)
                    results[idx] = RegenerationResult(generation=generation, verification=report, attempts=attempts_by_index[idx])
                else:
                    logger.info("Rejected rewrite article=%s attempt=%d reasons=%s", articles[idx].article_id, attempt, report.reasons)
                    next_pending.append(idx)
            failed_generation_indices = [idx for idx in pending if idx not in {generated_idx for generated_idx, _ in generated}]
            pending = next_pending + failed_generation_indices
        for idx, result in enumerate(results):
            if result is None:
                logger.warning("No passing rewrite after %d attempts for %s", self.config.max_attempts, articles[idx].article_id)
        return [RegenerationOutcome(result=result, attempts=attempts) for result, attempts in zip(results, attempts_by_index)]

    def _verify_generated_adaptive(
        self,
        generated: List[tuple[int, GeneratedRewrite]],
        articles: List[Article],
        plans: List[RewritePlan],
    ) -> List[VerificationReport]:
        try:
            return self.verifier.verify_batch(
                [articles[idx] for idx, _ in generated],
                [generation.rewritten_article for _, generation in generated],
                [plans[idx] for idx, _ in generated],
            )
        except Exception as exc:
            if len(generated) <= 1 or not is_cuda_oom(exc):
                raise
            clear_cuda_cache()
            midpoint = max(1, len(generated) // 2)
            logger.warning("CUDA OOM during verification batch size=%d; retrying as %d + %d", len(generated), midpoint, len(generated) - midpoint)
            return self._verify_generated_adaptive(generated[:midpoint], articles, plans) + self._verify_generated_adaptive(generated[midpoint:], articles, plans)
