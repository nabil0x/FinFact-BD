from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from src.generation.metadata import Article, AttemptRecord, GeneratedRewrite, RewritePlan, VerificationReport
from src.generation.rewrite_generator import RewriteGenerator
from src.generation.verifier import CompositeVerifier

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RegenerationResult:
    generation: GeneratedRewrite
    verification: VerificationReport
    attempts: List[AttemptRecord]


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
        attempts: List[AttemptRecord] = []
        for attempt in range(1, self.config.max_attempts + 1):
            temperature = self.config.base_temperature + self.config.temperature_step * (attempt - 1)
            attempt_seed = seed + attempt
            try:
                generation = self.generator.rewrite(
                    article=article,
                    plan=plan,
                    temperature=temperature,
                    seed=attempt_seed,
                    attempt=attempt,
                )
                report = self.verifier.verify(article, generation.rewritten_article, plan)
            except Exception as exc:
                logger.exception(
                    "Regeneration attempt failed article=%s attempt=%d",
                    article.article_id,
                    attempt,
                )
                attempts.append(
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
            attempts.append(
                AttemptRecord(
                    attempt=attempt,
                    temperature=temperature,
                    prompt_version=generation.params.prompt_version,
                    rewritten_article=generation.rewritten_article,
                    verification=report.to_dict(),
                )
            )
            if report.passed:
                logger.info("Accepted rewrite article=%s attempt=%d", article.article_id, attempt)
                return RegenerationResult(generation=generation, verification=report, attempts=attempts)
            logger.info(
                "Rejected rewrite article=%s attempt=%d reasons=%s",
                article.article_id,
                attempt,
                report.reasons,
            )
        logger.warning("No passing rewrite after %d attempts for %s", self.config.max_attempts, article.article_id)
        return None
