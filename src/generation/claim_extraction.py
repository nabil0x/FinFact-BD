from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Protocol

from src.generation.metadata import Article, Claim
from src.generation.utils import (
    CAUSAL_TERMS,
    contains_financial_language,
    extract_dates,
    extract_entities,
    extract_numbers,
    extract_policies,
    sentence_spans,
)

logger = logging.getLogger(__name__)


class ClaimExtractor(Protocol):
    def extract(self, article: Article) -> List[Claim]:
        """Return factual proposition candidates from an article."""


@dataclass(frozen=True)
class HeuristicClaimExtractor:
    min_confidence: float = 0.35
    min_sentence_chars: int = 18
    max_sentences: int = 80

    def extract(self, article: Article) -> List[Claim]:
        claims: List[Claim] = []
        for span in sentence_spans(article.text)[: self.max_sentences]:
            sentence = span.text.strip()
            if len(sentence) < self.min_sentence_chars:
                continue
            entities = extract_entities(sentence)
            numbers = extract_numbers(sentence)
            policies = extract_policies(sentence)
            dates = extract_dates(sentence)
            claim_type = self._claim_type(sentence, entities, numbers, policies, dates)
            confidence = self._confidence(span.index, sentence, entities, numbers, policies, dates)
            if claim_type == "other" or confidence < self.min_confidence:
                continue
            claims.append(
                Claim(
                    sentence_index=span.index,
                    sentence=sentence,
                    claim_type=claim_type,
                    entities=entities,
                    numbers=numbers,
                    policies=policies,
                    dates=dates,
                    confidence=round(confidence, 4),
                )
            )
        logger.info("Extracted %d claim candidates for article %s", len(claims), article.article_id)
        return claims

    def _claim_type(
        self,
        sentence: str,
        entities: List[str],
        numbers: List[str],
        policies: List[str],
        dates: List[str],
    ) -> str:
        if any(term in sentence for term in CAUSAL_TERMS):
            return "causal"
        if policies:
            return "policy"
        if numbers:
            return "numerical"
        if dates:
            return "temporal"
        if entities:
            return "entity"
        return "other"

    def _confidence(
        self,
        index: int,
        sentence: str,
        entities: List[str],
        numbers: List[str],
        policies: List[str],
        dates: List[str],
    ) -> float:
        score = 0.10
        score += max(0.0, 0.22 - 0.025 * min(index, 8))
        score += 0.18 if contains_financial_language(sentence) else 0.0
        score += min(0.18, 0.06 * len(entities))
        score += min(0.18, 0.06 * len(numbers))
        score += min(0.16, 0.08 * len(policies))
        score += min(0.12, 0.06 * len(dates))
        score += 0.08 if any(term in sentence for term in CAUSAL_TERMS) else 0.0
        if len(sentence.split()) < 5:
            score -= 0.18
        if len(sentence.split()) > 80:
            score -= 0.08
        return max(0.0, min(1.0, score))


def build_claim_extractor(config: Dict[str, object] | None = None) -> ClaimExtractor:
    cfg = config or {}
    return HeuristicClaimExtractor(
        min_confidence=float(cfg.get("min_confidence", 0.35)),
        min_sentence_chars=int(cfg.get("min_sentence_chars", 18)),
        max_sentences=int(cfg.get("max_sentences", 80)),
    )
