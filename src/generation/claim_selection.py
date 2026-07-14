from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.generation.metadata import Article, Claim, RankedClaim
from src.generation.utils import contains_financial_language

logger = logging.getLogger(__name__)


EDITABILITY_BY_TYPE = {
    "numerical": 0.90,
    "policy": 0.86,
    "entity": 0.72,
    "temporal": 0.78,
    "causal": 0.64,
}


@dataclass(frozen=True)
class ClaimRankingConfig:
    min_overall_score: float = 0.52
    max_risk_score: float = 0.72


@dataclass(frozen=True)
class ClaimRanker:
    config: ClaimRankingConfig = ClaimRankingConfig()

    def rank(self, article: Article, claims: List[Claim]) -> List[RankedClaim]:
        ranked = [self._score(article, claim) for claim in claims]
        filtered = [
            item
            for item in ranked
            if item.overall_score >= self.config.min_overall_score
            and item.risk_score <= self.config.max_risk_score
        ]
        filtered.sort(key=lambda item: item.overall_score, reverse=True)
        logger.info("Ranked %d claims; %d passed quality gate", len(claims), len(filtered))
        return filtered

    def select(self, article: Article, claims: List[Claim]) -> Optional[RankedClaim]:
        ranked = self.rank(article, claims)
        return ranked[0] if ranked else None

    def _score(self, article: Article, claim: Claim) -> RankedClaim:
        importance = self._importance(article, claim)
        editability = self._editability(claim)
        verification = self._verification_feasibility(claim)
        locality = self._locality(claim)
        risk = self._risk(claim)
        overall = (
            0.35 * importance
            + 0.25 * editability
            + 0.20 * locality
            + 0.20 * verification
        )
        if risk > self.config.max_risk_score:
            overall *= 0.70
        return RankedClaim(
            claim=claim,
            importance_score=round(importance, 4),
            editability_score=round(editability, 4),
            verification_score=round(verification, 4),
            locality_score=round(locality, 4),
            risk_score=round(risk, 4),
            overall_score=round(max(0.0, min(1.0, overall)), 4),
        )

    def _importance(self, article: Article, claim: Claim) -> float:
        score = claim.confidence * 0.45
        score += max(0.0, 0.28 - 0.03 * min(claim.sentence_index, 8))
        headline_terms = set(article.headline.split())
        sentence_terms = set(claim.sentence.split())
        if headline_terms and sentence_terms:
            score += min(0.18, len(headline_terms & sentence_terms) / max(len(headline_terms), 1))
        score += 0.09 if contains_financial_language(claim.sentence) else 0.0
        return max(0.0, min(1.0, score))

    def _editability(self, claim: Claim) -> float:
        score = EDITABILITY_BY_TYPE.get(claim.claim_type, 0.45)
        if claim.claim_type == "entity" and len(claim.entities) < 1:
            score -= 0.20
        if claim.claim_type == "numerical" and len(claim.numbers) < 1:
            score -= 0.25
        if claim.claim_type == "policy" and len(claim.policies) < 1:
            score -= 0.25
        return max(0.0, min(1.0, score))

    def _verification_feasibility(self, claim: Claim) -> float:
        anchors = len(claim.entities) + len(claim.numbers) + len(claim.policies) + len(claim.dates)
        score = 0.35 + min(0.45, 0.10 * anchors)
        if claim.claim_type in {"numerical", "policy", "temporal"}:
            score += 0.15
        if claim.claim_type == "causal":
            score += 0.08
        return max(0.0, min(1.0, score))

    def _locality(self, claim: Claim) -> float:
        words = len(claim.sentence.split())
        score = 0.92
        if words > 45:
            score -= 0.18
        if words > 75:
            score -= 0.22
        score -= min(0.20, 0.04 * max(0, claim.sentence.count(",") - 2))
        return max(0.0, min(1.0, score))

    def _risk(self, claim: Claim) -> float:
        score = 0.12
        score += min(0.35, 0.08 * max(0, len(claim.numbers) - 2))
        score += min(0.25, 0.06 * max(0, len(claim.entities) - 3))
        score += 0.20 if len(claim.sentence.split()) > 90 else 0.0
        score += 0.15 if "“" in claim.sentence or "\"" in claim.sentence else 0.0
        return max(0.0, min(1.0, score))


def build_claim_ranker(config: Dict[str, object] | None = None) -> ClaimRanker:
    cfg = config or {}
    return ClaimRanker(
        ClaimRankingConfig(
            min_overall_score=float(cfg.get("min_overall_score", 0.52)),
            max_risk_score=float(cfg.get("max_risk_score", 0.72)),
        )
    )
