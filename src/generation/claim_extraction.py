from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from src.generation.metadata import Article, Claim
from src.generation.models import InstructionModel
from src.generation.prompts import build_claim_extraction_prompt
from src.generation.utils import (
    CAUSAL_TERMS,
    contains_financial_language,
    extract_json_payload,
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
                    claim_text=sentence,
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


@dataclass(frozen=True)
class LLMClaimExtractor:
    model: InstructionModel
    max_claims: int = 12
    max_new_tokens: int = 1536
    temperature: float = 0.0
    seed: int = 42
    min_confidence: float = 0.35

    def extract(self, article: Article) -> List[Claim]:
        prompt = build_claim_extraction_prompt(article, self.max_claims)
        raw = self.model.generate_text(prompt, self.temperature, self.seed, self.max_new_tokens)
        payload = extract_json_payload(raw)
        rows = payload.get("claims", payload) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("Claim extractor JSON must be a list or object with a claims list")
        claims = [claim for claim in (self._claim_from_row(row) for row in rows) if claim is not None]
        logger.info("LLM extracted %d claim candidates for article %s", len(claims), article.article_id)
        return claims[: self.max_claims]

    def _claim_from_row(self, row: Any) -> Claim | None:
        if not isinstance(row, dict):
            raise ValueError("Each extracted claim must be a JSON object")
        confidence = float(row.get("confidence", 0.0))
        if confidence < self.min_confidence:
            return None
        sentence = str(row.get("sentence", "")).strip()
        if not sentence:
            raise ValueError("Extracted claim is missing sentence")
        return Claim(
            sentence_index=int(row["sentence_index"]),
            sentence=sentence,
            claim_type=self._normalize_type(str(row.get("type") or row.get("claim_type") or "")),
            entities=self._list(row.get("entities")),
            numbers=self._list(row.get("numbers")),
            policies=self._list(row.get("policies")),
            dates=self._list(row.get("dates")),
            confidence=max(0.0, min(1.0, confidence)),
            claim_text=str(row.get("claim") or row.get("claim_text") or sentence).strip(),
            extractor_model=self.model.model_name,
        )

    def _normalize_type(self, value: str) -> str:
        normalized = value.strip().lower().replace(" ", "_")
        aliases = {
            "numeric": "numerical",
            "number": "numerical",
            "numerical_fact": "numerical",
            "policy_reversal": "policy",
            "entity_replacement": "entity",
            "temporal_shift": "temporal",
            "causal_inversion": "causal",
        }
        normalized = aliases.get(normalized, normalized)
        allowed = {"numerical", "policy", "entity", "temporal", "causal"}
        if normalized not in allowed:
            raise ValueError(f"Unsupported claim type from extractor: {value!r}")
        return normalized

    def _list(self, value: Any) -> List[str]:
        if value is None:
            return []
        if not isinstance(value, list):
            raise ValueError("Claim extraction fields entities/numbers/policies/dates must be lists")
        return [str(item).strip() for item in value if str(item).strip()]


def build_claim_extractor(
    config: Dict[str, object] | None = None,
    model: InstructionModel | None = None,
) -> ClaimExtractor:
    cfg = config or {}
    backend = str(cfg.get("backend", "heuristic"))
    if backend == "llm_json":
        if model is None:
            raise ValueError("claim_extraction.backend=llm_json requires models.extractor")
        return LLMClaimExtractor(
            model=model,
            max_claims=int(cfg.get("max_claims", 12)),
            max_new_tokens=int(cfg.get("max_new_tokens", 1536)),
            temperature=float(cfg.get("temperature", 0.0)),
            seed=int(cfg.get("seed", 42)),
            min_confidence=float(cfg.get("min_confidence", 0.35)),
        )
    if backend != "heuristic":
        raise ValueError(f"Unsupported claim extraction backend: {backend}")
    return HeuristicClaimExtractor(
        min_confidence=float(cfg.get("min_confidence", 0.35)),
        min_sentence_chars=int(cfg.get("min_sentence_chars", 18)),
        max_sentences=int(cfg.get("max_sentences", 80)),
    )
