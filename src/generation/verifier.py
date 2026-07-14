from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Protocol

from src.generation.metadata import Article, RewritePlan, VerificationReport, VerifierResult
from src.generation.models import EmbeddingModel, FluencyModel, NLIModel
from src.generation.utils import (
    changed_sentence_indices,
    cosine,
    extract_dates,
    extract_entities,
    extract_numbers,
    sentence_spans,
)

logger = logging.getLogger(__name__)


class Verifier(Protocol):
    name: str

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        """Score one verification condition."""


@dataclass(frozen=True)
class IntendedChangeVerifier:
    name: str = "intended_change"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        original_spans = sentence_spans(article.text)
        rewritten_spans = sentence_spans(rewritten)
        idx = plan.target_claim.sentence_index
        if idx >= len(original_spans) or idx >= len(rewritten_spans):
            return VerifierResult(self.name, 0.0, False, "target_sentence_missing")
        changed = original_spans[idx].text.strip() != rewritten_spans[idx].text.strip()
        score = 1.0 if changed else 0.0
        return VerifierResult(self.name, score, changed, "passed" if changed else "target_unchanged")


@dataclass(frozen=True)
class LocalityVerifier:
    name: str = "locality"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        changed = changed_sentence_indices(article.text, rewritten)
        target = plan.target_claim.sentence_index
        unexpected = [idx for idx in changed if idx != target]
        passed = bool(changed) and not unexpected and target in changed
        score = 1.0 if passed else max(0.0, 1.0 - 0.35 * len(unexpected))
        reason = "passed" if passed else f"unexpected_sentence_changes:{unexpected}"
        return VerifierResult(self.name, round(score, 4), passed, reason, {"changed_indices": changed})


@dataclass(frozen=True)
class SemanticSimilarityVerifier:
    embedder: EmbeddingModel
    min_score: float = 0.74
    name: str = "semantic_similarity"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        original_vec, rewritten_vec = self.embedder.encode([article.text, rewritten])
        score = cosine(original_vec, rewritten_vec)
        passed = score >= self.min_score
        return VerifierResult(
            self.name,
            round(score, 4),
            passed,
            "passed" if passed else "embedding_similarity_below_threshold",
        )


@dataclass(frozen=True)
class ContradictionVerifier:
    nli: NLIModel
    min_score: float = 0.55
    name: str = "contradiction"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        idx = plan.target_claim.sentence_index
        original_spans = sentence_spans(article.text)
        rewritten_spans = sentence_spans(rewritten)
        if idx >= len(original_spans) or idx >= len(rewritten_spans):
            return VerifierResult(self.name, 0.0, False, "target_sentence_missing")
        score = self.nli.contradiction_score(original_spans[idx].text, rewritten_spans[idx].text)
        passed = score >= self.min_score
        return VerifierResult(
            self.name,
            round(score, 4),
            passed,
            "passed" if passed else "nli_contradiction_below_threshold",
        )


@dataclass(frozen=True)
class FluencyVerifier:
    fluency: FluencyModel
    max_perplexity: float = 220.0
    name: str = "fluency"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        perplexity = self.fluency.perplexity(rewritten)
        score = max(0.0, min(1.0, 1.0 - (perplexity / max(self.max_perplexity * 1.5, 1.0))))
        passed = perplexity <= self.max_perplexity
        reason = "passed" if passed else "perplexity_above_threshold"
        return VerifierResult(self.name, round(score, 4), passed, reason, {"perplexity": perplexity})


@dataclass(frozen=True)
class JournalisticStyleVerifier:
    name: str = "journalistic_style"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        issues: List[str] = []
        score = 1.0
        original_paras = [p for p in article.text.split("\n") if p.strip()]
        rewritten_paras = [p for p in rewritten.split("\n") if p.strip()]
        if original_paras and abs(len(rewritten_paras) - len(original_paras)) > 1:
            issues.append("paragraph_structure_shift")
            score -= 0.20
        if len(re.findall(r"[।!?]", rewritten)) < max(1, len(re.findall(r"[।!?]", article.text)) // 2):
            issues.append("punctuation_structure_shift")
            score -= 0.20
        if re.search(r"(.)\1{4,}", rewritten):
            issues.append("repeated_character_artifact")
            score -= 0.15
        if any(token in rewritten.lower() for token in ("click", "subscribe", "breaking!!!")):
            issues.append("non_news_register")
            score -= 0.25
        headline_terms = set(article.headline.split())
        if headline_terms and not (headline_terms & set(rewritten.split())):
            issues.append("headline_topic_drift")
            score -= 0.20
        score = max(0.0, min(1.0, score))
        passed = score >= 0.72
        return VerifierResult(
            self.name,
            round(score, 4),
            passed,
            "passed" if passed else ",".join(issues),
            {"issues": issues},
        )


@dataclass(frozen=True)
class HallucinationVerifier:
    name: str = "hallucination"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        target = plan.target_claim.sentence_index
        original_outside = self._outside_elements(article.text, target)
        rewritten_outside = self._outside_elements(rewritten, target)
        new_items = {
            key: sorted(rewritten_outside[key] - original_outside[key])
            for key in original_outside
        }
        flat_new = [item for values in new_items.values() for item in values]
        passed = not flat_new
        score = max(0.0, 1.0 - 0.15 * len(flat_new))
        return VerifierResult(
            self.name,
            round(score, 4),
            passed,
            "passed" if passed else "new_facts_outside_target",
            {"new_items": new_items},
        )

    def _outside_elements(self, text: str, target: int) -> Dict[str, set[str]]:
        sentences = [span.text for span in sentence_spans(text) if span.index != target]
        outside = " ".join(sentences)
        return {
            "entities": set(extract_entities(outside)),
            "numbers": set(extract_numbers(outside)),
            "dates": set(extract_dates(outside)),
            "organizations": set(extract_entities(outside)),
        }


@dataclass
class DuplicateVerifier:
    embedder: EmbeddingModel
    max_similarity: float = 0.985
    name: str = "duplicate_detection"
    _accepted_vectors: List[List[float]] = field(default_factory=list)

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        vector = self.embedder.encode([rewritten])[0]
        max_seen = max((cosine(vector, existing) for existing in self._accepted_vectors), default=0.0)
        passed = max_seen < self.max_similarity
        score = max(0.0, 1.0 - max_seen)
        if passed:
            self._accepted_vectors.append(vector)
        return VerifierResult(
            self.name,
            round(score, 4),
            passed,
            "passed" if passed else "near_duplicate_in_corpus",
            {"max_seen_similarity": round(max_seen, 4)},
        )


@dataclass(frozen=True)
class CompositeVerifier:
    verifiers: List[Verifier]

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerificationReport:
        results = [verifier.verify(article, rewritten, plan) for verifier in self.verifiers]
        report = VerificationReport(results)
        logger.info("Verification passed=%s reasons=%s", report.passed, report.reasons)
        return report


def build_verifier(models: object, config: Dict[str, object] | None = None) -> CompositeVerifier:
    cfg = config or {}
    return CompositeVerifier(
        [
            IntendedChangeVerifier(),
            LocalityVerifier(),
            SemanticSimilarityVerifier(
                embedder=models.embedder,
                min_score=float(cfg.get("semantic_similarity_min", 0.74)),
            ),
            ContradictionVerifier(
                nli=models.nli,
                min_score=float(cfg.get("contradiction_min", 0.55)),
            ),
            FluencyVerifier(
                fluency=models.fluency,
                max_perplexity=float(cfg.get("fluency_max_perplexity", 220.0)),
            ),
            JournalisticStyleVerifier(),
            HallucinationVerifier(),
            DuplicateVerifier(
                embedder=models.embedder,
                max_similarity=float(cfg.get("duplicate_max_similarity", 0.985)),
            ),
        ]
    )
