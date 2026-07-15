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


@dataclass
class SemanticSimilarityVerifier:
    embedder: EmbeddingModel
    min_score: float = 0.74
    name: str = "semantic_similarity"
    _original_cache: Dict[str, List[float]] = field(default_factory=dict)

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        return self.verify_batch([article], [rewritten], [plan])[0]

    def verify_batch(self, articles: List[Article], rewritten_articles: List[str], plans: List[RewritePlan]) -> List[VerifierResult]:
        original_vectors = self._original_vectors(articles)
        rewritten_vectors = self.embedder.encode(rewritten_articles)
        results: List[VerifierResult] = []
        for original_vec, rewritten_vec in zip(original_vectors, rewritten_vectors):
            score = cosine(original_vec, rewritten_vec)
            passed = score >= self.min_score
            results.append(
                VerifierResult(
                    self.name,
                    round(score, 4),
                    passed,
                    "passed" if passed else "embedding_similarity_below_threshold",
                )
            )
        return results

    def _original_vectors(self, articles: List[Article]) -> List[List[float]]:
        missing_ids: List[str] = []
        missing_texts: List[str] = []
        for article in articles:
            if article.article_id not in self._original_cache:
                missing_ids.append(article.article_id)
                missing_texts.append(article.text)
        if missing_texts:
            for article_id, vector in zip(missing_ids, self.embedder.encode(missing_texts)):
                self._original_cache[article_id] = vector
        return [self._original_cache[article.article_id] for article in articles]


@dataclass(frozen=True)
class ContradictionVerifier:
    nli: NLIModel
    min_score: float = 0.55
    name: str = "contradiction"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        return self.verify_batch([article], [rewritten], [plan])[0]

    def verify_batch(self, articles: List[Article], rewritten_articles: List[str], plans: List[RewritePlan]) -> List[VerifierResult]:
        results: List[VerifierResult | None] = [None] * len(articles)
        premises: List[str] = []
        hypotheses: List[str] = []
        valid_indices: List[int] = []
        for batch_idx, (article, rewritten, plan) in enumerate(zip(articles, rewritten_articles, plans)):
            idx = plan.target_claim.sentence_index
            original_spans = sentence_spans(article.text)
            rewritten_spans = sentence_spans(rewritten)
            if idx >= len(original_spans) or idx >= len(rewritten_spans):
                results[batch_idx] = VerifierResult(self.name, 0.0, False, "target_sentence_missing")
                continue
            valid_indices.append(batch_idx)
            premises.append(original_spans[idx].text)
            hypotheses.append(rewritten_spans[idx].text)
        scores = self._contradiction_scores(premises, hypotheses) if premises else []
        for batch_idx, score in zip(valid_indices, scores):
            passed = score >= self.min_score
            results[batch_idx] = VerifierResult(
                self.name,
                round(score, 4),
                passed,
                "passed" if passed else "nli_contradiction_below_threshold",
            )
        return [result for result in results if result is not None]

    def _contradiction_scores(self, premises: List[str], hypotheses: List[str]) -> List[float]:
        batch_fn = getattr(self.nli, "contradiction_scores", None)
        if callable(batch_fn):
            return list(batch_fn(premises, hypotheses))
        return [self.nli.contradiction_score(premise, hypothesis) for premise, hypothesis in zip(premises, hypotheses)]


@dataclass(frozen=True)
class FluencyVerifier:
    fluency: FluencyModel
    max_perplexity: float = 220.0
    name: str = "fluency"

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerifierResult:
        return self.verify_batch([article], [rewritten], [plan])[0]

    def verify_batch(self, articles: List[Article], rewritten_articles: List[str], plans: List[RewritePlan]) -> List[VerifierResult]:
        results: List[VerifierResult] = []
        for perplexity in self._perplexities(rewritten_articles):
            score = max(0.0, min(1.0, 1.0 - (perplexity / max(self.max_perplexity * 1.5, 1.0))))
            passed = perplexity <= self.max_perplexity
            reason = "passed" if passed else "perplexity_above_threshold"
            results.append(VerifierResult(self.name, round(score, 4), passed, reason, {"perplexity": perplexity}))
        return results

    def _perplexities(self, texts: List[str]) -> List[float]:
        batch_fn = getattr(self.fluency, "perplexities", None)
        if callable(batch_fn):
            return list(batch_fn(texts))
        return [self.fluency.perplexity(text) for text in texts]


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
        return self.verify_batch([article], [rewritten], [plan])[0]

    def verify_batch(self, articles: List[Article], rewritten_articles: List[str], plans: List[RewritePlan]) -> List[VerifierResult]:
        vectors = self.embedder.encode(rewritten_articles)
        results: List[VerifierResult] = []
        batch_vectors: List[List[float]] = []
        for vector in vectors:
            candidates = self._accepted_vectors + batch_vectors
            max_seen = max((cosine(vector, existing) for existing in candidates), default=0.0)
            passed = max_seen < self.max_similarity
            score = max(0.0, 1.0 - max_seen)
            results.append(
                VerifierResult(
                    self.name,
                    round(score, 4),
                    passed,
                    "passed" if passed else "near_duplicate_in_corpus",
                    {"max_seen_similarity": round(max_seen, 4)},
                )
            )
            batch_vectors.append(vector)
        return results

    def accept(self, rewritten: str) -> None:
        self._accepted_vectors.append(self.embedder.encode([rewritten])[0])


@dataclass(frozen=True)
class CompositeVerifier:
    verifiers: List[Verifier]

    def verify(self, article: Article, rewritten: str, plan: RewritePlan) -> VerificationReport:
        return self.verify_batch([article], [rewritten], [plan])[0]

    def verify_batch(self, articles: List[Article], rewritten_articles: List[str], plans: List[RewritePlan]) -> List[VerificationReport]:
        if not (len(articles) == len(rewritten_articles) == len(plans)):
            raise ValueError("Verifier batch inputs must have identical lengths")
        result_rows: List[List[VerifierResult]] = [[] for _ in articles]
        for verifier in self.verifiers:
            batch_fn = getattr(verifier, "verify_batch", None)
            if callable(batch_fn):
                results = batch_fn(articles, rewritten_articles, plans)
            else:
                results = [
                    verifier.verify(article, rewritten, plan)
                    for article, rewritten, plan in zip(articles, rewritten_articles, plans)
                ]
            if len(results) != len(articles):
                raise RuntimeError(f"Verifier {verifier.name} returned {len(results)} results for {len(articles)} inputs")
            for row, result in zip(result_rows, results):
                row.append(result)

        reports = [VerificationReport(results) for results in result_rows]
        for rewritten, report in zip(rewritten_articles, reports):
            if report.passed:
                for verifier in self.verifiers:
                    accept = getattr(verifier, "accept", None)
                    if callable(accept):
                        accept(rewritten)
        logger.info("Batch verification complete passed=%d failed=%d", sum(report.passed for report in reports), sum(not report.passed for report in reports))
        return reports


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
