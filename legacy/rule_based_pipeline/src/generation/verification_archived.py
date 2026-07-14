#!/usr/bin/env python3
"""Multi-stage verification module for the FinFact-BD claim-guided rewriting pipeline.

Three rule-based stages check claim integrity, surface quality, and semantic quality
before a generated rewrite is accepted. No ML models are used — all checks are
string-based or heuristic.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Optional, TYPE_CHECKING
import re

if TYPE_CHECKING:
    from src.generation.claim_planning import RewritePlan  # noqa: F401

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SENTENCE_PATTERN = re.compile(r"[^\u0964।!?]+(?:[।!?]+|$)")
_WS = re.compile(r"\s+")
_REPEATED_CHAR = re.compile(r"(.)\1{3,}")
_EXCESSIVE_WS = re.compile(r"\s{3,}")

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StageResult:
    """Outcome of a single verification stage."""

    passed: bool
    score: float
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "score": round(self.score, 4), "issues": list(self.issues)}


@dataclass
class VerificationResult:
    """Aggregated outcome across all three verification stages."""

    passed: bool
    claim_integrity: StageResult
    surface_quality: StageResult
    semantic_quality: StageResult
    overall_score: float
    failure_reason: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize for metadata export — matches the paper schema."""
        return {
            "claim_integrity": self.claim_integrity.passed,
            "surface_quality": round(self.surface_quality.score, 2),
            "semantic_quality": round(self.semantic_quality.score, 2),
        }

    def to_full_dict(self) -> dict:
        """Full serialization including all stage details."""
        return {
            "passed": self.passed,
            "claim_integrity": self.claim_integrity.to_dict(),
            "surface_quality": self.surface_quality.to_dict(),
            "semantic_quality": self.semantic_quality.to_dict(),
            "overall_score": round(self.overall_score, 4),
            "failure_reason": self.failure_reason,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sentence_spans(text: str) -> list[dict[str, Any]]:
    """Split text into sentence spans, preserving offsets."""
    spans: list[dict[str, Any]] = []
    for idx, match in enumerate(_SENTENCE_PATTERN.finditer(text)):
        sent = match.group()
        if sent.strip():
            spans.append({
                "index": idx,
                "start": match.start(),
                "end": match.end(),
                "text": sent,
            })
    return spans


def _normalize_text(text: str) -> str:
    """Lower, strip punctuation, collapse whitespace."""
    t = text.lower()
    t = re.sub(r"[।!?;:,\-—()\"'“”‘’]", " ", t)
    t = _WS.sub(" ", t).strip()
    return t


def _char_ngram_jaccard(a: str, b: str, n: int = 2) -> float:
    """Character n-gram Jaccard similarity (default: bigrams)."""
    set_a = set(a[i:i + n] for i in range(len(a) - n + 1))
    set_b = set(b[i:i + n] for i in range(len(b) - n + 1))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class RewriteVerifier:
    """Three-stage rule-based verifier for claim-guided rewrites.

    All checks are string-based/heuristic — no model inference.
    """

    def __init__(
        self,
        contradiction_threshold: float = 0.55,
        semantic_similarity_min: float = 0.68,
        semantic_similarity_max: float = 0.999,
        fluency_min_score: float = 0.65,
        difficulty_aware: bool = True,
    ) -> None:
        self.contradiction_threshold = contradiction_threshold
        self.semantic_similarity_min = semantic_similarity_min
        self.semantic_similarity_max = semantic_similarity_max
        self.fluency_min_score = fluency_min_score
        self.difficulty_aware = difficulty_aware

        # Thresholds matching ``perturbation_pipeline.CONTRADICTION_MIN_SCORE_BY_DIFFICULTY``
        self._contradiction_by_difficulty: dict[str, float] = {
            "easy": 0.43,
            "medium": 0.52,
            "hard": 0.58,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def verify(
        self,
        original_text: str,
        rewritten_text: str,
        plan: Any,  # expected: RewritePlan from claim_planning.py
    ) -> VerificationResult:
        """Run all three verification stages and return the aggregated result."""
        if not original_text or not rewritten_text:
            return VerificationResult(
                passed=False,
                claim_integrity=StageResult(passed=False, score=0.0, issues=["empty_text"]),
                surface_quality=StageResult(passed=False, score=0.0, issues=["empty_text"]),
                semantic_quality=StageResult(passed=False, score=0.0, issues=["empty_text"]),
                overall_score=0.0,
                failure_reason="empty_text",
            )

        integrity = self._stage_claim_integrity(original_text, rewritten_text, plan)
        surface = self._stage_surface_quality(original_text, rewritten_text)
        semantic = self._stage_semantic_quality(original_text, rewritten_text, plan)

        # Overall: weighted composite — claim integrity is a gating factor.
        if not integrity.passed:
            overall_score = 0.0
            failure_reason = integrity.issues[0] if integrity.issues else "claim_integrity_failed"
        else:
            overall_score = (
                0.30 * integrity.score
                + 0.35 * surface.score
                + 0.35 * semantic.score
            )
            overall_score = max(0.0, min(1.0, overall_score))
            failure_reason = None

        passed = integrity.passed and surface.passed and semantic.passed

        return VerificationResult(
            passed=passed,
            claim_integrity=integrity,
            surface_quality=surface,
            semantic_quality=semantic,
            overall_score=round(overall_score, 4),
            failure_reason=failure_reason,
        )

    # ------------------------------------------------------------------
    # Stage 1 — Claim Integrity
    # ------------------------------------------------------------------

    def _stage_claim_integrity(
        self, original: str, rewritten: str, plan: Any,
    ) -> StageResult:
        """Check target sentence changed & no extra changes."""
        issues: list[str] = []
        orig_spans = _sentence_spans(original)
        rewr_spans = _sentence_spans(rewritten)

        target_idx = getattr(plan, "target_sentence_index", None)
        target_span = getattr(plan, "target_span", "")

        if target_idx is None:
            issues.append("no_target_sentence_index")
            return StageResult(passed=False, score=0.0, issues=issues)

        if target_idx >= len(orig_spans) or target_idx >= len(rewr_spans):
            issues.append("target_sentence_out_of_bounds")
            return StageResult(passed=False, score=0.0, issues=issues)

        orig_target = orig_spans[target_idx]["text"].strip()
        rewr_target = rewr_spans[target_idx]["text"].strip()

        target_changed = orig_target != rewr_target
        if not target_changed:
            issues.append("target_sentence_unchanged")

        if target_changed and target_span and target_span in rewr_target:
            issues.append("target_span_present_in_rewrite")

        extra_changes = 0
        for i in range(min(len(orig_spans), len(rewr_spans))):
            if i == target_idx:
                continue
            if orig_spans[i]["text"].strip() != rewr_spans[i]["text"].strip():
                extra_changes += 1
        extra_changes += abs(len(rewr_spans) - len(orig_spans))

        if extra_changes > 0:
            issues.append(f"extra_sentence_changes:{extra_changes}")

        score = 1.0
        if not target_changed:
            score = 0.0
        else:
            score -= min(0.5, 0.15 * extra_changes)
        score = max(0.0, min(1.0, score))

        passed = target_changed and extra_changes == 0
        return StageResult(passed=passed, score=score, issues=issues)

    # ------------------------------------------------------------------
    # Stage 2 — Surface Quality
    # ------------------------------------------------------------------

    def _stage_surface_quality(
        self, original: str, rewritten: str,
    ) -> StageResult:
        """Fluency, style preservation, visual distinctness."""
        issues: list[str] = []
        score = 1.0

        rewr_spans = _sentence_spans(rewritten)

        for span in rewr_spans:
            words = span["text"].split()
            if len(words) < 3:
                score -= 0.08
                if "very_short_sentence" not in issues:
                    issues.append("very_short_sentence")
            if len(words) > 100:
                score -= 0.05
                if "very_long_sentence" not in issues:
                    issues.append("very_long_sentence")

        avg_len = sum(len(s["text"].split()) for s in rewr_spans) / max(len(rewr_spans), 1)
        if avg_len < 5 or avg_len > 60:
            score -= 0.1
            issues.append("unusual_average_sentence_length")

        orig_punct = len(re.findall(r"[।!?]", original))
        rewr_punct = len(re.findall(r"[।!?]", rewritten))
        if orig_punct > 0:
            ratio = rewr_punct / orig_punct
            if ratio < 0.5 or ratio > 2.0:
                score -= 0.12
                issues.append("punctuation_count_shift")

        if _REPEATED_CHAR.search(rewritten):
            score -= 0.15
            issues.append("repeated_character_artefact")
        if _EXCESSIVE_WS.search(rewritten):
            score -= 0.1
            issues.append("excessive_whitespace")
        if rewritten.count("(") != rewritten.count(")"):
            score -= 0.05
            issues.append("mismatched_brackets")

        norm_orig = re.sub(r"\s+", "", original)
        norm_rewr = re.sub(r"\s+", "", rewritten)
        if SequenceMatcher(None, norm_orig, norm_rewr).ratio() > 0.995:
            score -= 0.3
            issues.append("too_similar_to_original")

        score = max(0.0, min(1.0, score))
        passed = score >= self.fluency_min_score
        return StageResult(passed=passed, score=score, issues=issues)

    # ------------------------------------------------------------------
    # Stage 3 — Semantic Quality
    # ------------------------------------------------------------------

    def _stage_semantic_quality(
        self, original: str, rewritten: str, plan: Any,
    ) -> StageResult:
        """Similarity bounds, contradiction proxy, duplicate check."""
        issues: list[str] = []
        score = 1.0

        if original.strip() == rewritten.strip():
            return StageResult(passed=False, score=0.0, issues=["exact_duplicate"])

        norm_orig = _normalize_text(original)
        norm_rewr = _normalize_text(rewritten)

        bigram_jac = _char_ngram_jaccard(norm_orig, norm_rewr, n=2)
        word_a, word_b = set(norm_orig.split()), set(norm_rewr.split())
        word_jac = len(word_a & word_b) / len(word_a | word_b) if word_a and word_b else 0.0
        similarity = 0.4 * bigram_jac + 0.6 * word_jac

        if similarity < self.semantic_similarity_min:
            issues.append("semantic_similarity_below_threshold")
            score -= 0.3
        elif similarity > self.semantic_similarity_max:
            issues.append("semantic_similarity_above_threshold")
            score -= 0.3

        target_idx = getattr(plan, "target_sentence_index", None)
        if target_idx is not None:
            orig_spans = _sentence_spans(original)
            rewr_spans = _sentence_spans(rewritten)
            if target_idx < len(orig_spans) and target_idx < len(rewr_spans):
                if orig_spans[target_idx]["text"].strip() == rewr_spans[target_idx]["text"].strip():
                    issues.append("target_unchanged")
                    score -= 0.3

        threshold = self.contradiction_threshold
        if self.difficulty_aware:
            threshold = self._contradiction_by_difficulty.get(
                getattr(plan, "difficulty", ""), self.contradiction_threshold
            )
        if 1.0 - similarity < threshold:
            issues.append("weak_contradiction_support")
            score -= 0.2

        score = max(0.0, min(1.0, score))
        passed = score >= (self.semantic_similarity_min * 0.85)
        return StageResult(passed=passed, score=score, issues=issues)
