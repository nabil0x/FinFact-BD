from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Article:
    article_id: str
    headline: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SentenceSpan:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Claim:
    sentence_index: int
    sentence: str
    claim_type: str
    entities: List[str]
    numbers: List[str]
    policies: List[str]
    dates: List[str]
    confidence: float
    claim_text: str = ""
    extractor_model: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RankedClaim:
    claim: Claim
    importance_score: float
    editability_score: float
    verification_score: float
    locality_score: float
    risk_score: float
    overall_score: float

    @property
    def passed_quality_gate(self) -> bool:
        return self.overall_score > 0.0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["claim"] = self.claim.to_dict()
        return data


@dataclass(frozen=True)
class RewritePlan:
    family: str
    target_claim: Claim
    edit_instruction: str
    edit_scope: str
    expected_change: str
    verification_constraints: Dict[str, Any]
    target_span: str = ""
    replacement: str = ""
    planner_model: str = "heuristic"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["target_claim"] = self.target_claim.to_dict()
        return data


@dataclass(frozen=True)
class GenerationParams:
    model_name: str
    model_revision: str
    prompt_version: str
    temperature: float
    seed: int
    attempt: int
    max_new_tokens: int


@dataclass(frozen=True)
class GeneratedRewrite:
    rewritten_article: str
    prompt: str
    params: GenerationParams


@dataclass(frozen=True)
class VerifierResult:
    name: str
    score: float
    passed: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VerificationReport:
    results: List[VerifierResult]

    @property
    def passed(self) -> bool:
        return all(result.passed for result in self.results)

    @property
    def scores(self) -> Dict[str, float]:
        return {result.name: result.score for result in self.results}

    @property
    def reasons(self) -> List[str]:
        return [result.reason for result in self.results if not result.passed]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "scores": self.scores,
            "reasons": self.reasons,
            "results": [result.to_dict() for result in self.results],
        }


@dataclass(frozen=True)
class AttemptRecord:
    attempt: int
    temperature: float
    prompt_version: str
    rewritten_article: Optional[str]
    verification: Optional[Dict[str, Any]]
    error: Optional[str] = None


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    article_id: str
    headline: str
    original_article: str
    rewritten_article: str
    selected_claim: Dict[str, Any]
    claim_index: int
    claim_type: str
    perturbation_family: str
    rewrite_plan: Dict[str, Any]
    generator_model: str
    model_revision: str
    prompt_version: str
    temperature: float
    seed: int
    verification_scores: Dict[str, Any]
    regeneration_attempts: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
