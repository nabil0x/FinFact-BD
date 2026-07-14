from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Protocol

from src.generation.perturbation_pipeline import (
    FAMILY_DIFFICULTY_WEIGHTS,
    DIFFICULTY_WEIGHTS,
)

# =============================================================================
# CLAIM TYPE ↔ REWRITE FAMILY MAPPING
# =============================================================================

CLAIM_TYPE_TO_FAMILY: Dict[str, str] = {
    "numerical": "numerical_fact_change",
    "policy": "policy_reversal",
    "entity": "entity_replacement",
    "temporal": "temporal_shift",
    "causal": "causal_inversion",
}

FAMILY_TO_CLAIM_TYPE: Dict[str, str] = {v: k for k, v in CLAIM_TYPE_TO_FAMILY.items()}

# =============================================================================
# DESIRED CHANGE / EXPECTED OUTPUT TEMPLATES
# =============================================================================

DESIRED_CHANGE_TEMPLATES: Dict[str, str] = {
    "numerical_fact_change": (
        "Change the numerical value '{span}' to a plausible but incorrect figure "
        "while preserving the surrounding financial context"
    ),
    "policy_reversal": (
        "Reverse the policy direction: '{span}' should become its semantic opposite "
        "(e.g., increase → decrease, raise → cut)"
    ),
    "entity_replacement": (
        "Replace '{span}' with a different entity from the same financial category, "
        "keeping the attribution and context unchanged"
    ),
    "temporal_shift": (
        "Shift the time reference '{span}' to a different but plausible timeframe, "
        "keeping the rest of the factual claim intact"
    ),
    "causal_inversion": (
        "Invert the cause-effect relationship involving '{span}' so that the outcome "
        "contradicts the stated cause"
    ),
}

EXPECTED_CHANGED_TEMPLATES: Dict[str, str] = {
    "numerical_fact_change": (
        "The numerical value '{span}' is changed to a different figure; "
        "the surrounding financial context is preserved exactly"
    ),
    "policy_reversal": (
        "The policy action is reversed to its semantic opposite; "
        "the rest of the sentence remains unchanged"
    ),
    "entity_replacement": (
        "The named entity '{span}' is replaced with a different entity from the same class; "
        "attribution and surrounding context are preserved"
    ),
    "temporal_shift": (
        "The time reference '{span}' is shifted by a controlled offset while "
        "the rest of the factual claim stays unchanged"
    ),
    "causal_inversion": (
        "The causal relationship is inverted so that the stated cause "
        "produces the opposite effect"
    ),
}

SCOPE_BY_FAMILY: Dict[str, str] = {
    "numerical_fact_change": "sentence",
    "policy_reversal": "sentence",
    "entity_replacement": "sentence",
    "temporal_shift": "sentence",
    "causal_inversion": "sentence",
}

# =============================================================================
# SELECTED CLAIM PROTOCOL
# =============================================================================


class SelectedClaim(Protocol):
    """Protocol for a claim selected by the claim_selection module.

    Every selected claim exposes the following attributes so that
    :func:`create_rewrite_plan` can build a structured plan from it
    without coupling to a concrete dataclass.
    """

    claim_type: str
    """One of ``"numerical"``, ``"policy"``, ``"entity"``, ``"temporal"``,
    ``"causal"``."""

    span_text: str
    """The exact surface text of the claim span in the article."""

    sentence_index: int
    """0-based index of the sentence containing the claim."""

    importance_score: float
    """Financial salience of the claim (higher = more central)."""

    editability_score: float
    """How suitable the claim is for automated rewriting (0-1)."""

    diversity_bonus: float
    """Bonus weight for family/diversity balance (0-1)."""


# =============================================================================
# REWRITE PLAN DATACLASS
# =============================================================================


@dataclass(frozen=True)
class RewritePlan:
    """A structured, executable rewrite plan for one claim.

    Every field is filled by :func:`create_rewrite_plan` and consumed
    by :class:`~src.generation.bangla_rewriter.BanglaRewriter`.
    """

    sample_id: str
    """Unique identifier for this rewrite sample."""

    original_id: str
    """Identifier of the source article in BENI v2."""

    rewrite_family: str
    """One of the five rewrite families
    (``numerical_fact_change``, ``policy_reversal``,
    ``entity_replacement``, ``temporal_shift``, ``causal_inversion``)."""

    target_sentence_index: int
    """0-based index of the sentence to rewrite."""

    target_span: str
    """The exact surface text of the claim to change."""

    desired_change: str
    """Human-readable description of what should change."""

    difficulty: str
    """``"easy"``, ``"medium"``, or ``"hard"``, sampled from the
    family-aware difficulty distribution."""

    importance_score: float
    """Financial salience of the selected claim."""

    editability_score: float
    """Editability score from claim selection."""

    diversity_bonus: float
    """Diversity bonus from claim selection."""

    expected_scope: str
    """``"sentence"`` or ``"paragraph"`` --- how much text the
    rewriter needs to touch."""

    expected_changed_claim: str
    """Description of what the claim should become after rewriting."""


# =============================================================================
# DIFFICULTY SAMPLING
# =============================================================================


def _select_difficulty_for_family(family: str, rng: random.Random) -> str:
    """Sample a difficulty level from the family-specific weight table.

    Falls back to the global difficulty distribution when the family
    has no explicit weight table.
    """
    family_weights = FAMILY_DIFFICULTY_WEIGHTS.get(family)
    if family_weights is None:
        labels = list(DIFFICULTY_WEIGHTS.keys())
        weights = [DIFFICULTY_WEIGHTS[label] for label in labels]
        return rng.choices(labels, weights=weights, k=1)[0]

    labels = list(family_weights.keys())
    weights = [family_weights[label] for label in labels]
    return rng.choices(labels, weights=weights, k=1)[0]


# =============================================================================
# PLAN BUILDER
# =============================================================================


def create_rewrite_plan(
    selected_claim: SelectedClaim,
    article_id: str,
    sample_id: str,
    rng: random.Random,
) -> RewritePlan:
    """Build a structured :class:`RewritePlan` from a selected claim.

    Parameters
    ----------
    selected_claim:
        A claim object (duck-typed via :class:`SelectedClaim`) that
        specifies the claim type, surface span, sentence index, and
        importance / editability / diversity scores.
    article_id:
        The BENI v2 article identifier this claim belongs to (stored
        as ``original_id`` in the plan).
    sample_id:
        Unique sample identifier for this rewrite instance.
    rng:
        Seeded random number generator for reproducible difficulty
        sampling.

    Returns
    -------
    RewritePlan
        A frozen plan ready to be executed by ``BanglaRewriter``.

    Raises
    ------
    ValueError
        If ``selected_claim.claim_type`` is not one of the recognised
        claim types.
    """
    # Map claim type → rewrite family
    family = CLAIM_TYPE_TO_FAMILY.get(selected_claim.claim_type)
    if family is None:
        raise ValueError(
            f"Unknown claim_type '{selected_claim.claim_type}'. "
            f"Expected one of {list(CLAIM_TYPE_TO_FAMILY.keys())}."
        )

    # Difficulty from family-aware distribution
    difficulty = _select_difficulty_for_family(family, rng)

    # Generate descriptions from templates
    span = selected_claim.span_text
    desired_change = DESIRED_CHANGE_TEMPLATES[family].format(span=span)
    expected_changed_claim = EXPECTED_CHANGED_TEMPLATES[family].format(span=span)
    expected_scope = SCOPE_BY_FAMILY[family]

    return RewritePlan(
        sample_id=sample_id,
        original_id=article_id,
        rewrite_family=family,
        target_sentence_index=selected_claim.sentence_index,
        target_span=span,
        desired_change=desired_change,
        difficulty=difficulty,
        importance_score=selected_claim.importance_score,
        editability_score=selected_claim.editability_score,
        diversity_bonus=selected_claim.diversity_bonus,
        expected_scope=expected_scope,
        expected_changed_claim=expected_changed_claim,
    )
