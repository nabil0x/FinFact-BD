#!/usr/bin/env python3
"""Score and select the best claim for rewriting.

Implements a three-component scoring function:

    selection_score = 0.4 × importance + 0.4 × editability + 0.2 × diversity_bonus

The scorer respects per-family budgets and can track previously selected
families to encourage a balanced rewrite distribution across claim types.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from src.generation.claim_extraction import ExtractedClaim
from src.generation.perturbation_pipeline import (
    ATTRIBUTION_CUES,
    DECREASE_CUES,
    FINANCIAL_CONTEXT_CUES,
    INCREASE_CUES,
)

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================

# Base editability per claim type (higher = more suitable for automated rewriting).
# Numeric and policy claims are the most straightforward to rewrite because they
# have well-defined rewrite pairs; causal claims are the hardest.
EDITABILITY_BY_TYPE: Dict[str, float] = {
    "numeric": 0.90,
    "policy": 0.85,
    "temporal": 0.75,
    "entity": 0.65,
    "causal": 0.45,
}

# Default per-family budget when none is provided.
DEFAULT_FAMILY_BUDGET: Dict[str, int] = {
    "numeric": 2000,
    "policy": 2000,
    "entity": 2000,
    "temporal": 2000,
    "causal": 2000,
}

# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class SelectedClaim:
    """A claim selected for rewriting, annotated with component scores.

    Attributes:
        claim: The underlying extracted claim.
        importance_score: How important / salient the claim is in the article.
        editability_score: How suitable the claim is for automated rewriting.
        diversity_bonus: Bonus for choosing an underrepresented family.
        selection_score: Combined score used for ranking.
    """
    claim: ExtractedClaim
    importance_score: float
    editability_score: float
    diversity_bonus: float
    selection_score: float


# =============================================================================
# SCORING COMPONENTS
# =============================================================================


def _compute_importance(claim: ExtractedClaim, text: str) -> float:
    """Score the importance of a claim within its article.

    Factors considered:

    - **Position bias** — lead sentences (index 0, 1, 2) get higher weight
      since readers and journalists place key information early.
    - **Financial context density** — sentences mentioning many financial
      cues (টাকা, কোটি, লাভ, etc.) are more claim-rich.
    - **Directional language** — increase / decrease verbs signal a claim
      about change.
    - **Attribution** — cited sources raise the claim's evidentiary weight.
    - **Number presence** — any numeric token adds importance.
    - **Length penalty** — very short fragments are unlikely to be
      substantive claims.
    """
    score = 0.0

    # --- Position (lead bias) ---
    if claim.sentence_index == 0:
        score += 2.0
    elif claim.sentence_index == 1:
        score += 1.5
    elif claim.sentence_index == 2:
        score += 1.0
    else:
        score += max(0.0, 0.6 - 0.05 * (claim.sentence_index - 2))

    sentence = claim.sentence_text

    # --- Financial context density ---
    financial_hits = sum(1 for cue in FINANCIAL_CONTEXT_CUES if cue in sentence)
    score += min(2.0, financial_hits * 0.25)

    # --- Directional cues ---
    direction_hits = sum(1 for cue in INCREASE_CUES | DECREASE_CUES if cue in sentence)
    score += min(1.5, direction_hits * 0.4)

    # --- Attribution weight ---
    if any(cue in sentence for cue in ATTRIBUTION_CUES):
        score += 0.8

    # --- Number presence ---
    if re.search(r"[০-৯0-9]", sentence):
        score += 0.6

    # --- Length penalty ---
    if len(sentence.split()) < 6:
        score -= 0.5

    return round(max(0.0, min(5.0, score)), 4)


def _compute_editability(claim_type: str) -> float:
    """Return the base editability score for a claim type.

    The editability reflects how straightforward it is to rewrite a claim
    of this type automatically:

    - **numeric** (0.90): trivial to shift values up/down.
    - **policy** (0.85): well-defined directional rewrite pairs.
    - **temporal** (0.75): predictable month/year/relative-time shifts.
    - **entity** (0.65): needs an ontology lookup; moderate difficulty.
    - **causal** (0.45): complex sentence restructuring, hardest to automate.
    """
    return EDITABILITY_BY_TYPE.get(claim_type, 0.5)


def _compute_diversity_bonus(
    claim_type: str,
    previous_families: Optional[List[str]] = None,
    family_budget: Optional[Dict[str, int]] = None,
) -> float:
    """Compute a diversity bonus that favours underrepresented claim families.

    A claim type that has never been used receives a strong bonus (0.8);
    one that is used proportionally receives a neutral score (0.2);
    one that is at or near its budget gets no bonus (0.0).
    """
    if not previous_families:
        return 0.5

    if family_budget is None:
        family_budget = DEFAULT_FAMILY_BUDGET

    family_counts: Dict[str, int] = {}
    for fam in previous_families:
        family_counts[fam] = family_counts.get(fam, 0) + 1

    budget = family_budget.get(claim_type, 2000)
    count = family_counts.get(claim_type, 0)

    if count == 0:
        return 0.8   # never used → strong bonus
    if count < budget * 0.3:
        return 0.5   # underused → moderate bonus
    if count < budget * 0.7:
        return 0.2   # proportional use → neutral
    return 0.0       # near budget → no bonus


# =============================================================================
# SELECTION POLICY
# =============================================================================


def select_claim(
    claims: List[ExtractedClaim],
    text: str,
    family_budget: Optional[Dict[str, int]] = None,
    previous_families: Optional[List[str]] = None,
) -> Optional[SelectedClaim]:
    """Select the best claim for rewriting subject to diversity constraints.

    Each claim is scored as::

        selection_score = 0.4 * importance + 0.4 * editability + 0.2 * diversity_bonus

    The function returns the single highest-scoring claim that respects the
    per-family budget and diversity objectives.

    Args:
        claims: Candidate claims from :func:`extract_claims`.
        text: Full article text (used for importance context).
        family_budget: Per-family claim limit.
            Defaults to 2000 per family.
        previous_families: Family labels of previously selected claims.
            Used to compute the diversity bonus and to enforce budget limits.

    Returns:
        The best ``SelectedClaim``, or ``None`` if no claims are available
        or all eligible families have reached their budget.
    """
    if not claims:
        return None

    if family_budget is None:
        family_budget = DEFAULT_FAMILY_BUDGET

    if previous_families is None:
        previous_families = []

    # --- Budget enforcement ---
    total_by_type: Dict[str, int] = {}
    for fam in previous_families:
        total_by_type[fam] = total_by_type.get(fam, 0) + 1

    eligible = [
        c for c in claims
        if total_by_type.get(c.claim_type, 0) < family_budget.get(c.claim_type, 2000)
    ]

    if not eligible:
        return None

    # --- Score each eligible claim and pick the best ---
    best: Optional[SelectedClaim] = None
    best_score = -1.0

    for claim in eligible:
        importance = _compute_importance(claim, text)
        editability = _compute_editability(claim.claim_type)
        diversity = _compute_diversity_bonus(claim.claim_type,
                                             previous_families,
                                             family_budget)

        selection = 0.4 * importance + 0.4 * editability + 0.2 * diversity

        if selection > best_score:
            best_score = selection
            best = SelectedClaim(
                claim=claim,
                importance_score=importance,
                editability_score=editability,
                diversity_bonus=diversity,
                selection_score=round(selection, 4),
            )

    return best
