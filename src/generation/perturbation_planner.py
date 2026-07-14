from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

from src.generation.metadata import RankedClaim, RewritePlan

logger = logging.getLogger(__name__)

FAMILIES = (
    "numerical_fact",
    "policy_reversal",
    "entity_replacement",
    "temporal_shift",
    "causal_inversion",
)


TYPE_TO_FAMILY = {
    "numerical": "numerical_fact",
    "policy": "policy_reversal",
    "entity": "entity_replacement",
    "temporal": "temporal_shift",
    "causal": "causal_inversion",
}


@dataclass(frozen=True)
class PerturbationPlanner:
    allowed_families: List[str]

    def create_plan(self, ranked_claim: RankedClaim) -> RewritePlan:
        claim = ranked_claim.claim
        family = TYPE_TO_FAMILY.get(claim.claim_type)
        if family not in self.allowed_families:
            raise ValueError(f"Claim type {claim.claim_type!r} cannot be planned")
        instruction, expected = self._instruction(family, ranked_claim)
        constraints = {
            "target_sentence_index": claim.sentence_index,
            "preserve_all_other_sentences": True,
            "preserve_unrelated_facts": True,
            "forbid_new_entities_outside_target": True,
            "forbid_new_numbers_outside_target": True,
            "forbid_new_dates_outside_target": True,
            "minimum_semantic_similarity": 0.74,
            "minimum_contradiction_score": 0.55,
            "selected_claim_scores": ranked_claim.to_dict(),
        }
        plan = RewritePlan(
            family=family,
            target_claim=claim,
            edit_instruction=instruction,
            edit_scope="target_sentence",
            expected_change=expected,
            verification_constraints=constraints,
        )
        logger.info("Planned %s rewrite for sentence %d", family, claim.sentence_index)
        return plan

    def _instruction(self, family: str, ranked_claim: RankedClaim) -> tuple[str, str]:
        claim = ranked_claim.claim
        if family == "numerical_fact":
            return (
                "Rewrite the selected claim so one financial number changes to a plausible but false value.",
                "The target numerical proposition changes while all non-target facts remain unchanged.",
            )
        if family == "policy_reversal":
            return (
                "Rewrite the selected claim so the policy or market direction is reversed.",
                "The policy direction changes, but source, topic, and surrounding context stay fixed.",
            )
        if family == "entity_replacement":
            return (
                "Rewrite the selected claim by replacing the central financial entity with a comparable entity.",
                "Only the target entity in the selected proposition changes.",
            )
        if family == "temporal_shift":
            return (
                "Rewrite the selected claim by shifting its time reference to another plausible time.",
                "The time anchor changes while the rest of the claim remains stable.",
            )
        if family == "causal_inversion":
            return (
                "Rewrite the selected claim so the cause-effect relation is inverted or contradicted.",
                "The causal proposition changes without introducing unrelated events.",
            )
        raise ValueError(f"Unsupported rewrite family: {family}")


def build_planner(config: Dict[str, object] | None = None) -> PerturbationPlanner:
    cfg = config or {}
    families = list(cfg.get("allowed_families", FAMILIES))
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(f"Unknown perturbation families: {unknown}")
    return PerturbationPlanner(families)
