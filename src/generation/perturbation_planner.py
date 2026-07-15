from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from src.generation.metadata import RankedClaim, RewritePlan
from src.generation.models import InstructionModel
from src.generation.prompts import PLANNING_SCHEMA, build_json_repair_prompt, build_planning_prompt
from src.generation.utils import extract_json_payload

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


class RewritePlanner(Protocol):
    def create_plan(self, ranked_claim: RankedClaim) -> RewritePlan:
        """Create a structured perturbation plan for one selected claim."""


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
            target_span=self._target_span(family, ranked_claim),
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

    def _target_span(self, family: str, ranked_claim: RankedClaim) -> str:
        claim = ranked_claim.claim
        if family == "numerical_fact" and claim.numbers:
            return claim.numbers[0]
        if family == "policy_reversal" and claim.policies:
            return claim.policies[0]
        if family == "entity_replacement" and claim.entities:
            return claim.entities[0]
        if family == "temporal_shift" and claim.dates:
            return claim.dates[0]
        return claim.sentence


@dataclass(frozen=True)
class LLMPerturbationPlanner:
    model: InstructionModel
    allowed_families: List[str]
    max_new_tokens: int = 768
    temperature: float = 0.0
    seed: int = 42
    json_repair_attempts: int = 1

    def create_plan(self, ranked_claim: RankedClaim) -> RewritePlan:
        prompt = build_planning_prompt(ranked_claim, self.allowed_families)
        raw = self.model.generate_text(prompt, self.temperature, self.seed, self.max_new_tokens)
        payload = self._extract_payload(raw)
        if not isinstance(payload, dict):
            raise ValueError("Planner JSON must be an object")
        family = self._family(payload)
        constraints = self._constraints(payload, ranked_claim)
        plan = RewritePlan(
            family=family,
            target_claim=ranked_claim.claim,
            edit_instruction=str(payload.get("edit_instruction") or "").strip(),
            edit_scope=self._edit_scope(payload),
            expected_change=str(payload.get("expected_change") or "").strip(),
            verification_constraints=constraints,
            target_span=str(payload.get("target_span") or "").strip(),
            replacement=str(payload.get("replacement") or "").strip(),
            planner_model=self.model.model_name,
        )
        if not plan.edit_instruction or not plan.expected_change or not plan.target_span:
            raise ValueError("Planner JSON must include edit_instruction, expected_change, and target_span")
        logger.info("LLM planned %s rewrite for sentence %d", family, ranked_claim.claim.sentence_index)
        return plan

    def _edit_scope(self, payload: Dict[str, Any]) -> str:
        scope = str(payload.get("locality") or payload.get("edit_scope") or "target_sentence").strip()
        if scope in {"target_sentence", "sentence"}:
            return "target_sentence"
        logger.warning("Planner produced non-local edit scope %r; normalizing to target_sentence", scope[:160])
        return "target_sentence"

    def _extract_payload(self, raw: str) -> object:
        try:
            return extract_json_payload(raw)
        except ValueError as first_error:
            last_error: Exception = first_error
        for attempt in range(1, self.json_repair_attempts + 1):
            repair_prompt = build_json_repair_prompt("rewrite planning", PLANNING_SCHEMA, raw)
            repaired = self.model.generate_text(
                repair_prompt,
                0.0,
                self.seed + 1000 + attempt,
                self.max_new_tokens,
            )
            try:
                return extract_json_payload(repaired)
            except ValueError as exc:
                last_error = exc
                raw = repaired
        raise ValueError(f"Planner did not return valid JSON after repair: {last_error}")

    def _family(self, payload: Dict[str, Any]) -> str:
        family = str(payload.get("family") or payload.get("type") or "").strip()
        family = family.replace(" ", "_")
        aliases = {
            "numeric": "numerical_fact",
            "numerical": "numerical_fact",
            "entity": "entity_replacement",
            "policy": "policy_reversal",
            "temporal": "temporal_shift",
            "causal": "causal_inversion",
        }
        family = aliases.get(family, family)
        if family not in self.allowed_families:
            raise ValueError(f"Planner produced unsupported family: {family!r}")
        return family

    def _constraints(self, payload: Dict[str, Any], ranked_claim: RankedClaim) -> Dict[str, Any]:
        provided = payload.get("verification_constraints", {})
        if provided is not None and not isinstance(provided, dict):
            raise ValueError("verification_constraints must be a JSON object")
        constraints = dict(provided or {})
        constraints.update(
            {
                "target_sentence_index": ranked_claim.claim.sentence_index,
                "preserve_all_other_sentences": True,
                "preserve_unrelated_facts": True,
                "forbid_new_entities_outside_target": True,
                "forbid_new_numbers_outside_target": True,
                "forbid_new_dates_outside_target": True,
                "selected_claim_scores": ranked_claim.to_dict(),
            }
        )
        return constraints


def build_planner(
    config: Dict[str, object] | None = None,
    model: InstructionModel | None = None,
) -> RewritePlanner:
    cfg = config or {}
    families = list(cfg.get("allowed_families", FAMILIES))
    unknown = [family for family in families if family not in FAMILIES]
    if unknown:
        raise ValueError(f"Unknown perturbation families: {unknown}")
    backend = str(cfg.get("backend", "heuristic"))
    if backend == "llm_json":
        if model is None:
            raise ValueError("planner.backend=llm_json requires models.planner")
        return LLMPerturbationPlanner(
            model=model,
            allowed_families=families,
            max_new_tokens=int(cfg.get("max_new_tokens", 768)),
            temperature=float(cfg.get("temperature", 0.0)),
            seed=int(cfg.get("seed", 42)),
            json_repair_attempts=int(cfg.get("json_repair_attempts", 1)),
        )
    if backend != "heuristic":
        raise ValueError(f"Unsupported planner backend: {backend}")
    return PerturbationPlanner(families)
