from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Protocol

from src.generation.metadata import RankedClaim, RewritePlan
from src.generation.models import InstructionModel
from src.generation.prompts import (
    PLAN_REVIEW_SCHEMA,
    PLANNING_SCHEMA,
    build_json_repair_prompt,
    build_plan_review_prompt,
    build_planning_prompt,
    build_planning_validation_repair_prompt,
)
from src.generation.utils import (
    CAUSAL_TERMS,
    entities_are_same_role,
    extract_json_payload,
    is_temporal_span,
    numeric_unit_mismatch_reason,
    numeric_values,
    numeric_values_equivalent,
    significant_numeric_scale_change,
    span_occurs_as_term,
)

logger = logging.getLogger(__name__)

FAMILIES = (
    "numerical_fact",
    "causal_inversion",
    "entity_replacement",
    "temporal_shift",
    "policy_reversal",
)


TYPE_TO_FAMILY = {
    "numerical": "numerical_fact",
    "causal": "causal_inversion",
    "entity": "entity_replacement",
    "temporal": "temporal_shift",
    "policy": "policy_reversal",
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
        validate_rewrite_plan(plan)
        logger.info("Planned %s rewrite for sentence %d", family, claim.sentence_index)
        return plan

    def _instruction(self, family: str, ranked_claim: RankedClaim) -> tuple[str, str]:
        claim = ranked_claim.claim
        if family == "numerical_fact":
            return (
                "Rewrite the selected claim so one financial number changes by a meaningful economic scale.",
                "The target numerical proposition crosses a scale boundary while all non-target facts remain unchanged.",
            )
        if family == "policy_reversal":
            return (
                "Rewrite the selected claim so the policy or market direction is reversed.",
                "The policy direction changes, but source, topic, and surrounding context stay fixed.",
            )
        if family == "entity_replacement":
            return (
                "Rewrite the selected claim by replacing the central entity with a wrong-belonging different-role entity.",
                "Only the target entity changes, and the replacement is not a same-class peer.",
            )
        if family == "temporal_shift":
            return (
                "Rewrite the selected claim by shifting its time reference to a contradictory time frame.",
                "The time anchor changes while the rest of the claim remains stable.",
            )
        if family == "causal_inversion":
            return (
                "Rewrite the selected claim so the same cause produces an opposite or implausible economic effect.",
                "The causal proposition changes without merely swapping clause order or introducing unrelated events.",
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
    plan_repair_attempts: int = 1
    plan_review_attempts: int = 0

    def create_plan(self, ranked_claim: RankedClaim) -> RewritePlan:
        prompt = build_planning_prompt(ranked_claim, self.allowed_families)
        raw = self.model.generate_text(prompt, self.temperature, self.seed, self.max_new_tokens)
        payload = self._extract_payload(raw)
        if not isinstance(payload, dict):
            raise ValueError("Planner JSON must be an object")
        last_error: ValueError | None = None
        for attempt in range(0, self.plan_repair_attempts + 1):
            try:
                plan = self._plan_from_payload(payload, ranked_claim)
                validate_rewrite_plan(plan)
                plan = self._review_valid_plan(plan, ranked_claim)
                logger.info("LLM planned %s rewrite for sentence %d", plan.family, ranked_claim.claim.sentence_index)
                return plan
            except ValueError as exc:
                last_error = exc
                if attempt >= self.plan_repair_attempts:
                    break
                logger.warning("Planner plan failed validation; requesting repair: %s", exc)
                repair_prompt = build_planning_validation_repair_prompt(
                    ranked_claim,
                    self.allowed_families,
                    payload,
                    str(exc),
                )
                repaired = self.model.generate_text(
                    repair_prompt,
                    0.0,
                    self.seed + 2000 + attempt,
                    self.max_new_tokens,
                )
                payload = self._extract_payload(repaired)
                if not isinstance(payload, dict):
                    raise ValueError("Planner repair JSON must be an object")
        raise last_error or ValueError("Planner did not produce a valid plan")

    def _review_valid_plan(self, plan: RewritePlan, ranked_claim: RankedClaim) -> RewritePlan:
        reviewed = plan
        for attempt in range(self.plan_review_attempts):
            prompt = build_plan_review_prompt(ranked_claim, reviewed, self.allowed_families)
            raw = self.model.generate_text(
                prompt,
                0.0,
                self.seed + 3000 + attempt,
                self.max_new_tokens,
            )
            try:
                payload = self._extract_payload(
                    raw,
                    schema=PLAN_REVIEW_SCHEMA,
                    task="plan review",
                    seed_offset=3000,
                )
            except ValueError as exc:
                logger.warning("Plan reviewer returned invalid JSON; keeping validated plan: %s", exc)
                return reviewed
            if not isinstance(payload, dict):
                logger.warning("Plan reviewer JSON was not an object; keeping validated plan")
                return reviewed
            decision = str(payload.get("decision") or "").strip().lower()
            reasons = self._review_reasons(payload)
            if decision == "pass":
                return self._annotate_review(reviewed, "pass", reasons)
            if decision != "repair":
                logger.warning("Plan reviewer returned unsupported decision %r; keeping validated plan", decision)
                return reviewed
            repaired_payload = payload.get("repaired_plan")
            if not isinstance(repaired_payload, dict):
                logger.warning("Plan reviewer requested repair without repaired_plan; keeping validated plan")
                return reviewed
            try:
                repaired = self._plan_from_payload(repaired_payload, ranked_claim)
                validate_rewrite_plan(repaired)
            except ValueError as exc:
                logger.warning("Plan reviewer repair failed validation; keeping validated plan: %s", exc)
                return reviewed
            reviewed = self._annotate_review(repaired, "repair", reasons)
            logger.info("Plan reviewer repaired %s plan reasons=%s", reviewed.family, reasons)
        return reviewed

    def _review_reasons(self, payload: Dict[str, Any]) -> List[str]:
        raw_reasons = payload.get("failure_reasons", [])
        if not isinstance(raw_reasons, list):
            return [str(raw_reasons)]
        return [str(reason) for reason in raw_reasons if str(reason).strip()]

    def _annotate_review(self, plan: RewritePlan, decision: str, reasons: List[str]) -> RewritePlan:
        constraints = dict(plan.verification_constraints)
        constraints["plan_review"] = {
            "decision": decision,
            "failure_reasons": reasons,
        }
        return replace(plan, verification_constraints=constraints)

    def _plan_from_payload(self, payload: Dict[str, Any], ranked_claim: RankedClaim) -> RewritePlan:
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
        return plan

    def _edit_scope(self, payload: Dict[str, Any]) -> str:
        scope = str(payload.get("locality") or payload.get("edit_scope") or "target_sentence").strip()
        if scope in {"target_sentence", "sentence"}:
            return "target_sentence"
        logger.warning("Planner produced non-local edit scope %r; normalizing to target_sentence", scope[:160])
        return "target_sentence"

    def _extract_payload(
        self,
        raw: str,
        schema: Dict[str, object] = PLANNING_SCHEMA,
        task: str = "rewrite planning",
        seed_offset: int = 1000,
    ) -> object:
        try:
            return extract_json_payload(raw)
        except ValueError as first_error:
            last_error: Exception = first_error
        for attempt in range(1, self.json_repair_attempts + 1):
            repair_prompt = build_json_repair_prompt(task, schema, raw)
            repaired = self.model.generate_text(
                repair_prompt,
                0.0,
                self.seed + seed_offset + attempt,
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
            plan_repair_attempts=int(cfg.get("plan_repair_attempts", 1)),
            plan_review_attempts=int(cfg.get("plan_review_attempts", 0)),
        )
    if backend != "heuristic":
        raise ValueError(f"Unsupported planner backend: {backend}")
    return PerturbationPlanner(families)


def validate_rewrite_plan(plan: RewritePlan) -> None:
    claim = plan.target_claim
    if plan.target_span and not span_occurs_as_term(claim.sentence, plan.target_span):
        raise ValueError(f"Planner target_span is not an exact claim term: {plan.target_span!r}")
    if plan.replacement and plan.target_span and plan.replacement.strip() == plan.target_span.strip():
        raise ValueError("Planner replacement must differ from target_span")
    if plan.family == "numerical_fact":
        if not numeric_values(claim.sentence):
            raise ValueError("Numerical plan requires a numeric target claim")
        if is_temporal_span(plan.target_span):
            raise ValueError("Numerical plan target_span appears temporal; use temporal_shift")
        if plan.replacement and not numeric_values(plan.replacement):
            raise ValueError("Numerical plan replacement must contain a numeric value")
        if plan.replacement:
            unit_reason = numeric_unit_mismatch_reason(plan.target_span, plan.replacement)
            if unit_reason:
                raise ValueError(f"Numerical plan replacement has incompatible units: {unit_reason}")
        if plan.replacement and numeric_values_equivalent(plan.target_span, plan.replacement):
            raise ValueError("Numerical plan replacement is value-equivalent to target_span")
        if plan.replacement and not significant_numeric_scale_change(plan.target_span, plan.replacement):
            raise ValueError("Numerical plan replacement is not a significant scale contradiction")
    if plan.family == "policy_reversal":
        if not (claim.policies or any(term in claim.sentence for term in ("বৃদ্ধি", "হ্রাস", "কম", "বাড়", "প্রয়োজন", "বাধা"))):
            raise ValueError("Policy reversal requires a policy or directional claim")
    if plan.family == "entity_replacement":
        if not claim.entities:
            raise ValueError("Entity replacement requires at least one extracted entity")
        if plan.replacement and entities_are_same_role(plan.target_span, plan.replacement):
            raise ValueError("Entity replacement must not use a same-role peer entity")
    if plan.family == "temporal_shift" and not (claim.dates or numeric_values(claim.sentence)):
        raise ValueError("Temporal shift requires a date or time-like numeric anchor")
    if plan.family == "causal_inversion" and not any(term in claim.sentence for term in CAUSAL_TERMS):
        raise ValueError("Causal inversion requires a causal claim")
