from __future__ import annotations

from typing import Any, Dict, Tuple

from src.generation.metadata import RewritePlan
from src.generation.utils import CAUSAL_TERMS, extract_dates, extract_entities, numeric_values, span_occurs_as_term


POLICY_DIRECTION_TERMS = {
    "increase": ("বৃদ্ধি", "বাড়", "আরোপ", "উন্নীত", "চালু", "অনুমোদন", "প্রয়োজন আছে", "কাটল"),
    "decrease": ("হ্রাস", "কম", "প্রত্যাহার", "বাতিল", "স্থগিত", "বন্ধ", "প্রয়োজন নেই", "বাধা"),
}


def intended_change_verdict(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    if original.strip() == rewritten.strip():
        return 0.0, False, "target_unchanged", {}
    family = plan.family
    if family == "numerical_fact":
        return _numeric_change(original, rewritten, plan)
    if family == "policy_reversal":
        return _policy_change(original, rewritten, plan)
    if family == "entity_replacement":
        return _entity_change(original, rewritten, plan)
    if family == "temporal_shift":
        return _temporal_change(original, rewritten, plan)
    if family == "causal_inversion":
        return _causal_change(original, rewritten, plan)
    return _replacement_or_string_change(original, rewritten, plan)


def _numeric_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    original_values = numeric_values(original)
    rewritten_values = numeric_values(rewritten)
    replacement_values = numeric_values(plan.replacement)
    details = {"original_values": original_values, "rewritten_values": rewritten_values, "replacement_values": replacement_values}
    if not original_values or not rewritten_values:
        return 0.0, False, "numeric_values_missing", details
    if replacement_values and not any(value in rewritten_values for value in replacement_values):
        return 0.25, False, "planned_numeric_replacement_missing", details
    if original_values == rewritten_values:
        return 0.0, False, "numeric_value_unchanged", details
    return 1.0, True, "passed", details


def _policy_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    original_direction = _policy_direction(original)
    rewritten_direction = _policy_direction(rewritten)
    original_values = numeric_values(original)
    rewritten_values = numeric_values(rewritten)
    details = {
        "original_direction": original_direction,
        "rewritten_direction": rewritten_direction,
        "original_values": original_values,
        "rewritten_values": rewritten_values,
    }
    replacement = plan.replacement.strip()
    if replacement and (replacement in rewritten or span_occurs_as_term(rewritten, replacement)):
        return 1.0, True, "passed", details
    if original_direction and rewritten_direction and original_direction != rewritten_direction:
        return 1.0, True, "passed", details
    if original_values and rewritten_values and original_values != rewritten_values:
        return 0.85, True, "passed", details
    return 0.25, False, "policy_direction_not_reversed", details


def _entity_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    original_entities = set(extract_entities(original))
    rewritten_entities = set(extract_entities(rewritten))
    replacement = plan.replacement.strip()
    details = {"original_entities": sorted(original_entities), "rewritten_entities": sorted(rewritten_entities)}
    if replacement and replacement in rewritten and replacement != plan.target_span:
        return 1.0, True, "passed", details
    if original_entities != rewritten_entities:
        return 1.0, True, "passed", details
    return 0.0, False, "entity_unchanged", details


def _temporal_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    original_dates = set(extract_dates(original))
    rewritten_dates = set(extract_dates(rewritten))
    original_values = numeric_values(original)
    rewritten_values = numeric_values(rewritten)
    details = {"original_dates": sorted(original_dates), "rewritten_dates": sorted(rewritten_dates)}
    if original_dates != rewritten_dates or original_values != rewritten_values:
        return 1.0, True, "passed", details
    return 0.0, False, "temporal_anchor_unchanged", details


def _causal_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    has_causal_language = any(term in rewritten for term in CAUSAL_TERMS)
    replacement = plan.replacement.strip()
    if replacement and replacement in rewritten:
        return 1.0, True, "passed", {"causal_language": has_causal_language}
    if original.strip() != rewritten.strip() and has_causal_language:
        return 0.75, True, "passed", {"causal_language": has_causal_language}
    return 0.25, False, "causal_relation_not_changed", {"causal_language": has_causal_language}


def _replacement_or_string_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    replacement = plan.replacement.strip()
    if replacement and replacement in rewritten:
        return 1.0, True, "passed", {}
    return 0.5, original.strip() != rewritten.strip(), "passed" if original.strip() != rewritten.strip() else "target_unchanged", {}


def _policy_direction(text: str) -> str:
    matches = [name for name, terms in POLICY_DIRECTION_TERMS.items() if any(term in text for term in terms)]
    return "|".join(matches)
