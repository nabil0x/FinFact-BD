from __future__ import annotations

from typing import Any, Dict, Tuple

from src.generation.metadata import RewritePlan
from src.generation.utils import (
    CAUSAL_TERMS,
    entities_are_same_role,
    extract_dates,
    extract_entities,
    numeric_values,
    scaled_numeric_values,
    significant_numeric_scale_change,
    span_occurs_as_term,
)


POLICY_DIRECTION_TERMS = {
    "increase": ("বৃদ্ধি", "বাড়", "আরোপ", "উন্নীত", "চালু", "অনুমোদন", "প্রয়োজন আছে", "কাটল"),
    "decrease": ("হ্রাস", "কম", "প্রত্যাহার", "বাতিল", "স্থগিত", "বন্ধ", "প্রয়োজন নেই", "বাধা"),
}

EFFECT_DIRECTION_TERMS = {
    "positive": ("বেড়েছে", "বেড়েছে", "বৃদ্ধি", "লাভ", "উন্নতি", "স্থিতিশীল", "উদ্বৃত্ত", "বাড়বে", "বাড়বে"),
    "negative": ("চাপ", "কমেছে", "হ্রাস", "ক্ষতি", "সংকট", "ঘাটতি", "ব্যাহত", "পতন", "ঝুঁকি"),
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
    original_values = scaled_numeric_values(original)
    rewritten_values = scaled_numeric_values(rewritten)
    replacement_values = scaled_numeric_values(plan.replacement)
    details = {"original_values": original_values, "rewritten_values": rewritten_values, "replacement_values": replacement_values}
    if not original_values or not rewritten_values:
        return 0.0, False, "numeric_values_missing", details
    if replacement_values and not any(value in rewritten_values for value in replacement_values):
        return 0.25, False, "planned_numeric_replacement_missing", details
    if original_values == rewritten_values:
        return 0.0, False, "numeric_value_unchanged", details
    if plan.replacement and not significant_numeric_scale_change(plan.target_span, plan.replacement):
        return 0.35, False, "numeric_scale_change_too_weak", details
    if not plan.replacement and not significant_numeric_scale_change(original, rewritten):
        return 0.35, False, "numeric_scale_change_too_weak", details
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
    target_span = plan.target_span.strip()
    details = {
        "original_entities": sorted(original_entities),
        "rewritten_entities": sorted(rewritten_entities),
        "target_span": target_span,
        "replacement": replacement,
    }
    if replacement and entities_are_same_role(target_span, replacement):
        return 0.25, False, "entity_same_role_replacement", details
    if replacement and replacement not in rewritten:
        return 0.0, False, "planned_entity_replacement_missing", details
    if target_span and target_span in rewritten:
        return 0.25, False, "original_entity_still_present", details
    if replacement and replacement != target_span:
        return 1.0, True, "passed", details
    if original_entities != rewritten_entities:
        return 1.0, True, "passed", details
    return 0.0, False, "entity_unchanged", details


def _temporal_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    original_dates = set(extract_dates(original))
    rewritten_dates = set(extract_dates(rewritten))
    original_values = numeric_values(original)
    rewritten_values = numeric_values(rewritten)
    replacement = plan.replacement.strip()
    target_span = plan.target_span.strip()
    details = {"original_dates": sorted(original_dates), "rewritten_dates": sorted(rewritten_dates)}
    if replacement and replacement not in rewritten:
        return 0.25, False, "planned_temporal_replacement_missing", details
    if target_span and target_span in rewritten:
        return 0.25, False, "temporal_anchor_unchanged", details
    if original_dates != rewritten_dates or original_values != rewritten_values:
        return 1.0, True, "passed", details
    return 0.0, False, "temporal_anchor_unchanged", details


def _causal_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    has_causal_language = any(term in rewritten for term in CAUSAL_TERMS)
    original_direction = _causal_effect_direction(original)
    rewritten_direction = _causal_effect_direction(rewritten)
    replacement = plan.replacement.strip()
    details = {
        "causal_language": has_causal_language,
        "original_effect_direction": original_direction,
        "rewritten_effect_direction": rewritten_direction,
    }
    if not has_causal_language:
        return 0.0, False, "causal_marker_missing", details
    if original_direction and rewritten_direction and original_direction != rewritten_direction:
        return 1.0, True, "passed", details
    if replacement and replacement in rewritten and original.strip() != rewritten.strip():
        return 0.7, True, "passed", details
    return 0.25, False, "causal_effect_not_inverted", details


def _replacement_or_string_change(original: str, rewritten: str, plan: RewritePlan) -> Tuple[float, bool, str, Dict[str, Any]]:
    replacement = plan.replacement.strip()
    if replacement and replacement in rewritten:
        return 1.0, True, "passed", {}
    return 0.5, original.strip() != rewritten.strip(), "passed" if original.strip() != rewritten.strip() else "target_unchanged", {}


def _policy_direction(text: str) -> str:
    matches = [name for name, terms in POLICY_DIRECTION_TERMS.items() if any(term in text for term in terms)]
    return "|".join(matches)


def _causal_effect_direction(text: str) -> str:
    effect = _effect_clause(text)
    matches = [name for name, terms in EFFECT_DIRECTION_TERMS.items() if any(term in effect for term in terms)]
    return "|".join(matches)


def _effect_clause(text: str) -> str:
    positions = [(text.find(term), term) for term in CAUSAL_TERMS if term in text]
    positions = [(idx, term) for idx, term in positions if idx >= 0]
    if not positions:
        return text
    idx, term = min(positions)
    return text[idx + len(term) :]
