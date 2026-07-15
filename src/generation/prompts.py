from __future__ import annotations

import json

from src.generation.metadata import Article, RankedClaim, RewritePlan
from src.generation.utils import context_window


PROMPT_VERSION = "planning-guided-v3-sentence-only"

SYSTEM_INSTRUCTION = (
    "You are one component in a constrained Bangla financial misinformation "
    "generation system. You must only realize the provided rewrite plan."
)

VARIANT_INSTRUCTIONS = (
    "Use a direct, conservative journalistic rewrite.",
    "Keep the same reporting tone, but make the planned factual edit more explicit.",
    "Use concise Bangla news prose and avoid adding any new background information.",
)

CLAIM_EXTRACTION_SCHEMA = {
    "claims": [
        {
            "sentence_index": 0,
            "sentence": "exact source sentence",
            "claim": "short normalized factual proposition in English or Bangla",
            "type": "numerical|policy|entity|temporal|causal",
            "entities": ["entity"],
            "numbers": ["number"],
            "policies": ["policy term"],
            "dates": ["date"],
            "confidence": 0.0,
        }
    ]
}

PLANNING_SCHEMA = {
    "family": "numerical_fact|policy_reversal|entity_replacement|temporal_shift|causal_inversion",
    "target_span": "exact span in selected sentence",
    "replacement": "planned replacement or concise target description",
    "locality": "sentence",
    "edit_instruction": "one-sentence instruction for the rewriter",
    "expected_change": "what factual property changes",
    "verification_constraints": {
        "preserve_all_other_sentences": True,
        "forbid_new_entities_outside_target": True,
        "forbid_new_numbers_outside_target": True,
        "forbid_new_dates_outside_target": True,
    },
}


def build_rewrite_prompt(article: Article, plan: RewritePlan, attempt: int) -> str:
    variant = VARIANT_INSTRUCTIONS[(attempt - 1) % len(VARIANT_INSTRUCTIONS)]
    local_context = context_window(article.text, plan.target_claim.sentence_index, radius=1)
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Attempt guidance: {variant}\n\n"
        f"Headline:\n{article.headline}\n\n"
        f"Selected claim sentence index: {plan.target_claim.sentence_index}\n"
        f"Selected claim:\n{plan.target_claim.sentence}\n\n"
        f"Local context:\n{local_context}\n\n"
        f"Rewrite family: {plan.family}\n"
        f"Edit instruction: {plan.edit_instruction}\n"
        f"Expected change: {plan.expected_change}\n"
        f"Edit scope: {plan.edit_scope}\n\n"
        "Verification constraints:\n"
        "- Preserve journalistic style and Bangla financial news register.\n"
        "- Rewrite only the selected claim sentence.\n"
        "- Preserve all unrelated facts, entities, dates, numbers, quotes, and attributions.\n"
        "- Do not add background, explanations, or unsupported facts.\n"
        "- Output only one sentence: the rewritten selected claim sentence.\n"
        "- Do not output the full article.\n"
        "- Do not output Markdown, labels, analysis, or explanations.\n\n"
        "Rewritten selected sentence:"
    )


def build_claim_extraction_prompt(article: Article, max_claims: int) -> str:
    return (
        "Extract factual claims from this Bangla financial news article.\n"
        "Do not summarize the article. Do not extract keywords. Convert each "
        "claim-bearing sentence into one factual proposition record.\n"
        f"Return at most {max_claims} claims as valid JSON only.\n"
        "Do not output Markdown, commentary, or analysis. The response must begin with { and end with }.\n"
        "If no factual financial claims are found, return {\"claims\": []}.\n\n"
        f"JSON schema:\n{json.dumps(CLAIM_EXTRACTION_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"Headline:\n{article.headline}\n\n"
        f"Article:\n{article.text}\n\n"
        "Valid JSON:"
    )


def build_planning_prompt(ranked_claim: RankedClaim, allowed_families: list[str]) -> str:
    claim = ranked_claim.claim
    return (
        "Create a structured rewrite plan for one Bangla financial claim.\n"
        "Do not rewrite the article. Do not invent an article. Decide exactly "
        "what factual property should change and what must remain unchanged.\n"
        f"Allowed families: {', '.join(allowed_families)}\n"
        "Return valid JSON only. Do not output Markdown, commentary, or analysis.\n"
        "The response must begin with { and end with }.\n\n"
        f"Selected claim sentence index: {claim.sentence_index}\n"
        f"Selected claim sentence: {claim.sentence}\n"
        f"Claim type: {claim.claim_type}\n"
        f"Entities: {claim.entities}\n"
        f"Numbers: {claim.numbers}\n"
        f"Policies: {claim.policies}\n"
        f"Dates: {claim.dates}\n"
        f"Ranking metadata: {json.dumps(ranked_claim.to_dict(), ensure_ascii=False)}\n\n"
        f"JSON schema:\n{json.dumps(PLANNING_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Valid JSON:"
    )


def build_json_repair_prompt(task: str, schema: dict[str, object], raw_output: str) -> str:
    return (
        f"The previous {task} response was not valid JSON.\n"
        "Convert it into valid JSON matching the schema below.\n"
        "Return JSON only. Do not output Markdown, commentary, analysis, or code fences.\n"
        "If the previous response does not contain usable information for claim extraction, return {\"claims\": []}.\n\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Previous response:\n{raw_output[:3000]}\n\n"
        "Valid JSON:"
    )
