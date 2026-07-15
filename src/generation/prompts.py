from __future__ import annotations

import json

from src.generation.metadata import Article, RankedClaim, RewritePlan
from src.generation.utils import context_window


PROMPT_VERSION = "planning-guided-v4-rulebook"

SYSTEM_INSTRUCTION = (
    "You are one component in a constrained Bangla financial misinformation "
    "generation system. You must only realize the provided rewrite plan."
)

VARIANT_INSTRUCTIONS = (
    "Use a direct, conservative journalistic rewrite.",
    "Keep the same reporting tone, but make the planned factual edit more explicit.",
    "Use concise Bangla news prose and avoid adding any new background information.",
)

FAMILY_RULES = {
    "numerical_fact": (
        "Change a financial number by a meaningful scale, not a small adjustment. "
        "Skip dates. Preserve units coherently. Examples: ৫০ লাখ -> ৫ কোটি, ১ শতাংশ -> ১০ শতাংশ."
    ),
    "entity_replacement": (
        "Replace the entity with a different-role or wrong-belonging entity, not a same-class peer. "
        "If linked mentions exist in headline/context, they must remain consistent."
    ),
    "temporal_shift": (
        "Change the time frame so it contradicts the original time claim. "
        "Use dates, months, years, fiscal years, or reporting periods only."
    ),
    "policy_reversal": (
        "Reverse the policy direction clearly: approval vs rejection, increase vs decrease, "
        "implementation vs suspension, barrier removed vs barrier increased."
    ),
    "causal_inversion": (
        "Do not merely swap sentence order. Preserve the cause when possible and change the effect "
        "to an economically opposite or implausible effect."
    ),
}

FAMILY_PRIORITY = ["numerical_fact", "causal_inversion", "entity_replacement", "temporal_shift", "policy_reversal"]

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
    "target_span": "short exact span in selected sentence",
    "replacement": "short planned replacement",
    "locality": "target_sentence",
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
    family_rule = FAMILY_RULES.get(plan.family, "")
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Prompt version: {PROMPT_VERSION}\n"
        f"Attempt guidance: {variant}\n\n"
        f"Headline:\n{article.headline}\n\n"
        f"Selected claim sentence index: {plan.target_claim.sentence_index}\n"
        f"Selected claim:\n{plan.target_claim.sentence}\n\n"
        f"Local context:\n{local_context}\n\n"
        f"Rewrite family: {plan.family}\n"
        f"Family-specific rule: {family_rule}\n"
        f"Target span: {plan.target_span}\n"
        f"Replacement: {plan.replacement}\n"
        f"Edit instruction: {plan.edit_instruction}\n"
        f"Expected change: {plan.expected_change}\n"
        f"Edit scope: {plan.edit_scope}\n\n"
        "Verification constraints:\n"
        "- Preserve journalistic style and Bangla financial news register.\n"
        "- Rewrite only the selected claim sentence.\n"
        "- Preserve all unrelated facts, entities, dates, numbers, quotes, and attributions.\n"
        "- Do not add background, explanations, or unsupported facts.\n"
        "- Output only one sentence: the rewritten selected claim sentence.\n"
        "- Keep the rewritten sentence close to the original length.\n"
        "- If target span and replacement are provided, use the replacement exactly.\n"
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
    priority = [family for family in FAMILY_PRIORITY if family in allowed_families]
    rule_text = "\n".join(f"- {family}: {FAMILY_RULES[family]}" for family in allowed_families if family in FAMILY_RULES)
    return (
        "Create a structured rewrite plan for one Bangla financial claim.\n"
        "Do not rewrite the article. Do not invent an article. Decide exactly "
        "what factual property should change and what must remain unchanged.\n"
        f"Allowed families: {', '.join(allowed_families)}\n"
        f"Preferred family priority: {', '.join(priority)}\n"
        "Return valid JSON only. Do not output Markdown, commentary, or analysis.\n"
        "The response must begin with { and end with }.\n\n"
        "Output constraints:\n"
        "- Return one compact JSON object under 180 words.\n"
        "- target_span must be a short exact span from the selected claim sentence.\n"
        "- replacement must be short, preferably under 12 Bangla words.\n"
        "- locality must be exactly \"target_sentence\".\n"
        "- edit_instruction and expected_change must each be one concise sentence.\n"
        "- Do not put the full claim sentence inside locality.\n\n"
        "Family-specific planning rules:\n"
        f"{rule_text}\n\n"
        f"Selected claim sentence index: {claim.sentence_index}\n"
        f"Selected claim sentence: {claim.sentence}\n"
        f"Claim type: {claim.claim_type}\n"
        f"Entities: {claim.entities}\n"
        f"Numbers: {claim.numbers}\n"
        f"Policies: {claim.policies}\n"
        f"Dates: {claim.dates}\n"
        "Scores: "
        f"importance={ranked_claim.importance_score:.4f}, "
        f"editability={ranked_claim.editability_score:.4f}, "
        f"verification={ranked_claim.verification_score:.4f}, "
        f"locality={ranked_claim.locality_score:.4f}, "
        f"risk={ranked_claim.risk_score:.4f}, "
        f"overall={ranked_claim.overall_score:.4f}\n\n"
        f"JSON schema:\n{json.dumps(PLANNING_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Valid JSON:"
    )


def build_json_repair_prompt(task: str, schema: dict[str, object], raw_output: str) -> str:
    fallback = '{"claims": []}' if task == "claim extraction" else "{}"
    return (
        f"The previous {task} response was not valid JSON.\n"
        "Convert it into valid JSON matching the schema below.\n"
        "Return JSON only. Do not output Markdown, commentary, analysis, or code fences.\n"
        f"If the previous response does not contain usable information, return {fallback}.\n\n"
        f"JSON schema:\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Previous response:\n{raw_output[:3000]}\n\n"
        "Valid JSON:"
    )
