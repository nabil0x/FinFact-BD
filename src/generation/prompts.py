from __future__ import annotations

import json

from src.generation.metadata import Article, RankedClaim, RewritePlan
from src.generation.utils import context_window


PROMPT_VERSION = "planning-guided-v6-plan-review"

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
        "Create a high-contrast numerical contradiction, not a cosmetic edit. Prefer crossing scale "
        "boundaries such as লাখ->কোটি, কোটি->লাখ/হাজার, ১ শতাংশ->১০ শতাংশ, ১৫০ ডলার->১৫০০ ডলার. "
        "If the original amount is small, inflate it sharply; if it is large, deflate it sharply. "
        "Keep the economic unit coherent and skip dates, years, fiscal years, and ordinal event dates. "
        "For count facts, keep the replacement as a count, not a percentage: use ১শ’টির বেশি->২শ’টির বেশি "
        "or ১শ’টির বেশি->৫শ’টির বেশি, never ১০০ শতাংশ or কোটি শতাংশ. "
        "Good examples: ৫০ লাখ টাকার->৫ কোটি টাকার; ৩ লাখ ডলার->৩০ লাখ ডলার; "
        "১ শতাংশ->১০ শতাংশ; ১৫০ মার্কিন ডলার->১৫০০ মার্কিন ডলার; ২০ কারখানা->২০০ কারখানা; "
        "১শ’টির বেশি->৫শ’টির বেশি. "
        "Bad examples: ৫০ লাখ->৪৫ লাখ, ১শ->১০০, ১শটির বেশি->১০০ শতাংশ, ১শটির বেশি->১ কোটি শতাংশ, ২০১৪-১৫->২০১৫-১৬."
    ),
    "entity_replacement": (
        "Replace the entity with a wrong-belonging or different-role actor, not a same-class peer. "
        "Avoid ADB->World Bank, one bank->another bank, or one regulator->another regulator. Prefer "
        "cross-role contradictions such as development bank->foreign state, regulator->private company, "
        "bank->telecom/company, or company->ministry when the sentence stays grammatical. If linked "
        "mentions in headline/context would expose the swap, choose another local entity or another family. "
        "Good examples: এডিবি->ইসরায়েল সরকার; বাংলাদেশ ব্যাংক->ওয়ালটন; সিটি ব্যাংক->রবি; "
        "বিএসইসি->বসুন্ধরা গ্রুপ; মাহিন্দ্রা->অর্থ মন্ত্রণালয়. "
        "Bad examples: এডিবি->বিশ্বব্যাংক, সিটি ব্যাংক->ব্র্যাক ব্যাংক, বিএসইসি->বাংলাদেশ ব্যাংক."
    ),
    "temporal_shift": (
        "Change the reporting time anchor so the claim becomes temporally false. Use only dates, months, "
        "years, fiscal years, deadlines, quarters, or reporting periods. Prefer meaningful shifts: current "
        "period->previous/future period, deadline moved earlier/later, fixed event date moved to a conflicting "
        "date. Do not rewrite numeric amounts as temporal facts. "
        "Good examples: চলতি জুলাই মাসে->গত ডিসেম্বর মাসে; ২০১৫-১৬ অর্থবছরে->২০১২-১৩ অর্থবছরে; "
        "গত মঙ্গলবার->আগামী মঙ্গলবার; ১৫ আগস্ট->২৫ আগস্ট; ডিসেম্বর শেষে->জুন শেষে. "
        "Bad examples: ৫০ লাখ->৫ কোটি, ৩ লাখ ডলার->৩০ লাখ ডলার."
    ),
    "policy_reversal": (
        "Reverse the policy or market direction with clear Bangla news phrasing. Map approval->rejection, "
        "implementation->suspension, withdrawal->reinstatement, increase->decrease, relief->restriction, "
        "barrier removed->barrier increased. Use direct replacements like অনুমোদন করেছে->অনুমোদন দেয়নি, "
        "কমিয়েছে->বাড়িয়েছে, চালুর বাধা কাটল->চালুর বাধা আরও বেড়েছে. Avoid vague invented phrases. "
        "Good examples: অনুমোদন করেছে->অনুমোদন দেয়নি; কমিয়েছে->বাড়িয়েছে; প্রত্যাহার করেছে->বহাল রেখেছে; "
        "চালুর বাধা কাটল->চালুর বাধা আরও বেড়েছে; ছাড় দেবে->ছাড় দেবে না; স্থগিত করেছে->কার্যকর করেছে. "
        "Bad examples: সিদ্ধান্ত পরিবর্তন করা হয়েছে, বিষয়টি পুনর্বিবেচনা করা হয়েছে, প্রক্রিয়া চলছে."
    ),
    "causal_inversion": (
        "Invert the economic logic, not the word order. Preserve the stated cause when possible and replace "
        "the effect with the opposite or implausible effect: import cost rises->reserves increase, revenue "
        "falls->budget capacity improves, inflation rises->consumer pressure eases, losses grow->profit rises. "
        "Do not merely swap clauses; the rewritten claim must assert a logically contradictory cause-effect link. "
        "Good examples: আমদানি ব্যয় বেড়ে যাওয়ার কারণে রিজার্ভে চাপ সৃষ্টি হয়েছে->আমদানি ব্যয় বেড়ে যাওয়ার কারণে রিজার্ভ বেড়েছে; "
        "মূল্যস্ফীতি বাড়ায় ভোক্তার চাপ বেড়েছে->মূল্যস্ফীতি বাড়ায় ভোক্তার চাপ কমেছে; "
        "রাজস্ব কমায় ঘাটতি বেড়েছে->রাজস্ব কমায় ঘাটতি কমেছে; লোকসান বাড়ায় মুনাফা কমেছে->লোকসান বাড়ায় মুনাফা বেড়েছে. "
        "Bad examples: রিজার্ভে চাপ সৃষ্টি হওয়ায় আমদানি ব্যয় বেড়েছে, because that only swaps cause and effect."
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

PLAN_REVIEW_SCHEMA = {
    "decision": "pass|repair",
    "failure_reasons": ["short_machine_readable_reason"],
    "review_notes": "one short reviewer note",
    "repaired_plan": "null if decision=pass; complete planning JSON object if decision=repair",
}


def _compact_plan_payload(plan: RewritePlan) -> dict[str, object]:
    return {
        "family": plan.family,
        "target_span": plan.target_span,
        "replacement": plan.replacement,
        "locality": plan.edit_scope,
        "edit_instruction": plan.edit_instruction,
        "expected_change": plan.expected_change,
        "verification_constraints": {
            key: value
            for key, value in plan.verification_constraints.items()
            if key
            in {
                "preserve_all_other_sentences",
                "forbid_new_entities_outside_target",
                "forbid_new_numbers_outside_target",
                "forbid_new_dates_outside_target",
            }
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


def build_plan_review_prompt(ranked_claim: RankedClaim, plan: RewritePlan, allowed_families: list[str]) -> str:
    claim = ranked_claim.claim
    rule_text = "\n".join(f"- {family}: {FAMILY_RULES[family]}" for family in allowed_families if family in FAMILY_RULES)
    return (
        "Review one structured Bangla financial misinformation rewrite plan.\n"
        "Do not rewrite the article. Decide whether the plan is ready, or repair it into a stronger plan.\n"
        "Return valid JSON only. Do not output Markdown, commentary, or analysis.\n"
        "The response must begin with { and end with }.\n\n"
        "Decision rules:\n"
        "- Use decision=\"pass\" only if the plan is specific, grammatical, local, and strongly contradictory.\n"
        "- Use decision=\"repair\" if the plan is valid JSON but weak, vague, same-role, same-meaning, or likely to produce awkward Bangla.\n"
        "- If repairing, repaired_plan must be a complete planning JSON object using the same schema as the original planner.\n"
        "- Keep target_span as a short exact span from the selected claim sentence.\n"
        "- Keep replacement short and directly usable for exact phrase replacement.\n"
        "- Do not choose a family outside the allowed families.\n\n"
        "Family review checklist:\n"
        "- numerical_fact: accept high-contrast scale contradictions. Keep count facts as counts, e.g. ১শ’টির বেশি->৫শ’টির বেশি. Allow price->crore changes. Reject count->percentage, weak/value-equivalent changes, scaled percentage phrases like কোটি শতাংশ, and incoherent money->percentage changes.\n"
        "- entity_replacement: replacement must be a different-role or wrong-belonging actor, not a same-role peer.\n"
        "- temporal_shift: target/replacement must be time anchors, not financial amounts.\n"
        "- policy_reversal: replacement must be a clear opposite direction, not vague wording like পুনর্বিবেচনা or পরিবর্তন.\n"
        "- causal_inversion: preserve the cause when possible and replace the effect with an opposite economic outcome; do not merely swap clause order.\n\n"
        "Allowed family rules:\n"
        f"{rule_text}\n\n"
        f"Selected claim sentence index: {claim.sentence_index}\n"
        f"Selected claim sentence: {claim.sentence}\n"
        f"Claim type: {claim.claim_type}\n"
        f"Entities: {claim.entities}\n"
        f"Numbers: {claim.numbers}\n"
        f"Policies: {claim.policies}\n"
        f"Dates: {claim.dates}\n\n"
        "Current plan JSON:\n"
        f"{json.dumps(_compact_plan_payload(plan), ensure_ascii=False, indent=2)}\n\n"
        f"Review JSON schema:\n{json.dumps(PLAN_REVIEW_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        f"Planning JSON schema for repaired_plan:\n{json.dumps(PLANNING_SCHEMA, ensure_ascii=False, indent=2)}\n\n"
        "Valid review JSON:"
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


def build_planning_validation_repair_prompt(
    ranked_claim: RankedClaim,
    allowed_families: list[str],
    invalid_payload: dict[str, object],
    validation_error: str,
) -> str:
    base_prompt = build_planning_prompt(ranked_claim, allowed_families)
    return (
        f"{base_prompt}\n\n"
        "The previous JSON plan was syntactically valid but failed validation.\n"
        f"Validation error: {validation_error}\n"
        "Previous JSON plan:\n"
        f"{json.dumps(invalid_payload, ensure_ascii=False, indent=2)}\n\n"
        "Repair requirements:\n"
        "- Return a new valid JSON plan only.\n"
        "- Fix the validation error directly.\n"
        "- Keep target_span as an exact short span from the selected claim sentence.\n"
        "- If numerical, choose a significant high-contrast contradiction; cross-unit changes are allowed when readable.\n"
        "- If the target is a count, keep it as a count, e.g. ১শ’টির বেশি -> ২শ’টির বেশি or ৫শ’টির বেশি; do not use percentages.\n"
        "- If entity replacement, do not use a same-role peer entity; choose a wrong-role actor instead.\n\n"
        "Valid repaired JSON:"
    )
