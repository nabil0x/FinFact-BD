from __future__ import annotations

from src.generation.metadata import Article, RewritePlan
from src.generation.utils import context_window


PROMPT_VERSION = "claim-rewrite-v1"

SYSTEM_INSTRUCTION = (
    "You are a controlled Bangla financial news rewriting engine. "
    "You must only execute the provided rewrite plan."
)

VARIANT_INSTRUCTIONS = (
    "Use a direct, conservative journalistic rewrite.",
    "Keep the same reporting tone, but make the planned factual edit more explicit.",
    "Use concise Bangla news prose and avoid adding any new background information.",
)


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
        "- Output the complete rewritten article, not only the sentence.\n\n"
        f"Original complete article:\n{article.text}\n\n"
        "Complete rewritten article:"
    )
