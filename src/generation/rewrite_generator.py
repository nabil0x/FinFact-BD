from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.generation.metadata import Article, GeneratedRewrite, GenerationParams, RewritePlan
from src.generation.models import GenerationModel
from src.generation.prompts import PROMPT_VERSION, build_rewrite_prompt
from src.generation.utils import (
    artifact_reasons,
    has_text_artifacts,
    replace_all_exact,
    replace_first_exact,
    sentence_spans,
    span_occurs_as_term,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteGenerator:
    model: GenerationModel
    max_new_tokens: int = 768

    def rewrite(
        self,
        article: Article,
        plan: RewritePlan,
        temperature: float,
        seed: int,
        attempt: int,
    ) -> GeneratedRewrite:
        controlled = self._controlled_rewrite(article, plan, temperature, seed, attempt)
        if controlled is not None:
            logger.info(
                "Controlled rewrite for article=%s sentence=%d family=%s attempt=%d",
                article.article_id,
                plan.target_claim.sentence_index,
                plan.family,
                attempt,
            )
            return controlled
        prompt = build_rewrite_prompt(article, plan, attempt)
        outputs = self.model.generate_batch(
            prompts=[prompt],
            temperatures=[temperature],
            seeds=[seed],
            max_new_tokens=self.max_new_tokens,
        )
        if len(outputs) != 1:
            raise RuntimeError(f"Generation model returned {len(outputs)} outputs for one prompt")
        rewritten_article = self._localize_output(article.text, outputs[0], plan)
        if not rewritten_article:
            raise RuntimeError("Generation model returned an empty rewrite")
        logger.info(
            "Generated rewrite for article=%s sentence=%d attempt=%d",
            article.article_id,
            plan.target_claim.sentence_index,
            attempt,
        )
        return GeneratedRewrite(
            rewritten_article=rewritten_article,
            prompt=prompt,
            params=GenerationParams(
                model_name=self.model.model_name,
                model_revision=self.model.model_revision,
                prompt_version=PROMPT_VERSION,
                temperature=temperature,
                seed=seed,
                attempt=attempt,
                max_new_tokens=self.max_new_tokens,
            ),
        )

    def _controlled_rewrite(
        self,
        article: Article,
        plan: RewritePlan,
        temperature: float,
        seed: int,
        attempt: int,
    ) -> GeneratedRewrite | None:
        if plan.family not in {"numerical_fact", "temporal_shift", "entity_replacement", "policy_reversal"}:
            return None
        target_span = plan.target_span.strip()
        replacement = plan.replacement.strip()
        if not target_span or not replacement or target_span == replacement:
            return None

        original_spans = sentence_spans(article.text)
        target_index = plan.target_claim.sentence_index
        if target_index >= len(original_spans):
            raise RuntimeError(f"Target sentence index {target_index} missing from original article")
        target = original_spans[target_index]
        if target_span not in target.text:
            return None

        if plan.family == "entity_replacement":
            rewritten_article = replace_all_exact(article.text, target_span, replacement)
            rewritten_headline = replace_all_exact(article.headline, target_span, replacement)
            quality_text = replace_first_exact(target.text, target_span, replacement)
        else:
            rewritten_sentence = replace_first_exact(target.text, target_span, replacement)
            rewritten_article = article.text[: target.start] + rewritten_sentence + article.text[target.end :]
            rewritten_headline = replace_first_exact(article.headline, target_span, replacement)
            quality_text = rewritten_sentence

        if rewritten_article == article.text:
            return None
        if has_text_artifacts(quality_text):
            raise RuntimeError(f"Controlled rewrite contains text artifacts: {artifact_reasons(quality_text)}")
        return GeneratedRewrite(
            rewritten_article=rewritten_article,
            rewritten_headline=rewritten_headline if rewritten_headline != article.headline else None,
            prompt=f"{PROMPT_VERSION}+controlled-realization:{plan.family}",
            params=GenerationParams(
                model_name=f"{self.model.model_name}+controlled-realization",
                model_revision=self.model.model_revision,
                prompt_version=f"{PROMPT_VERSION}+controlled-realization",
                temperature=temperature,
                seed=seed,
                attempt=attempt,
                max_new_tokens=0,
            ),
        )

    def rewrite_batch(
        self,
        articles: List[Article],
        plans: List[RewritePlan],
        temperatures: List[float],
        seeds: List[int],
        attempt: int,
    ) -> List[GeneratedRewrite]:
        if not (len(articles) == len(plans) == len(temperatures) == len(seeds)):
            raise ValueError("Batch inputs must have identical lengths")
        controlled = [
            self._controlled_rewrite(article, plan, temperature, seed, attempt)
            for article, plan, temperature, seed in zip(articles, plans, temperatures, seeds)
        ]
        if all(item is not None for item in controlled):
            return [item for item in controlled if item is not None]
        if any(item is not None for item in controlled):
            return [
                item if item is not None else self.rewrite(article, plan, temperature, seed, attempt)
                for item, article, plan, temperature, seed in zip(controlled, articles, plans, temperatures, seeds)
            ]
        prompts = [build_rewrite_prompt(article, plan, attempt) for article, plan in zip(articles, plans)]
        outputs = self.model.generate_batch(prompts, temperatures, seeds, self.max_new_tokens)
        if len(outputs) != len(prompts):
            raise RuntimeError(f"Generation model returned {len(outputs)} outputs for {len(prompts)} prompts")
        return [
            GeneratedRewrite(
                rewritten_article=self._localize_output(article.text, output, plan),
                prompt=prompt,
                params=GenerationParams(
                    model_name=self.model.model_name,
                    model_revision=self.model.model_revision,
                    prompt_version=PROMPT_VERSION,
                    temperature=temperature,
                    seed=seed,
                    attempt=attempt,
                    max_new_tokens=self.max_new_tokens,
                ),
            )
            for article, plan, output, prompt, temperature, seed in zip(articles, plans, outputs, prompts, temperatures, seeds)
        ]

    def _clean_output(self, output: str) -> str:
        text = output.strip()
        fences = ("```", "'''")
        for fence in fences:
            if text.startswith(fence) and text.endswith(fence):
                text = text[len(fence) : -len(fence)].strip()
        for prefix in (
            "Complete rewritten article:",
            "Rewritten selected sentence:",
            "পুনর্লিখিত নিবন্ধ:",
            "পুনর্লিখিত নির্বাচিত বাক্য:",
            "নির্বাচিত বাক্য:",
        ):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        text = text.strip(" \n\t\"'“”‘’")
        return text

    def _localize_output(self, original_article: str, output: str, plan: RewritePlan) -> str:
        if has_text_artifacts(output):
            raise RuntimeError(f"Generation output contains text artifacts: {artifact_reasons(output)}")
        cleaned = self._clean_output(output)
        original_spans = sentence_spans(original_article)
        target_index = plan.target_claim.sentence_index
        if target_index >= len(original_spans):
            raise RuntimeError(f"Target sentence index {target_index} missing from original article")

        rewritten_sentence = self._select_target_sentence(cleaned, plan)
        if not rewritten_sentence:
            raise RuntimeError("Generation model returned an empty target sentence")
        if has_text_artifacts(rewritten_sentence):
            raise RuntimeError(f"Localized sentence contains text artifacts: {artifact_reasons(rewritten_sentence)}")

        target = original_spans[target_index]
        rewritten_sentence = self._ensure_sentence_terminator(rewritten_sentence, target.text)
        return original_article[: target.start] + rewritten_sentence + original_article[target.end :]

    def _select_target_sentence(self, cleaned: str, plan: RewritePlan) -> str:
        generated_spans = sentence_spans(cleaned)
        if not generated_spans:
            return cleaned.strip()
        if len(generated_spans) == 1:
            return generated_spans[0].text.strip()
        replacement = plan.replacement.strip()
        if replacement:
            matches = [span.text.strip() for span in generated_spans if replacement in span.text or span_occurs_as_term(span.text, replacement)]
            if len(matches) == 1:
                return matches[0]
            raise RuntimeError("Generation output contains multiple sentences and no unique planned replacement sentence")
        return generated_spans[0].text.strip()

    def _ensure_sentence_terminator(self, rewritten_sentence: str, original_sentence: str) -> str:
        sentence = rewritten_sentence.strip()
        if sentence.endswith(("।", "!", "?")):
            return sentence
        original = original_sentence.strip()
        if original.endswith(("।", "!", "?")):
            return sentence + original[-1]
        return sentence
