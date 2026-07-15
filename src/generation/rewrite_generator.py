from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.generation.metadata import Article, GeneratedRewrite, GenerationParams, RewritePlan
from src.generation.models import GenerationModel
from src.generation.prompts import PROMPT_VERSION, build_rewrite_prompt
from src.generation.utils import artifact_reasons, has_text_artifacts, sentence_spans, span_occurs_as_term

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
