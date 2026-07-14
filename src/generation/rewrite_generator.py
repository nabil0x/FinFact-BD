from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List

from src.generation.metadata import Article, GeneratedRewrite, GenerationParams, RewritePlan
from src.generation.models import GenerationModel
from src.generation.prompts import PROMPT_VERSION, build_rewrite_prompt

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
        rewritten_article = self._clean_output(outputs[0])
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
                rewritten_article=self._clean_output(output),
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
            for output, prompt, temperature, seed in zip(outputs, prompts, temperatures, seeds)
        ]

    def _clean_output(self, output: str) -> str:
        text = output.strip()
        fences = ("```", "'''")
        for fence in fences:
            if text.startswith(fence) and text.endswith(fence):
                text = text[len(fence) : -len(fence)].strip()
        for prefix in ("Complete rewritten article:", "পুনর্লিখিত নিবন্ধ:"):
            if text.startswith(prefix):
                text = text[len(prefix) :].strip()
        return text
