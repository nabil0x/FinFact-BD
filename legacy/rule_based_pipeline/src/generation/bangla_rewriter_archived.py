from __future__ import annotations

import logging
import re
from typing import List

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.generation.claim_planning import RewritePlan

logger = logging.getLogger(__name__)

# =============================================================================
# BENGALI SENTENCE BOUNDARY
# =============================================================================

# Bengali danda (।) and other sentence-ending punctuation.
_SENTENCE_SPLIT_RE = re.compile(r"[^\u0964।!?]+(?:[।!?]+|$)")
_CONTEXT_WINDOW = 2  # sentences of context on each side of the target


def _sentence_spans(text: str) -> List[str]:
    """Split text into sentences on Bengali sentence boundaries."""
    return [m.group().strip() for m in _SENTENCE_SPLIT_RE.finditer(text) if m.group().strip()]


def _extract_context_sentences(
    sentences: List[str],
    target_idx: int,
    window: int = _CONTEXT_WINDOW,
) -> str:
    """Return the target sentence with surrounding context as a single block."""
    start = max(0, target_idx - window)
    end = min(len(sentences), target_idx + window + 1)
    context_parts: List[str] = []
    for i in range(start, end):
        prefix = ">>> " if i == target_idx else "    "
        context_parts.append(f"{prefix}{sentences[i]}")
    return "\n".join(context_parts)


# =============================================================================
# REWRITER
# =============================================================================


class BanglaRewriter:
    """Controlled sentence-level rewriter for Bengali financial articles.

    Loads a Bangla seq2seq model (e.g. ``csebuetnlp/banglat5`` or
    ``Vacaspati/BanglaByT5``), builds a prompt from the article and
    the rewrite plan, generates a rewritten version of the **target
    sentence only**, and splices it back into the original article.

    The rewriter is deliberately narrow: it rewrites exactly one
    sentence, preserving everything else verbatim.  Free-form article
    generation is outside its scope.
    """

    def __init__(self, model_name: str, device: str = "cuda") -> None:
        """Load the generation model and tokenizer.

        Parameters
        ----------
        model_name:
            HuggingFace model identifier, e.g.
            ``"csebuetnlp/banglat5"`` or ``"Vacaspati/BanglaByT5"``.
        device:
            Target device (``"cuda"`` or ``"cpu"``).
        """
        logger.info("Loading BanglaRewriter model: %s on %s", model_name, device)
        self._device = torch.device(device) if torch.cuda.is_available() and device == "cuda" else torch.device("cpu")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self._device)
        self._model.eval()
        logger.info("BanglaRewriter model loaded successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def rewrite(
        self,
        article_text: str,
        plan: RewritePlan,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        num_beams: int = 4,
    ) -> str:
        """Rewrite the targeted segment of *article_text* according to *plan*.

        Parameters
        ----------
        article_text:
            The full Bangla financial article.
        plan:
            A :class:`RewritePlan` specifying which sentence to rewrite
            and how.
        max_new_tokens:
            Maximum number of tokens to generate.
        temperature:
            Sampling temperature (lower = more deterministic).
        num_beams:
            Beam size for beam search.

        Returns
        -------
        str
            The full article with **only** the target sentence rewritten.
            All other sentences are returned verbatim.

        Raises
        ------
        IndexError
            If ``plan.target_sentence_index`` exceeds the sentence count.
        ValueError
            If the model produces an empty rewrite.
        """
        sentences = _sentence_spans(article_text)
        if plan.target_sentence_index >= len(sentences):
            raise IndexError(
                f"target_sentence_index {plan.target_sentence_index} "
                f"out of range for article with {len(sentences)} sentences"
            )

        # Build the prompt and generate
        prompt = self._build_prompt(article_text, plan)
        rewritten_sentence = self._generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
        )

        # Fall back to the original if generation produced nothing usable.
        if not rewritten_sentence or rewritten_sentence.isspace():
            logger.warning(
                "Empty rewrite for sentence %d; returning original",
                plan.target_sentence_index,
            )
            return article_text

        return self._apply_rewrite(
            original=article_text,
            rewritten_segment=rewritten_sentence,
            plan=plan,
        )

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_prompt(article_text: str, plan: RewritePlan) -> str:
        """Build the instruction prompt for the generation model.

        The prompt presents the article, identifies the target claim
        via the rewrite plan, and instructs the model to rewrite only
        that sentence.
        """
        sentences = _sentence_spans(article_text)
        context_block = _extract_context_sentences(sentences, plan.target_sentence_index)

        return (
            f"Rewrite only the targeted claim in this Bangla financial article.\n"
            f"\n"
            f"Article context (>>> marks the sentence to rewrite):\n"
            f"{context_block}\n"
            f"\n"
            f"Rewrite plan:\n"
            f"- Target span: {plan.target_span}\n"
            f"- Rewrite family: {plan.rewrite_family}\n"
            f"- Desired change: {plan.desired_change}\n"
            f"- Expected result: {plan.expected_changed_claim}\n"
            f"\n"
            f"Rules:\n"
            f"- Preserve journalistic style.\n"
            f"- Change exactly one factual proposition.\n"
            f"- Keep the rest of the article coherent.\n"
            f"- Do not add unrelated facts.\n"
            f"- Return only the rewritten sentence.\n"
            f"\n"
            f"Rewritten sentence:"
        )

    # ------------------------------------------------------------------
    # Generation internals
    # ------------------------------------------------------------------

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        num_beams: int = 4,
    ) -> str:
        """Run the model on *prompt* and decode the output."""
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._device)

        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            num_beams=num_beams,
            do_sample=temperature > 0.0,
            early_stopping=True,
        )

        # Skip the input part for seq2seq models (output has no input prefix).
        decoded = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return decoded.strip()

    # ------------------------------------------------------------------
    # Splice rewritten segment back
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_rewrite(
        original: str,
        rewritten_segment: str,
        plan: RewritePlan,
    ) -> str:
        """Replace the target sentence in *original* with *rewritten_segment*.

        The replacement uses the span boundaries determined by
        :func:`_sentence_spans` so that only the targeted sentence is
        affected.
        """
        # Re-split to get exact character spans of the target sentence.
        spans: List[tuple[int, int]] = []
        for m in _SENTENCE_SPLIT_RE.finditer(original):
            sent = m.group().strip()
            if sent:
                spans.append((m.start(), m.end()))

        if plan.target_sentence_index >= len(spans):
            logger.warning(
                "Target index %d out of range for %d spans; returning original",
                plan.target_sentence_index,
                len(spans),
            )
            return original

        start, end = spans[plan.target_sentence_index]
        original_segment = original[start:end].strip()

        # Avoid replacing with an identical segment (no-op guard).
        if rewritten_segment == original_segment:
            logger.info(
                "Rewritten segment identical to original for sentence %d; "
                "returning original article unchanged",
                plan.target_sentence_index,
            )
            return original

        return original[:start] + rewritten_segment + original[end:]
