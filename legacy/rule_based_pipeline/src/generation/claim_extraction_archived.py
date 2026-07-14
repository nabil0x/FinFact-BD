#!/usr/bin/env python3
"""Extract candidate financial claims from Bengali articles.

Identifies five claim types: numeric, policy, entity, temporal, causal.
Reuses the dictionaries and patterns defined in perturbation_pipeline.py
without importing private helpers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set, Tuple

from src.generation.perturbation_pipeline import (
    ATTRIBUTION_CUES,
    CAUSAL_REWRITE_PAIRS,
    CAUSAL_SIGNALS,
    DECREASE_CUES,
    ENTITY_SUFFIXES,
    FINANCIAL_CONTEXT_CUES,
    INCREASE_CUES,
    MONTH_NAMES,
    OUTCOME_REWRITE_PAIRS,
    POLICY_REWRITE_PAIRS,
    RELATIVE_TIME_REWRITE,
    WORD_CHARS,
    SentenceSpan,
    bn_to_en_number,
    load_entity_catalog,
)


@dataclass(frozen=True)
class ExtractedClaim:
    """A candidate financial claim extracted from a Bengali article sentence."""
    sentence_index: int
    span_start: int
    span_end: int
    span_text: str
    claim_type: str
    confidence: float
    sentence_text: str


# ---------------------------------------------------------------------------
# Private helpers (reimplemented from perturbation_pipeline private functions)
# ---------------------------------------------------------------------------

def _word_pattern(term: str) -> str:
    suffix_alt = '|'.join(re.escape(s) for s in ENTITY_SUFFIXES)
    return rf"(?<![{WORD_CHARS}]){re.escape(term)}(?:{suffix_alt})?(?![{WORD_CHARS}])"


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms if term)


def _find_phrase_span(text: str, phrase: str) -> Optional[Tuple[int, int, str]]:
    pattern = re.compile(_word_pattern(phrase))
    match = pattern.search(text)
    if not match:
        return None
    return match.start(), match.end(), match.group()


def _is_year_like(number_text: str) -> bool:
    digits = re.sub(r"[^0-9০-৯]", "", number_text)
    if len(digits) != 4:
        return False
    try:
        year = int(bn_to_en_number(digits))
    except ValueError:
        return False
    return 1900 <= year <= 2099


def _has_token_boundary(text: str, start: int, end: int) -> bool:
    left_ok = start == 0 or not re.match(rf"[{WORD_CHARS}]", text[start - 1])
    right_ok = end == len(text) or not re.match(rf"[{WORD_CHARS}]", text[end])
    return left_ok and right_ok


def split_sentences(text: str) -> List[SentenceSpan]:
    """Split Bengali text into SentenceSpan objects with character offsets."""
    spans: List[SentenceSpan] = []
    pattern = re.compile(r"[^।!?]+(?:[।!?]+|$)")
    for idx, match in enumerate(pattern.finditer(text)):
        sent = match.group()
        if sent.strip():
            spans.append(SentenceSpan(
                index=idx, start=match.start(), end=match.end(), text=sent,
            ))
    return spans


def _score_sentence_salience(sentence: SentenceSpan) -> float:
    """Score how claim-like a sentence is (position bias + cues)."""
    score = 0.0
    if sentence.index == 0:
        score += 1.8
    elif sentence.index == 1:
        score += 1.2
    elif sentence.index == 2:
        score += 0.7
    elif sentence.index > 2:
        score += max(0.0, 0.4 - (0.05 * (sentence.index - 2)))
    if _contains_any(sentence.text, FINANCIAL_CONTEXT_CUES):
        score += 1.0
    if _contains_any(sentence.text, ATTRIBUTION_CUES):
        score += 0.9
    if _contains_any(sentence.text, INCREASE_CUES | DECREASE_CUES):
        score += 0.7
    if re.search(r"[০-৯0-9]", sentence.text):
        score += 0.8
    if _contains_any(sentence.text, list(CAUSAL_SIGNALS["cause"]) + list(CAUSAL_SIGNALS["effect"])):
        score += 0.6
    if any(month in sentence.text for month in MONTH_NAMES):
        score += 0.5
    if re.search(r"(?<!\d)(?:19\d{2}|20\d{2}|[০-৯]{4})(?!\d)", sentence.text):
        score += 0.5
    if len(sentence.text.split()) < 6:
        score -= 0.3
    return round(score, 4)


# ---------------------------------------------------------------------------
# Extraction logic per claim type
# ---------------------------------------------------------------------------

def _extract_numeric_claims(sentence: SentenceSpan) -> List[ExtractedClaim]:
    """Extract numeric claims: numbers in financial context."""
    claims: List[ExtractedClaim] = []
    pattern = re.compile(r"[০-৯0-9]+(?:[.,][০-৯0-9]+)?%?")
    for match in pattern.finditer(sentence.text):
        span_text = match.group().strip()
        if not span_text or _is_year_like(span_text):
            continue
        window = sentence.text[max(0, match.start() - 24):min(len(sentence.text), match.end() + 24)]
        score = 0.0
        if _contains_any(window, FINANCIAL_CONTEXT_CUES):
            score += 2.0
        if _contains_any(window, INCREASE_CUES | DECREASE_CUES):
            score += 1.0
        if '%' in span_text or 'শতাংশ' in window:
            score += 2.0
        if any(t in window for t in ("কোটি", "লাখ", "দশমিক", "টাকা", "ডলার")):
            score += 1.0
        score += _score_sentence_salience(sentence)
        if score <= 0:
            continue
        confidence = round(min(0.97, 0.5 + score / 10.0), 4)
        claims.append(ExtractedClaim(
            sentence_index=sentence.index, span_start=match.start(),
            span_end=match.end(), span_text=span_text, claim_type="numeric",
            confidence=confidence, sentence_text=sentence.text,
        ))
    return claims


def _extract_policy_claims(sentence: SentenceSpan) -> List[ExtractedClaim]:
    """Extract policy claims: directional policy phrases (rate/tax changes)."""
    claims: List[ExtractedClaim] = []
    specific_terms = {"নীতিগত", "সুদের", "কর", "ভ্যাট", "শুল্ক", "সাবসিডি", "দাম", "মূল্য", "নীতি", "হার"}
    phrase_pairs = sorted(POLICY_REWRITE_PAIRS, key=lambda p: len(p[0]), reverse=True)
    for source, _target in phrase_pairs:
        span = _find_phrase_span(sentence.text, source)
        if not span:
            continue
        start, end, matched = span
        score = _score_sentence_salience(sentence)
        if _contains_any(source, specific_terms):
            score += 1.2
        if len(source) >= 8:
            score += 0.4
        if source in sentence.text[:120]:
            score += 0.2
        if score <= 0:
            continue
        confidence = round(min(0.96, 0.6 + score / 10.0), 4)
        claims.append(ExtractedClaim(
            sentence_index=sentence.index, span_start=start, span_end=end,
            span_text=matched, claim_type="policy", confidence=confidence,
            sentence_text=sentence.text,
        ))
        break
    return claims


def _extract_entity_claims(sentence: SentenceSpan) -> List[ExtractedClaim]:
    """Extract entity claims: named financial entities in context."""
    claims: List[ExtractedClaim] = []
    catalog, _ = load_entity_catalog()
    used_spans: List[Tuple[int, int]] = []
    for entity, entity_class in catalog:
        pattern = re.compile(_word_pattern(entity))
        for match in pattern.finditer(sentence.text):
            span = (match.start(), match.end())
            if not _has_token_boundary(sentence.text, *span):
                continue
            if any(not (span[1] <= s or span[0] >= e) for s, e in used_spans):
                continue
            used_spans.append(span)
            window = sentence.text[max(0, span[0] - 24):min(len(sentence.text), span[1] + 24)]
            score = _score_sentence_salience(sentence)
            if _contains_any(window, ATTRIBUTION_CUES):
                score += 0.8
            if _contains_any(window, FINANCIAL_CONTEXT_CUES):
                score += 0.5
            score += min(0.3, len(entity) / 100.0)
            if score <= 0:
                continue
            confidence = round(min(0.95, 0.5 + score / 10.0), 4)
            claims.append(ExtractedClaim(
                sentence_index=sentence.index, span_start=span[0], span_end=span[1],
                span_text=match.group(), claim_type="entity", confidence=confidence,
                sentence_text=sentence.text,
            ))
            break
    return claims


def _extract_temporal_claims(sentence: SentenceSpan) -> List[ExtractedClaim]:
    """Extract temporal claims: years, months, relative time references."""
    claims: List[ExtractedClaim] = []
    for source, _target in RELATIVE_TIME_REWRITE:
        if source not in sentence.text:
            continue
        span = _find_phrase_span(sentence.text, source)
        if not span:
            continue
        start, end, matched = span
        score = _score_sentence_salience(sentence) + 1.0
        confidence = round(min(0.95, 0.5 + score / 10.0), 4)
        claims.append(ExtractedClaim(
            sentence_index=sentence.index, span_start=start, span_end=end,
            span_text=matched, claim_type="temporal", confidence=confidence,
            sentence_text=sentence.text,
        ))
    year_match = re.search(r"(?<!\d)(?:19\d{2}|20\d{2}|[০-৯]{4})(?!\d)", sentence.text)
    if year_match:
        score = _score_sentence_salience(sentence) + 0.5
        confidence = round(min(0.90, 0.4 + score / 10.0), 4)
        claims.append(ExtractedClaim(
            sentence_index=sentence.index, span_start=year_match.start(),
            span_end=year_match.end(), span_text=year_match.group(),
            claim_type="temporal", confidence=confidence, sentence_text=sentence.text,
        ))
    for month in MONTH_NAMES:
        if month not in sentence.text:
            continue
        span = _find_phrase_span(sentence.text, month)
        if not span:
            continue
        start, end, matched = span
        score = _score_sentence_salience(sentence) + 0.8
        confidence = round(min(0.92, 0.4 + score / 10.0), 4)
        claims.append(ExtractedClaim(
            sentence_index=sentence.index, span_start=start, span_end=end,
            span_text=matched, claim_type="temporal", confidence=confidence,
            sentence_text=sentence.text,
        ))
    return claims


def _extract_causal_claims(sentence: SentenceSpan) -> List[ExtractedClaim]:
    """Extract causal claims: causal connectors and outcome phrases."""
    claims: List[ExtractedClaim] = []
    seen_spans: Set[Tuple[int, int]] = set()
    for source, _target in sorted(CAUSAL_REWRITE_PAIRS, key=lambda p: len(p[0]), reverse=True):
        if source not in sentence.text:
            continue
        span = _find_phrase_span(sentence.text, source)
        if not span:
            continue
        start, end, matched = span
        key = (start, end)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        score = _score_sentence_salience(sentence) + 1.0
        confidence = round(min(0.93, 0.5 + score / 10.0), 4)
        claims.append(ExtractedClaim(
            sentence_index=sentence.index, span_start=start, span_end=end,
            span_text=matched, claim_type="causal", confidence=confidence,
            sentence_text=sentence.text,
        ))
    for source, _target in OUTCOME_REWRITE_PAIRS:
        if source not in sentence.text:
            continue
        span = _find_phrase_span(sentence.text, source)
        if not span:
            continue
        start, end, matched = span
        key = (start, end)
        if key in seen_spans:
            continue
        seen_spans.add(key)
        score = _score_sentence_salience(sentence) + 0.8
        confidence = round(min(0.90, 0.4 + score / 10.0), 4)
        claims.append(ExtractedClaim(
            sentence_index=sentence.index, span_start=start, span_end=end,
            span_text=matched, claim_type="causal", confidence=confidence,
            sentence_text=sentence.text,
        ))
    return claims


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_claims(text: str) -> List[ExtractedClaim]:
    """Extract all candidate financial claims from a Bengali article.

    Runs five specialised extractors over every sentence and returns a
    deduplicated list sorted by sentence index then span position.

    Claim types extracted:

    - **numeric** — numbers in financial context (prices, percentages, amounts)
    - **policy** — policy-action verb phrases via POLICY_REWRITE_PAIRS
    - **entity** — named entities (banks, regulators, companies) in context
    - **temporal** — years, month names, relative time references
    - **causal** — causal connectors (কারণে, ফলে, সুতরাং) and outcomes

    Args:
        text: Raw Bengali article text.

    Returns:
        List of ExtractedClaim objects, sorted by (sentence_index, span_start).
        May be empty if no candidates are found.
    """
    sentences = split_sentences(text)
    all_claims: List[ExtractedClaim] = []
    seen: Set[Tuple[int, int, int, str]] = set()

    for sentence in sentences:
        for extractor in (
            _extract_numeric_claims, _extract_policy_claims,
            _extract_entity_claims, _extract_temporal_claims,
            _extract_causal_claims,
        ):
            for claim in extractor(sentence):
                key = (claim.sentence_index, claim.span_start, claim.span_end, claim.claim_type)
                if key not in seen:
                    seen.add(key)
                    all_claims.append(claim)

    all_claims.sort(key=lambda c: (c.sentence_index, c.span_start))
    return all_claims
