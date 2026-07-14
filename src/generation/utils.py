from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from src.generation.metadata import SentenceSpan


SENTENCE_RE = re.compile(r"[^।!?]+(?:[।!?]+|$)")
NUMBER_RE = re.compile(r"[০-৯0-9]+(?:[.,][০-৯0-9]+)?%?")
DATE_RE = re.compile(
    r"(?:19\d{2}|20\d{2}|[০-৯]{4}|"
    r"জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|"
    r"সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|আজ|গতকাল|আগামীকাল|"
    r"গত বছর|চলতি বছর|আগামী বছর|গত মাস|চলতি মাস|আগামী মাস)"
)

ENTITY_TERMS = {
    "বাংলাদেশ ব্যাংক", "অর্থ মন্ত্রণালয়", "বাণিজ্য মন্ত্রণালয়", "এনবিআর",
    "জাতীয় রাজস্ব বোর্ড", "ডিএসই", "ঢাকা স্টক এক্সচেঞ্জ", "সিএসই",
    "আইএমএফ", "বিশ্বব্যাংক", "এডিবি", "সরকার", "সিটি ব্যাংক",
    "ব্র্যাক ব্যাংক", "ইসলামি ব্যাংক", "ডাচ-বাংলা ব্যাংক", "বিকাশ",
    "নগদ", "রকেট", "মাহিন্দ্রা", "গ্রামীণফোন", "রবি",
}

POLICY_TERMS = {
    "সুদের হার", "নীতিগত সুদের হার", "কর", "ভ্যাট", "শুল্ক",
    "সাবসিডি", "দাম", "মূল্য", "নীতিমালা", "নির্দেশনা",
    "আরোপ", "প্রত্যাহার", "বাড়িয়েছে", "কমিয়েছে", "বৃদ্ধি", "হ্রাস",
}

CAUSAL_TERMS = {
    "কারণে", "ফলে", "সুতরাং", "অতএব", "যেহেতু", "এর ফলে", "এ কারণে",
}

FINANCIAL_TERMS = {
    "টাকা", "কোটি", "লাখ", "ডলার", "ব্যাংক", "ঋণ", "সুদ", "লাভ",
    "ক্ষতি", "প্রবৃদ্ধি", "বিনিয়োগ", "রপ্তানি", "আমদানি", "শেয়ার",
    "সূচক", "বাজার", "রাজস্ব", "মুদ্রাস্ফীতি", "রিজার্ভ",
}


def sentence_spans(text: str) -> List[SentenceSpan]:
    spans: List[SentenceSpan] = []
    for idx, match in enumerate(SENTENCE_RE.finditer(text)):
        sentence = match.group().strip()
        if sentence:
            spans.append(SentenceSpan(idx, match.start(), match.end(), sentence))
    return spans


def extract_numbers(text: str) -> List[str]:
    return [match.group() for match in NUMBER_RE.finditer(text)]


def extract_dates(text: str) -> List[str]:
    return [match.group() for match in DATE_RE.finditer(text)]


def extract_terms(text: str, terms: Iterable[str]) -> List[str]:
    found = [term for term in terms if term and term in text]
    return sorted(set(found), key=lambda value: (text.find(value), value))


def extract_entities(text: str) -> List[str]:
    entities = extract_terms(text, ENTITY_TERMS)
    suffix_matches = re.findall(r"[\u0980-\u09FF]+(?: ব্যাংক| মন্ত্রণালয়| কমিশন| কর্পোরেশন)", text)
    return sorted(set(entities + suffix_matches), key=lambda value: (text.find(value), value))


def extract_policies(text: str) -> List[str]:
    return extract_terms(text, POLICY_TERMS)


def contains_financial_language(text: str) -> bool:
    return any(term in text for term in FINANCIAL_TERMS)


def changed_sentence_indices(original: str, rewritten: str) -> List[int]:
    original_sentences = sentence_spans(original)
    rewritten_sentences = sentence_spans(rewritten)
    changed: List[int] = []
    max_len = max(len(original_sentences), len(rewritten_sentences))
    for idx in range(max_len):
        original_text = original_sentences[idx].text if idx < len(original_sentences) else ""
        rewritten_text = rewritten_sentences[idx].text if idx < len(rewritten_sentences) else ""
        if original_text.strip() != rewritten_text.strip():
            changed.append(idx)
    return changed


def context_window(text: str, sentence_index: int, radius: int = 1) -> str:
    spans = sentence_spans(text)
    if not spans:
        return text.strip()
    start = max(0, sentence_index - radius)
    end = min(len(spans), sentence_index + radius + 1)
    return " ".join(span.text for span in spans[start:end]).strip()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("Cosine vectors must be non-empty and same length")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def stable_sample_id(article_id: str, sentence_index: int, family: str, seed: int) -> str:
    raw = f"{article_id}:{sentence_index}:{family}:{seed}".encode("utf-8")
    return "rw_" + hashlib.sha1(raw).hexdigest()[:16]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def extract_json_payload(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for match in re.finditer(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL):
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue
    decoder = json.JSONDecoder()
    for idx, char in enumerate(cleaned):
        if char not in "{[":
            continue
        try:
            payload, _ = decoder.raw_decode(cleaned[idx:])
            return payload
        except json.JSONDecodeError:
            continue
    candidates = []
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_char)
        end = cleaned.rfind(close_char)
        if start != -1 and end > start:
            candidates.append(cleaned[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Model output did not contain valid JSON; excerpt={compact_excerpt(text)}")


def compact_excerpt(text: str, limit: int = 500) -> str:
    excerpt = " ".join(text.strip().split())
    if len(excerpt) <= limit:
        return excerpt
    return excerpt[:limit] + "..."
