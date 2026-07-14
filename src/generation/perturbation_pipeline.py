#!/usr/bin/env python3
"""
FinFact-BD: Perturbation Pipeline for Bengali Financial Misinformation Detection
Generates 20K samples (10K original + 10K perturbed) from BENI v2 dataset.
"""

import csv
import io
import json
from collections import Counter
from difflib import SequenceMatcher
import random
import re
import sys
import zstandard as zstd
from pathlib import Path
from typing import Iterable, List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
from functools import lru_cache
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add parent directory to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_config, get_data_dir, get_output_dir, get_beni_v2_path

# =============================================================================
# CONFIGURATION
# =============================================================================

config = get_config()
DATA_DIR = get_data_dir()
BENI_V2_PATH = get_beni_v2_path()
OUTPUT_DIR = get_output_dir()
OUTPUT_DIR.mkdir(exist_ok=True)

# Frozen benchmark release marker.
DATASET_VERSION = "1.0"
DATASET_RELEASE_TAG = f"FinFact-BD-v{DATASET_VERSION}"
DATASET_RELEASE_DATE = "2026-07-14"
DATASET_RELEASE_STATE = "frozen"

# Sampling config
NUM_SAMPLES = int(config['sampling']['num_samples'])
TARGET_PER_TYPE = int(config['perturbation']['target_per_type'])
RANDOM_SEED = int(config['sampling']['random_seed'])
TRAIN_SPLIT = float(config['training'].get('train_split', 0.8))
VAL_SPLIT = float(config['training'].get('val_split', 0.1))
TEST_SPLIT = float(config['training'].get('test_split', 0.1))

# Financial categories and sectors to include
FINANCIAL_CATEGORIES = set(config['sampling'].get('financial_categories', ['Economy']))
FINANCIAL_SECTORS = {
    "trade_commerce", "energy_power", "construction_housing", 
    "agriculture", "telecommunication", "financial_institutions",
    "banking", "insurance", "stock_market", "remittance"
}

# =============================================================================
# BENGALI FINANCIAL DICTIONARIES
# =============================================================================

# Directional flipping: positive <-> negative sentiment words
SENTIMENT_DICT = {
    "positive": [
        "বৃদ্ধি", "উন্নতি", "লাভ", "সমৃদ্ধি", "প্রবৃদ্ধি", "উচ্চ", "বেশি",
        "ভালো", "সফল", "শক্তিশালী", "স্থিতিশীল", "ইতিবাচক", "আশাব্যঞ্জক",
        "বৃদ্ধিপ্রাপ্ত", "বেড়েছে", "উন্নীত", "প্রগতিশীল", "মজবুত",
        "স্বাস্থ্যকর", "লাভজনক", "সফলতা", "অগ্রগতি", "ক্ষেত্রে উন্নতি"
    ],
    "negative": [
        "হ্রাস", "পতন", "ক্ষতি", "মন্দা", "ঝুঁকি", "কম", "নেতিবাচক",
        "দুর্বল", "ক্ষয়", "সংকট", "সমস্যা", "বিপদ", "অবনতি", "কমেছে",
        "পতিত", "দুর্বলতা", "ঝুঁকিপূর্ণ", "ক্ষতিকর", "বিপর্যস্ত",
        "অস্থির", "মন্দাচক্র", "অবসাদ", "ঘাটতি", "অভাব"
    ]
}

# Numerical perturbation patterns
NUMBER_PATTERN = re.compile(r'[০-৯]+(?:\.[০-৯]+)?%?|[0-9]+(?:\.[0-9]+)?%?')

# Bengali number mapping
BN_DIGITS = {'০': '0', '১': '1', '২': '2', '৩': '3', '৪': '4', 
             '৫': '5', '৬': '6', '৭': '7', '৮': '8', '৯': '9'}
EN_DIGITS = {v: k for k, v in BN_DIGITS.items()}

# Causal signal words
CAUSAL_SIGNALS = {
    "cause": ["কারণে", "যোগে", "ফলে", "তাই", "সুতরাং", "অতএব", "কারণ", 
              "যেহেতু", "তাই বলে", "এর ফলে", "এ কারণে", "এ প্রেক্ষিতে"],
    "effect": ["হয়েছে", "বৃদ্ধি পেয়েছে", "কমেছে", "পরিবর্তন হয়েছে",
               "প্রভাবিত হয়েছে", "ক্ষতিগ্রস্ত হয়েছে", "উন্নত হয়েছে"]
}

# Entity swap dictionaries
ENTITY_SWAP = {
    "companies": [
        ("স্কয়ার ফার্মাসিউটিক্স", "ওরিয়ন ফার্মা"),
        ("বেক্সিমকো", "অলিম্পিক ইন্ডাস্ট্রিজ"),
        ("গ্রামীণফোন", "রবি"),
        ("ব্র্যাক ব্যাংক", "প্রাইম ব্যাংক"),
        ("ইসলামি ব্যাংক", "ইউনাইটেড কমার্শিয়াল ব্যাংক"),
        ("ডাচ-বাংলা ব্যাংক", "মার্কেটাইল ব্যাংক"),
        ("মেঘনা পেট্রোলিয়াম", "পদ্মা অয়েল"),
        ("বাটা", "এপেক্স"),
        ("র‍্যাক সেরামিক্স", "রিয়াজুদ্দিন"),
        ("ওয়ালটন", "স্যামসাং"),
        ("প্রান", "ইউসিবি"),
        ("ইন্ডেক্স", "বেকন"),
    ],
    "banks": [
        ("ডাচ-বাংলা ব্যাংক", "আইএফআইসি ব্যাংক"),
        ("সোনালী ব্যাংক", "পূবালি ব্যাংক"),
        ("ট্রাস্ট ব্যাংক", "ন্যাশনাল ব্যাংক"),
        ("আরব বাংলা ব্যাংক", "এক্সিম ব্যাংক"),
    ],
    "remittance_services": [
        ("বিকাশ", "নগদ"),
        ("নগদ", "বিকাশ"),
        ("রকেট", "বিকাশ"),
        ("উপায়", "বিকাশ"),
        ("মানি গ্রাম", "বিকাশ"),
    ]
}

# Sentiment amplifiers
AMPLIFIERS = {
    "strong_positive": ["অত্যন্ত", "বিশাল", "চমৎকার", "অসাধারণ", "অভূতপূর্ব"],
    "strong_negative": ["ভয়াবহ", "বিপর্যয়কর", "দুর্ভোষাময়", "বিনাশকর", "আপোসহীন"],
    "moderate": ["কিছুটা", "সামান্য", "অল্প", "মাঝারি"]
}

# =============================================================================
# FACT-AWARE PERTURBATION HELPERS
# =============================================================================

WORD_CHARS = r"0-9A-Za-z\u0980-\u09FF"
ENTITY_SUFFIXES = (
    "গুলোর", "গুলোকে", "গুলো", "দের", "টির", "টি", "কে", "তে", "য়ে", "য়ে",
    "এর", "ের", "র", "য়", "য়", "ে"
)

FACT_AWARE_PERTURBATION_TYPES = [
    "numerical_fact_change",
    "policy_reversal",
    "entity_replacement",
    "temporal_shift",
    "causal_inversion",
]

PERTURBATION_REASONING = {
    "numerical_fact_change": "numerical_fraud",
    "policy_reversal": "policy_reversal",
    "entity_replacement": "entity_confusion",
    "temporal_shift": "temporal_distortion",
    "causal_inversion": "causal_fallacy",
}

PERTURBATION_DIFFICULTY = {
    "numerical_fact_change": "medium",
    "policy_reversal": "easy",
    "entity_replacement": "medium",
    "temporal_shift": "easy",
    "causal_inversion": "hard",
}

DIFFICULTY_TO_HOPS = {
    "easy": 1,
    "medium": 2,
    "hard": 3,
}

DIFFICULTY_WEIGHTS = {
    "easy": 0.25,
    "medium": 0.45,
    "hard": 0.30,
}

FAMILY_DIFFICULTY_WEIGHTS = {
    "policy_reversal": {
        "easy": 0.65,
        "medium": 0.25,
        "hard": 0.10,
    },
    "entity_replacement": {
        "easy": 0.45,
        "medium": 0.45,
        "hard": 0.10,
    },
    "numerical_fact_change": {
        "easy": 0.05,
        "medium": 0.65,
        "hard": 0.30,
    },
    "temporal_shift": {
        "easy": 0.05,
        "medium": 0.65,
        "hard": 0.30,
    },
    "causal_inversion": {
        "easy": 0.00,
        "medium": 0.30,
        "hard": 0.70,
    },
}

FAMILY_PRIORITY = {
    "policy_reversal": 0,
    "numerical_fact_change": 1,
    "entity_replacement": 2,
    "temporal_shift": 3,
    "causal_inversion": 4,
}

FAMILY_CONTRADICTION_WEIGHTS = {
    "numerical_fact_change": 0.34,
    "policy_reversal": 0.39,
    "entity_replacement": 0.24,
    "temporal_shift": 0.22,
    "causal_inversion": 0.42,
}

VALIDATION_CONFIG = config.get("quality_filter", {})
SEMANTIC_SIMILARITY_MIN = float(VALIDATION_CONFIG.get("semantic_similarity_min", 0.68))
# Fact-aware perturbations should stay close to the source text, so the upper
# similarity bound is intentionally permissive and only blocks near-duplicates.
SEMANTIC_SIMILARITY_MAX = float(VALIDATION_CONFIG.get("semantic_similarity_max", 0.999))
FLUENCY_MIN_SCORE = float(VALIDATION_CONFIG.get("fluency_min_score", 0.65))
CONTRADICTION_MIN_SCORE = float(VALIDATION_CONFIG.get("contradiction_min_score", VALIDATION_CONFIG.get("contradiction_threshold", 0.55)))
CONTRADICTION_MIN_SCORE_BY_DIFFICULTY = {
    "easy": float(VALIDATION_CONFIG.get("easy_contradiction_min_score", 0.43)),
    "medium": float(VALIDATION_CONFIG.get("medium_contradiction_min_score", 0.52)),
    "hard": float(VALIDATION_CONFIG.get("hard_contradiction_min_score", 0.58)),
}
MAX_GENERATION_ATTEMPTS = int(config.get("perturbation", {}).get("max_generation_attempts", 6))
ENABLE_NLI_VERIFIER = bool(VALIDATION_CONFIG.get("use_nli_verifier", False))

FINANCIAL_CONTEXT_CUES = {
    "টাকা", "টাকার", "কোটি", "লাখ", "দাম", "মূল্য", "শতাংশ", "হার", "সুদ",
    "নীতি", "নীতিগত", "বাজেট", "রাজস্ব", "লাভ", "ক্ষতি", "আয়", "আয়",
    "প্রবৃদ্ধি", "মুদ্রাস্ফীতি", "রিজার্ভ", "ডলার", "শেয়ার", "শেয়ার",
    "সূচক", "ভ্যাট", "কর", "শুল্ক", "সাবসিডি", "রপ্তানি", "আমদানি",
    "বিনিয়োগ", "বিনিয়োগ", "বিতরণ", "ঋণ", "ঋণের", "সঞ্চয়", "সঞ্চয়"
}

ATTRIBUTION_CUES = {
    "বলেন", "বলেন,", "জানান", "জানায়", "জানায়", "জানিয়েছে", "জানিয়েছে",
    "মতে", "অনুযায়ী", "অনুসারে", "প্রকাশ", "উল্লেখ", "অভিমত", "দাবি করেন"
}

SOURCE_ENTITY_CLASSES = {
    "central_bank",
    "regulator",
    "international_org",
    "government_body",
}

INCREASE_CUES = {
    "বাড়িয়েছে", "বাড়িয়েছে", "বাড়িয়েছে", "বাড়বে", "বেড়েছে", "বৃদ্ধি",
    "বৃদ্ধি পেয়েছে", "বৃদ্ধি পায়", "উন্নতি", "উঠেছে", "উর্ধ্বমুখী"
}

DECREASE_CUES = {
    "কমিয়েছে", "কমিয়েছে", "কমাবে", "কমেছে", "হ্রাস", "হ্রাস পেয়েছে",
    "পতন", "অবনতি", "নিম্নমুখী", "কমে গেছে"
}

POLICY_REWRITE_PAIRS = [
    ("নীতিগত সুদের হার বাড়িয়েছে", "নীতিগত সুদের হার কমিয়েছে"),
    ("নীতিগত সুদের হার বাড়িয়েছে", "নীতিগত সুদের হার কমিয়েছে"),
    ("নীতিগত সুদের হার কমিয়েছে", "নীতিগত সুদের হার বাড়িয়েছে"),
    ("নীতিগত সুদের হার কমিয়েছে", "নীতিগত সুদের হার বাড়িয়েছে"),
    ("সুদের হার বাড়িয়েছে", "সুদের হার কমিয়েছে"),
    ("সুদের হার বাড়িয়েছে", "সুদের হার কমিয়েছে"),
    ("সুদের হার কমিয়েছে", "সুদের হার বাড়িয়েছে"),
    ("সুদের হার কমিয়েছে", "সুদের হার বাড়িয়েছে"),
    ("সুদের হার বৃদ্ধি পেয়েছে", "সুদের হার হ্রাস পেয়েছে"),
    ("সুদের হার বৃদ্ধি পেয়েছে", "সুদের হার হ্রাস পেয়েছে"),
    ("সুদের হার হ্রাস পেয়েছে", "সুদের হার বৃদ্ধি পেয়েছে"),
    ("সুদের হার হ্রাস হয়েছে", "সুদের হার বৃদ্ধি হয়েছে"),
    ("কর কমিয়েছে", "কর বাড়িয়েছে"),
    ("কর কমিয়েছে", "কর বাড়িয়েছে"),
    ("কর বাড়িয়েছে", "কর কমিয়েছে"),
    ("কর বাড়িয়েছে", "কর কমিয়েছে"),
    ("ভ্যাট কমিয়েছে", "ভ্যাট বাড়িয়েছে"),
    ("ভ্যাট কমিয়েছে", "ভ্যাট বাড়িয়েছে"),
    ("ভ্যাট বাড়িয়েছে", "ভ্যাট কমিয়েছে"),
    ("ভ্যাট বাড়িয়েছে", "ভ্যাট কমিয়েছে"),
    ("শুল্ক কমিয়েছে", "শুল্ক বাড়িয়েছে"),
    ("শুল্ক বাড়িয়েছে", "শুল্ক কমিয়েছে"),
    ("সাবসিডি বাড়িয়েছে", "সাবসিডি কমিয়েছে"),
    ("সাবসিডি কমিয়েছে", "সাবসিডি বাড়িয়েছে"),
    ("দাম বেড়েছে", "দাম কমেছে"),
    ("দাম বেড়েছে", "দাম কমেছে"),
    ("মূল্য বেড়েছে", "মূল্য কমেছে"),
    ("মূল্য বেড়েছে", "মূল্য কমেছে"),
    ("পরিমাণ বেড়েছে", "পরিমাণ কমেছে"),
    ("পরিমাণ বেড়েছে", "পরিমাণ কমেছে"),
    ("বৃদ্ধি পেয়েছে", "হ্রাস পেয়েছে"),
    ("বৃদ্ধি পেয়েছে", "হ্রাস পেয়েছে"),
    ("হ্রাস পেয়েছে", "বৃদ্ধি পেয়েছে"),
    ("হ্রাস পেয়েছে", "বৃদ্ধি পেয়েছে"),
    ("বেড়েছে", "কমেছে"),
    ("বেড়েছে", "কমেছে"),
    ("উঠেছে", "নেমেছে"),
    ("নেমেছে", "উঠেছে"),
    ("বাড়ছে", "কমছে"),
    ("কমছে", "বাড়ছে"),
    ("বাড়ল", "কমল"),
    ("কমল", "বাড়ল"),
    ("উন্নতি", "অবনতি"),
    ("অবনতি", "উন্নতি"),
    ("উচ্চ", "নিম্ন"),
    ("নিম্ন", "উচ্চ"),
    ("বেশি", "কম"),
    ("কম", "বেশি"),
    ("লাভ", "ক্ষতি"),
    ("ক্ষতি", "লাভ"),
    ("সফল", "ব্যর্থ"),
    ("ব্যর্থ", "সফল"),
    ("শক্তিশালী", "দুর্বল"),
    ("দুর্বল", "শক্তিশালী"),
    ("ইতিবাচক", "নেতিবাচক"),
    ("নেতিবাচক", "ইতিবাচক"),
    ("বাড়িয়েছে", "কমিয়েছে"),
    ("বাড়িয়েছে", "কমিয়েছে"),
    ("কমিয়েছে", "বাড়িয়েছে"),
    ("কমিয়েছে", "বাড়িয়েছে"),
    ("বৃদ্ধি করেছে", "হ্রাস করেছে"),
    ("হ্রাস করেছে", "বৃদ্ধি করেছে"),
    ("বাড়াবে", "কমাবে"),
    ("কমাবে", "বাড়াবে"),
    ("বাতিল করেছে", "বৃদ্ধি করেছে"),
    ("আরোপ করেছে", "প্রত্যাহার করেছে"),
    ("প্রত্যাহার করেছে", "আরোপ করেছে"),
]

CAUSAL_REWRITE_PAIRS = [
    ("এর ফলে", "তার বিপরীতে"),
    ("এ কারণে", "তবুও"),
    ("কারণে", "সত্ত্বেও"),
    ("কারণ", "যদিও"),
    ("ফলে", "যদিও"),
    ("তাই", "তবুও"),
    ("সুতরাং", "তবুও"),
    ("অতএব", "তবুও"),
    ("যেহেতু", "যদিও"),
]

OUTCOME_REWRITE_PAIRS = [
    ("বৃদ্ধি পেয়েছে", "হ্রাস পেয়েছে"),
    ("বৃদ্ধি পায়", "হ্রাস পায়"),
    ("বাড়িয়েছে", "কমিয়েছে"),
    ("বাড়িয়েছে", "কমিয়েছে"),
    ("বেড়েছে", "কমেছে"),
    ("কমেছে", "বেড়েছে"),
    ("হ্রাস পেয়েছে", "বৃদ্ধি পেয়েছে"),
    ("হ্রাস পেয়েছে", "বৃদ্ধি পেয়েছে"),
    ("উন্নত হয়েছে", "পিছিয়ে গেছে"),
    ("ক্ষতিগ্রস্ত হয়েছে", "উন্নত হয়েছে"),
    ("পরিবর্তন হয়েছে", "পরিবর্তন হয়নি"),
    ("হয়েছে", "হয়নি"),
]

MONTH_REWRITE = {
    "জানুয়ারি": "ফেব্রুয়ারি",
    "ফেব্রুয়ারি": "মার্চ",
    "মার্চ": "এপ্রিল",
    "এপ্রিল": "মে",
    "মে": "জুন",
    "জুন": "জুলাই",
    "জুলাই": "আগস্ট",
    "আগস্ট": "সেপ্টেম্বর",
    "সেপ্টেম্বর": "অক্টোবর",
    "অক্টোবর": "নভেম্বর",
    "নভেম্বর": "ডিসেম্বর",
    "ডিসেম্বর": "জানুয়ারি",
}

RELATIVE_TIME_REWRITE = [
    ("গত বছর", "চলতি বছর"),
    ("চলতি বছর", "গত বছর"),
    ("এ বছর", "গত বছর"),
    ("এই বছর", "গত বছর"),
    ("আগামী বছর", "চলতি বছর"),
    ("গত মাসে", "আগামী মাসে"),
    ("চলতি মাসে", "গত মাসে"),
    ("এই মাসে", "গত মাসে"),
    ("আগামী মাসে", "গত মাসে"),
    ("গতকাল", "আজ"),
    ("আজ", "গতকাল"),
    ("আগামীকাল", "গতকাল"),
    ("সাম্প্রতিক", "পূর্ববর্তী"),
    ("পূর্ববর্তী", "সাম্প্রতিক"),
]

MONTH_NAMES = list(MONTH_REWRITE.keys())


@dataclass(frozen=True)
class SentenceSpan:
    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class EntityMention:
    start: int
    end: int
    text: str
    base_text: str
    entity_class: str


@dataclass
class PerturbationOutcome:
    text: str
    family: str
    operator: str
    reasoning_type: str
    difficulty: str
    changed_span_original: str
    changed_span_replacement: str
    changed_span_role: str
    sentence_index: int
    proposition_confidence: float
    ontology_class: str
    proposition_schema: Dict[str, object]


@dataclass(frozen=True)
class PropositionCandidate:
    family: str
    sentence_index: int
    span_start: int
    span_end: int
    span_text: str
    replacement_text: str
    role: str
    ontology_class: str
    score: float


@dataclass
class PerturbationPlan:
    primary_family: str
    difficulty: str
    hop_count: int
    family_sequence: List[str]
    proposition_graph: Dict[str, List[Dict[str, object]]]
    candidate_summary: Dict[str, int]
    planned_operations: List[Dict[str, object]]


@dataclass
class ValidationResult:
    passed: bool
    contradiction_score: float
    semantic_similarity: float
    fluency_score: float
    issues: List[str]


def _word_pattern(term: str) -> str:
    """Build a boundary-aware regex for a Bengali term."""
    return rf"(?<![{WORD_CHARS}]){re.escape(term)}(?:{'|'.join(re.escape(s) for s in ENTITY_SUFFIXES)})?(?![{WORD_CHARS}])"


def _sentence_spans(text: str) -> List[SentenceSpan]:
    """Split text into sentence spans while preserving offsets."""
    spans: List[SentenceSpan] = []
    pattern = re.compile(r"[^।!?]+(?:[।!?]+|$)")
    for idx, match in enumerate(pattern.finditer(text)):
        sent = match.group()
        if sent.strip():
            spans.append(SentenceSpan(index=idx, start=match.start(), end=match.end(), text=sent))
    return spans


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(term in text for term in terms if term)


def _sentence_salience_score(sentence: SentenceSpan) -> float:
    """Score how visible or claim-like a sentence is for human readers."""
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
    if _contains_any(sentence.text, CAUSAL_SIGNALS["cause"] + CAUSAL_SIGNALS["effect"]):
        score += 0.6
    if any(month in sentence.text for month in MONTH_NAMES):
        score += 0.5
    if re.search(r"(?<!\d)(?:19\d{2}|20\d{2}|[০-৯]{4})(?!\d)", sentence.text):
        score += 0.5

    # Very short fragments are rarely useful validation targets.
    if len(sentence.text.split()) < 6:
        score -= 0.3

    return round(score, 4)


def _replace_span(text: str, start: int, end: int, replacement: str) -> str:
    return text[:start] + replacement + text[end:]


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


def _find_phrase_span(text: str, phrase: str) -> Optional[Tuple[int, int, str]]:
    """Find the first boundary-aware span for a phrase."""
    pattern = re.compile(_word_pattern(phrase))
    match = pattern.search(text)
    if not match:
        return None
    return match.start(), match.end(), match.group()


def _split_entity_suffix(surface: str, base: str) -> Tuple[str, str]:
    """Split a matched entity surface form into canonical base + suffix."""
    if surface == base:
        return base, ""
    for suffix in sorted(ENTITY_SUFFIXES, key=len, reverse=True):
        if surface.endswith(suffix) and len(surface) > len(suffix):
            candidate_base = surface[:-len(suffix)]
            if candidate_base.startswith(base):
                return candidate_base, suffix
    return surface, ""


@lru_cache(maxsize=1)
def load_financial_ontology() -> Dict[str, List[str]]:
    """Load ontology-constrained replacement pools from local glossary files."""
    paths = get_config()['paths']

    def load_terms(path_key: str, column: str = 'Bengali') -> List[str]:
        path = Path(paths[path_key])
        if not path.exists():
            return []
        terms: List[str] = []
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                value = (row.get(column) or '').strip()
                if value:
                    terms.append(value)
        return terms

    central_banks = [
        "বাংলাদেশ ব্যাংক",
        "ফেডারেল রিজার্ভ",
        "রিজার্ভ ব্যাংক অব ইন্ডিয়া",
        "ব্যাংক অব ইংল্যান্ড",
        "ইউরোপীয় কেন্দ্রীয় ব্যাংক",
    ]
    regulators = [
        "বাংলাদেশ সিকিউরিটিজ অ্যান্ড এক্সচেঞ্জ কমিশন",
        "বিএসইসি",
        "জাতীয় রাজস্ব বোর্ড",
        "এনবিআর",
        "অর্থ মন্ত্রণালয়",
        "বাণিজ্য মন্ত্রণালয়",
    ]
    stock_exchanges = [
        "ঢাকা স্টক এক্সচেঞ্জ",
        "চট্টগ্রাম স্টক এক্সচেঞ্জ",
        "ডিএসই",
        "সিএসই",
        "স্টক এক্সচেঞ্জ",
    ]
    mobile_finance = [
        "বিকাশ",
        "নগদ",
        "রকেট",
        "উপায়",
    ]
    international_orgs = [
        "আইএমএফ",
        "আন্তর্জাতিক মুদ্রা তহবিল",
        "বিশ্বব্যাংক",
        "ওয়ার্ল্ড ব্যাংক",
        "এডিবি",
        "এশীয় উন্নয়ন ব্যাংক",
        "এআইআইবি",
    ]
    government_bodies = [
        "সরকার",
        "প্রধানমন্ত্রী",
        "মন্ত্রিসভা",
        "অর্থ মন্ত্রণালয়",
        "বাণিজ্য মন্ত্রণালয়",
        "শিল্প মন্ত্রণালয়",
    ]

    bank_entities: List[str] = []
    insurance_entities: List[str] = []
    listed_companies: List[str] = []

    dse_path = Path(paths['dse_companies'])
    if dse_path.exists():
        with open(dse_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                bn_name = (row.get('Bengali') or '').strip()
                sector = (row.get('Sector') or '').strip().lower()
                if not bn_name:
                    continue
                if 'bank' in sector:
                    bank_entities.append(bn_name)
                elif 'insurance' in sector:
                    insurance_entities.append(bn_name)
                else:
                    listed_companies.append(bn_name)

    banking_terms = load_terms('banking_terms')
    remittance_terms = load_terms('remittance_terms')
    glossary_terms = load_terms('financial_glossary')

    return {
        'central_bank': sorted(set(central_banks)),
        'regulator': sorted(set(regulators)),
        'stock_exchange': sorted(set(stock_exchanges)),
        'mobile_finance': sorted(set(mobile_finance)),
        'international_org': sorted(set(international_orgs)),
        'government_body': sorted(set(government_bodies)),
        'commercial_bank': sorted(set(bank_entities)),
        'insurance_company': sorted(set(insurance_entities)),
        'listed_company': sorted(set(listed_companies)),
        'banking_terms': sorted(set(banking_terms)),
        'remittance_terms': sorted(set(remittance_terms)),
        'glossary_terms': sorted(set(glossary_terms)),
    }


@lru_cache(maxsize=1)
def load_entity_catalog() -> Tuple[List[Tuple[str, str]], Dict[str, List[str]]]:
    """Return a flat phrase catalog and class-to-entities mapping."""
    ontology = load_financial_ontology()
    class_map: Dict[str, List[str]] = {
        key: value for key, value in ontology.items()
        if key in {
            'central_bank', 'regulator', 'stock_exchange', 'mobile_finance',
            'international_org', 'government_body', 'commercial_bank',
            'insurance_company', 'listed_company'
        }
    }

    catalog: List[Tuple[str, str]] = []
    for entity_class, entities in class_map.items():
        for entity in entities:
            catalog.append((entity, entity_class))

    catalog.sort(key=lambda item: len(item[0]), reverse=True)
    return catalog, class_map


def _entity_class_priority(entity_class: str) -> int:
    priority = {
        'central_bank': 0,
        'regulator': 1,
        'stock_exchange': 2,
        'commercial_bank': 3,
        'insurance_company': 4,
        'mobile_finance': 5,
        'listed_company': 6,
        'government_body': 7,
        'international_org': 8,
    }
    return priority.get(entity_class, 99)


def _pick_replacement_from_class(entity_class: str, current: str, rng: random.Random) -> Optional[str]:
    _, class_map = load_entity_catalog()
    candidates = [value for value in class_map.get(entity_class, []) if value != current]
    if not candidates:
        return None
    return rng.choice(candidates)


def _match_entity_mentions(text: str) -> List[EntityMention]:
    catalog, _ = load_entity_catalog()
    matches: List[EntityMention] = []
    used_spans: List[Tuple[int, int]] = []

    for entity, entity_class in catalog:
        pattern = re.compile(_word_pattern(entity))
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if not _has_token_boundary(text, *span):
                continue
            if any(not (span[1] <= s or span[0] >= e) for s, e in used_spans):
                continue
            used_spans.append(span)
            matches.append(
                EntityMention(
                    start=span[0],
                    end=span[1],
                    text=match.group(),
                    base_text=entity,
                    entity_class=entity_class,
                )
            )
            break

    matches.sort(key=lambda item: item.start)
    return matches


def _sentence_has_financial_context(sentence: str) -> bool:
    ontology = load_financial_ontology()
    cues = set(FINANCIAL_CONTEXT_CUES)
    cues.update(ontology['glossary_terms'][:100])
    cues.update(ontology['banking_terms'][:100])
    cues.update(ontology['remittance_terms'][:100])
    return _contains_any(sentence, cues)


def _score_numeric_candidate(sentence: SentenceSpan, start: int, end: int, span_text: str) -> float:
    score = 0.0
    sentence_text = sentence.text
    window = sentence_text[max(0, start - 24):min(len(sentence_text), end + 24)]
    if _contains_any(window, FINANCIAL_CONTEXT_CUES):
        score += 2.0
    if _contains_any(window, INCREASE_CUES | DECREASE_CUES):
        score += 1.0
    if '%' in span_text or 'শতাংশ' in window:
        score += 2.0
    if any(term in window for term in ["কোটি", "লাখ", "দশমিক", "টাকা", "ডলার"]):
        score += 1.0
    score += _sentence_salience_score(sentence)
    return round(score, 4)


def _format_number_like(original: str, value: float) -> str:
    is_bn = bool(re.search(r"[০-৯]", original))
    is_percent = '%' in original or 'শতাংশ' in original
    if abs(value - int(round(value))) < 1e-8:
        num_text = str(int(round(value)))
    else:
        num_text = f"{value:.1f}".rstrip('0').rstrip('.')
    if is_bn:
        num_text = en_to_bn_number(num_text)
    if is_percent and '%' in original:
        num_text += '%'
    return num_text


def _shift_numeric_value(
    sentence: str,
    span_text: str,
    rng: random.Random,
) -> Tuple[str, str]:
    raw = span_text.strip().rstrip('%')
    digits_only = re.sub(r"[^0-9০-৯]", "", raw)
    is_year = _is_year_like(raw)
    is_percent = '%' in span_text or 'শতাংশ' in sentence

    if raw.startswith('০') or re.search(r"[০-৯]", raw):
        parsed = bn_to_en_number(raw.replace(',', ''))
    else:
        parsed = raw.replace(',', '')

    try:
        current_value = float(parsed)
    except ValueError:
        current_value = None

    if current_value is None:
        return span_text, span_text

    if is_year:
        delta = rng.choice([-2, -1, 1, 2])
        new_value = max(1900, min(2099, int(round(current_value)) + delta))
    elif is_percent or len(digits_only) <= 3:
        delta = rng.uniform(0.5, 3.0)
        if _contains_any(sentence, INCREASE_CUES):
            new_value = max(0.1, current_value - delta)
        elif _contains_any(sentence, DECREASE_CUES):
            new_value = current_value + delta
        else:
            new_value = current_value + rng.choice([-1, 1]) * delta
    else:
        factor = rng.choice([
            rng.uniform(0.65, 0.85),
            rng.uniform(1.15, 1.65),
        ])
        new_value = max(0.1, current_value * factor)

    replacement = _format_number_like(span_text, new_value)
    return span_text, replacement


def _choose_best_numeric_candidate(text: str) -> Optional[Tuple[SentenceSpan, re.Match, float]]:
    best: Optional[Tuple[SentenceSpan, re.Match, float]] = None
    pattern = re.compile(r"[০-৯0-9]+(?:[.,][০-৯0-9]+)?%?")

    for sentence in _sentence_spans(text):
        if not _sentence_has_financial_context(sentence.text):
            continue
        for match in pattern.finditer(sentence.text):
            span_text = match.group().strip()
            if not span_text:
                continue
            if _is_year_like(span_text):
                # Keep year-like numbers for temporal manipulation.
                continue
            score = _score_numeric_candidate(sentence, match.start(), match.end(), span_text)
            if score <= 0:
                continue
            if best is None or score > best[2] or (score == best[2] and len(span_text) > len(best[1].group())):
                best = (sentence, match, score)
    return best


def _score_policy_candidate(sentence: SentenceSpan, source: str, target: str) -> float:
    score = _sentence_salience_score(sentence)
    source_specific_terms = (
        "নীতিগত",
        "সুদের",
        "কর",
        "ভ্যাট",
        "শুল্ক",
        "সাবসিডি",
        "দাম",
        "মূল্য",
        "নীতি",
        "হার",
    )
    if _contains_any(source, source_specific_terms):
        score += 1.2
    if len(source) >= 8:
        score += 0.4
    if source in sentence.text[:120]:
        score += 0.2
    return round(score, 4)


def _choose_policy_candidate(text: str) -> Optional[Tuple[SentenceSpan, str, str]]:
    best: Optional[Tuple[SentenceSpan, str, str, float]] = None
    phrase_pairs = sorted(POLICY_REWRITE_PAIRS, key=lambda item: len(item[0]), reverse=True)
    for sentence in _sentence_spans(text):
        for source, target in phrase_pairs:
            if _find_phrase_span(sentence.text, source):
                score = _score_policy_candidate(sentence, source, target)
                if best is None or score > best[3]:
                    best = (sentence, source, target, score)
    if best is None:
        return None
    return best[0], best[1], best[2]


def _choose_temporal_candidate(text: str) -> Optional[Tuple[SentenceSpan, str, str, str]]:
    best: Optional[Tuple[SentenceSpan, str, str, str, float]] = None
    for sentence in _sentence_spans(text):
        for source, target in RELATIVE_TIME_REWRITE:
            if source in sentence.text:
                score = _sentence_salience_score(sentence) + 2.0
                candidate = (sentence, source, target, "relative_time", score)
                if best is None or score > best[4]:
                    best = candidate

        year_match = re.search(r"(?<!\d)(?:19\d{2}|20\d{2}|[০-৯]{4})(?!\d)", sentence.text)
        if year_match:
            score = _sentence_salience_score(sentence) + 0.6
            candidate = (sentence, year_match.group(), "", "year", score)
            if best is None or score > best[4]:
                best = candidate

        for month in MONTH_NAMES:
            if month in sentence.text:
                score = _sentence_salience_score(sentence) + 1.2
                candidate = (sentence, month, MONTH_REWRITE[month], "month", score)
                if best is None or score > best[4]:
                    best = candidate
    if best is None:
        return None
    return best[0], best[1], best[2], best[3]


def _choose_causal_candidate(text: str) -> Optional[Tuple[SentenceSpan, str, str, Optional[Tuple[str, str]]]]:
    best: Optional[Tuple[SentenceSpan, str, str, Optional[Tuple[str, str]], float]] = None
    for sentence in _sentence_spans(text):
        connector_hit: Optional[Tuple[str, str]] = None
        for source, target in CAUSAL_REWRITE_PAIRS:
            if source in sentence.text:
                connector_hit = (source, target)
                break

        outcome_pair = None
        for outcome_source, outcome_target in OUTCOME_REWRITE_PAIRS:
            if outcome_source in sentence.text:
                outcome_pair = (outcome_source, outcome_target)
                break

        if connector_hit or outcome_pair:
            score = _sentence_salience_score(sentence)
            if connector_hit is not None:
                score += 1.0 + (0.2 * len(connector_hit[0]))
            if outcome_pair is not None:
                score += 1.1
            if connector_hit is not None and outcome_pair is not None:
                score += 0.6
            candidate = (
                sentence,
                connector_hit[0] if connector_hit else "",
                connector_hit[1] if connector_hit else "",
                outcome_pair,
                score,
            )
            if best is None or score > best[4]:
                best = candidate
    if best is None:
        return None
    return best[0], best[1], best[2], best[3]


def _difficulty_from_confidence(family: str, confidence: float) -> str:
    score = confidence
    if family == "causal_inversion":
        score -= 0.05
    if score >= 0.85:
        return "easy"
    if score >= 0.72:
        return "medium"
    if score >= 0.58:
        return "hard"
    return "very_hard"


# =============================================================================
# PERTURBATION FUNCTIONS
# =============================================================================

def detect_language_digits(text: str) -> bool:
    """Check if text uses Bengali digits."""
    return bool(re.search(r'[০-৯]', text))

def bn_to_en_number(bn_num: str) -> str:
    """Convert Bengali number to English."""
    return ''.join(BN_DIGITS.get(c, c) for c in bn_num)

def en_to_bn_number(en_num: str) -> str:
    """Convert English number to Bengali."""
    return ''.join(EN_DIGITS.get(c, c) for c in en_num)

def directional_flipping(text: str) -> Optional[str]:
    """
    Flip sentiment direction: replace positive words with negative and vice versa.
    """
    modified = text
    changes = 0
    
    # Find and replace positive words with negative
    for pos_word in SENTIMENT_DICT["positive"]:
        if pos_word in modified:
            neg_word = random.choice(SENTIMENT_DICT["negative"])
            modified = modified.replace(pos_word, neg_word, 1)
            changes += 1
            if changes >= 2:
                break
    
    # Find and replace negative words with positive
    if changes < 2:
        for neg_word in SENTIMENT_DICT["negative"]:
            if neg_word in modified:
                pos_word = random.choice(SENTIMENT_DICT["positive"])
                modified = modified.replace(neg_word, pos_word, 1)
                changes += 1
                if changes >= 2:
                    break
    
    return modified if changes >= 1 else None

def numerical_perturbation(text: str) -> Optional[str]:
    """
    Perturb numerical values: apply random factor or add/subtract offset.
    Handles both percentage and plain numbers (Bengali and English digits).
    """
    matches = list(NUMBER_PATTERN.finditer(text))
    if not matches:
        return None
    
    # Filter out very small numbers (single digits often are ordinals/counts)
    # and very large numbers (year-like). Keep "interesting" financial numbers.
    def _parse_num(match_obj):
        raw = match_obj.group().rstrip('%')
        bn = detect_language_digits(raw)
        en = bn_to_en_number(raw) if bn else raw
        try:
            return float(en), bn
        except ValueError:
            return None, None

    usable = []
    for m in matches:
        val, _ = _parse_num(m)
        if val is not None and 1 <= abs(val) <= 99999:
            usable.append(m)

    if not usable:
        return None

    # Work right-to-left so earlier indices stay valid
    selected = random.sample(usable, min(3, len(usable)))
    selected.sort(key=lambda m: m.start(), reverse=True)

    modified = text
    for match in selected:
        original = match.group()
        is_pct = '%' in original
        num_str = original.replace('%', '')
        is_bn = detect_language_digits(num_str)

        try:
            num = float(bn_to_en_number(num_str) if is_bn else num_str)
        except ValueError:
            continue

        # Strategy: perturb by factor (×0.5–0.8 or ×1.2–2.0)
        direction = random.choice([-1, 1])
        if direction == -1:
            factor = random.uniform(0.5, 0.8)
        else:
            factor = random.uniform(1.2, 2.0)
        new_num = num * factor

        # Format back
        if new_num == int(new_num):
            new_num_str = str(int(new_num))
        else:
            new_num_str = f"{new_num:.1f}"

        if is_bn:
            new_num_str = en_to_bn_number(new_num_str)

        new_match = new_num_str + ('%' if is_pct else '')
        modified = modified[:match.start()] + new_match + modified[match.end():]

    return modified if modified != text else None

def sentiment_amplification(text: str) -> Optional[str]:
    """
    Amplify existing sentiment with strong modifiers.
    """
    modified = text
    changes = 0
    
    # Try to amplify positive sentiment
    for pos_word in SENTIMENT_DICT["positive"][:5]:
        if pos_word in modified:
            amplifier = random.choice(AMPLIFIERS["strong_positive"])
            # Insert amplifier before the word
            modified = modified.replace(pos_word, f"{amplifier} {pos_word}", 1)
            changes += 1
            break
    
    # Try to amplify negative sentiment
    if changes == 0:
        for neg_word in SENTIMENT_DICT["negative"][:5]:
            if neg_word in modified:
                amplifier = random.choice(AMPLIFIERS["strong_negative"])
                modified = modified.replace(neg_word, f"{amplifier} {neg_word}", 1)
                changes += 1
                break
    
    return modified if changes >= 1 else None

def causal_distortion(text: str) -> Optional[str]:
    """
    Distort causal relationships by:
    1. Reversing cause-effect connectors
    2. Inserting false causality at sentence boundaries
    3. Negating financial claims
    """
    sentences = text.split('।')

    # Group into sentence pairs (cause → effect across sentences)
    # or find single sentences with embedded causality
    all_causal_words = CAUSAL_SIGNALS["cause"] + CAUSAL_SIGNALS["effect"]

    # Strategy 1: Find a sentence with a causal connector and rewrite it
    for i, sent in enumerate(sentences):
        sent_stripped = sent.strip()
        if len(sent_stripped) < 10:
            continue

        for word in all_causal_words:
            if word in sent_stripped:
                # Apply a transformation that always produces a different sentence
                new_sent = _apply_causal_distortion(sent_stripped, word)
                if new_sent and new_sent != sent_stripped:
                    sentences[i] = new_sent
                    return '।'.join(sentences)

    # Strategy 2: Pick a long sentence and insert false causality or negate a claim
    long_sents = [(i, s.strip()) for i, s in enumerate(sentences)
                  if len(s.strip()) > 30]
    if long_sents:
        idx, sent = random.choice(long_sents)
        new_sent = _apply_claim_negation(sent)
        if new_sent and new_sent != sent:
            sentences[idx] = new_sent
            return '।'.join(sentences)

    # Strategy 3: Swap two adjacent sentences (disrupts causal flow)
    valid_pairs = [(i, i+1) for i in range(len(sentences)-1)
                   if len(sentences[i].strip()) > 15 and len(sentences[i+1].strip()) > 15]
    if valid_pairs:
        i, j = random.choice(valid_pairs)
        sentences[i], sentences[j] = sentences[j], sentences[i]
        return '।'.join(sentences)

    return None


def _apply_causal_distortion(sentence: str, causal_word: str) -> Optional[str]:
    """Apply a distortion to a sentence containing a causal word."""
    replacements = {
        "ফলে": ["তবুও", "যদিও", "কিন্তু"],
        "কারণে": ["ফলে", "সত্ত্বেও"],
        "যোগে": ["ব্যতীত", "ছাড়াও"],
        "সুতরাং": ["তবুও", "যদিও"],
        "তাই": ["যদিও", "কিন্তু"],
        "কারণ": ["তবুও", "যদিও"],
        "যেহেতু": ["যদিও"],
        "হয়েছে": ["হয়নি", "নাও হতে পারে"],
        "বৃদ্ধি পেয়েছে": ["বৃদ্ধি পায়নি"],
        "কমেছে": ["বেড়েছে"],
        "প্রভাবিত হয়েছে": ["প্রভাবিত হয়নি"],
        "উন্নত হয়েছে": ["পিছিয়ে গেছে"],
        "পরিবর্তন হয়েছে": ["কোনো পরিবর্তন হয়নি"],
    }

    # Try direct word replacement
    if causal_word in replacements:
        for alt in replacements[causal_word]:
            new_sent = sentence.replace(causal_word, alt, 1)
            if new_sent != sentence:
                return new_sent

    # Generic: negate the verb-like claim around the causal word
    # Insert "নাও" before common verb endings to negate
    negation_patterns = [
        (r'ছে\b', 'ছে নাও'),
        (r'েছে\b', 'েছে নাও'),
        (r'েছিল\b', 'েছিল নাও'),
    ]
    for pat, repl in negation_patterns:
        new_sent = re.sub(pat, repl, sentence, count=1)
        if new_sent != sentence:
            return new_sent

    # Insert false causality at end
    false_causes = [
        " যদিও এটি আসলে ভিন্ন কারণে ঘটেছে",
        " কিন্তু প্রকৃত কারণ ভিন্ন",
        " তবে এর পেছনে অন্য কারণ রয়েছে",
    ]
    return sentence + random.choice(false_causes)


def _apply_claim_negation(sentence: str) -> Optional[str]:
    """Negate a financial claim in a sentence without relying on specific words."""
    # Bengali financial claim patterns
    claim_negations = [
        # "বৃদ্ধি পেয়েছে" → "বৃদ্ধি পায়নি"
        (r'বৃদ্ধি পেয়েছে', 'বৃদ্ধি পায়নি'),
        (r'বৃদ্ধি পাওয়া গেছে', 'বৃদ্ধি পাওয়া যায়নি'),
        # "বেড়েছে" → "বাড়েনি"
        (r'বেড়েছে', 'বাড়েনি'),
        (r'বৃদ্ধি নেওয়া গেছে', 'বৃদ্ধি নেওয়া যায়নি'),
        # "কমেছে" → "কমেনি"
        (r'কমেছে', 'কমেনি'),
        # "হ্রাস" → "বৃদ্ধি"
        (r'হ্রাস', 'বৃদ্ধি'),
        (r'পতন', 'উত্তোলন'),
        # "সফল" → "ব্যর্থ"
        (r'সফল', 'ব্যর্থ'),
        (r'উন্নয়ন', 'পতন'),
        # "প্রবৃদ্ধি" → "প্রত্যাগমন"
        (r'প্রবৃদ্ধি', 'প্রত্যাগমন'),
        # Generic verb negation: add "না" pattern
        (r'হয়েছে', 'হয়নি'),
        (r'দেখা গেছে', 'দেখা যায়নি'),
    ]

    random.shuffle(claim_negations)
    for pattern, repl in claim_negations:
        new_sent = re.sub(pattern, repl, sentence, count=1)
        if new_sent != sentence:
            return new_sent

    # If nothing matched, insert a causal disconnector at the end
    false_causes = [
        " যদিও এটি সম্পূর্ণ ভিন্ন কারণে ঘটেছে",
        " তবে প্রকৃত কারণ অন্যরকম",
        " কিন্তু বিশেষজ্ঞরা একমত নন",
    ]
    return sentence + random.choice(false_causes)

def entity_swap(text: str) -> Optional[str]:
    """
    Swap entity names (companies, banks, services) with alternatives.
    """
    modified = text
    changes = 0
    
    # Try company swaps
    for original, replacement in ENTITY_SWAP["companies"]:
        if original in modified:
            modified = modified.replace(original, replacement, 1)
            changes += 1
            break
    
    # Try bank swaps
    if changes == 0:
        for original, replacement in ENTITY_SWAP["banks"]:
            if original in modified:
                modified = modified.replace(original, replacement, 1)
                changes += 1
                break
    
    # Try remittance service swaps
    if changes == 0:
        for original, replacement in ENTITY_SWAP["remittance_services"]:
            if original in modified:
                modified = modified.replace(original, replacement, 1)
                changes += 1
                break
    
    return modified if changes >= 1 else None


def fact_aware_numerical_fact_change(text: str, rng: random.Random) -> Optional[PerturbationOutcome]:
    """Change one numerical fact while preserving the surrounding sentence."""
    candidate = _choose_best_numeric_candidate(text)
    if not candidate:
        return None

    sentence, match, score = candidate
    original_span = match.group()
    _, replacement = _shift_numeric_value(sentence.text, original_span, rng)
    if not replacement or replacement == original_span:
        return None

    new_sentence = _replace_span(sentence.text, match.start(), match.end(), replacement)
    new_text = _replace_span(text, sentence.start, sentence.end, new_sentence)
    confidence = min(0.97, 0.58 + (0.08 * score) + (0.04 if _contains_any(sentence.text, INCREASE_CUES | DECREASE_CUES) else 0.0))

    schema = {
        "family": "numerical_fact_change",
        "operator": "numeric_shift",
        "reasoning_type": PERTURBATION_REASONING["numerical_fact_change"],
        "sentence_index": sentence.index,
        "changed_span_original": original_span,
        "changed_span_replacement": replacement,
        "changed_span_role": "value",
        "ontology_class": "numeric_value",
        "confidence": round(confidence, 4),
    }

    return PerturbationOutcome(
        text=new_text,
        family="numerical_fact_change",
        operator="numeric_shift",
        reasoning_type=PERTURBATION_REASONING["numerical_fact_change"],
        difficulty=_difficulty_from_confidence("numerical_fact_change", confidence),
        changed_span_original=original_span,
        changed_span_replacement=replacement,
        changed_span_role="value",
        sentence_index=sentence.index,
        proposition_confidence=confidence,
        ontology_class="numeric_value",
        proposition_schema=schema,
    )


def fact_aware_policy_reversal(text: str, rng: random.Random) -> Optional[PerturbationOutcome]:
    """Reverse an explicit policy action such as raising/lowering rates or taxes."""
    candidate = _choose_policy_candidate(text)
    if not candidate:
        return None

    sentence, source, target = candidate
    span = _find_phrase_span(sentence.text, source)
    if not span:
        return None

    start, end, matched = span
    new_sentence = _replace_span(sentence.text, start, end, target)
    new_text = _replace_span(text, sentence.start, sentence.end, new_sentence)
    confidence = min(0.96, 0.82 + (0.03 * (len(source) > 6)))

    schema = {
        "family": "policy_reversal",
        "operator": "policy_phrase_flip",
        "reasoning_type": PERTURBATION_REASONING["policy_reversal"],
        "sentence_index": sentence.index,
        "changed_span_original": matched,
        "changed_span_replacement": target,
        "changed_span_role": "policy_action",
        "ontology_class": "policy_action",
        "confidence": round(confidence, 4),
    }

    return PerturbationOutcome(
        text=new_text,
        family="policy_reversal",
        operator="policy_phrase_flip",
        reasoning_type=PERTURBATION_REASONING["policy_reversal"],
        difficulty=_difficulty_from_confidence("policy_reversal", confidence),
        changed_span_original=matched,
        changed_span_replacement=target,
        changed_span_role="policy_action",
        sentence_index=sentence.index,
        proposition_confidence=confidence,
        ontology_class="policy_action",
        proposition_schema=schema,
    )


def fact_aware_entity_replacement(text: str, rng: random.Random) -> Optional[PerturbationOutcome]:
    """Swap an entity with another entity from the same ontology class."""
    mentions = _match_entity_mentions(text)
    if not mentions:
        return None

    best_mention: Optional[EntityMention] = None
    best_score = -1
    sentence_spans = _sentence_spans(text)

    for mention in mentions:
        window = text[max(0, mention.start - 36):min(len(text), mention.end + 36)]
        score = len(mention.base_text)
        sentence = next((s for s in sentence_spans if s.start <= mention.start < s.end), None)
        if sentence is not None:
            score += _sentence_salience_score(sentence)
        if mention.entity_class in SOURCE_ENTITY_CLASSES:
            score += 4
        if _contains_any(window, ATTRIBUTION_CUES):
            score += 3
        if _contains_any(window, FINANCIAL_CONTEXT_CUES):
            score += 1
        if score > best_score:
            best_score = score
            best_mention = mention

    if best_mention is None:
        return None

    replacement_base = _pick_replacement_from_class(best_mention.entity_class, best_mention.base_text, rng)
    if not replacement_base:
        return None

    matched_surface = best_mention.text
    matched_base, suffix = _split_entity_suffix(matched_surface, best_mention.base_text)
    replacement_surface = replacement_base + suffix
    if replacement_surface == matched_surface:
        return None

    new_text = _replace_span(text, best_mention.start, best_mention.end, replacement_surface)
    confidence = min(0.95, 0.76 + (0.02 * len(best_mention.base_text) / 10) + (0.03 if best_mention.entity_class in SOURCE_ENTITY_CLASSES else 0.0))
    operator = "source_attribution_swap" if best_mention.entity_class in SOURCE_ENTITY_CLASSES and _contains_any(
        text[max(0, best_mention.start - 40):min(len(text), best_mention.end + 40)],
        ATTRIBUTION_CUES,
    ) else f"{best_mention.entity_class}_swap"

    schema = {
        "family": "entity_replacement",
        "operator": operator,
        "reasoning_type": PERTURBATION_REASONING["entity_replacement"],
        "sentence_index": next((s.index for s in _sentence_spans(text) if s.start <= best_mention.start < s.end), 0),
        "changed_span_original": matched_surface,
        "changed_span_replacement": replacement_surface,
        "changed_span_role": "entity",
        "ontology_class": best_mention.entity_class,
        "confidence": round(confidence, 4),
    }

    return PerturbationOutcome(
        text=new_text,
        family="entity_replacement",
        operator=operator,
        reasoning_type=PERTURBATION_REASONING["entity_replacement"],
        difficulty=_difficulty_from_confidence("entity_replacement", confidence),
        changed_span_original=matched_surface,
        changed_span_replacement=replacement_surface,
        changed_span_role="entity",
        sentence_index=schema["sentence_index"],
        proposition_confidence=confidence,
        ontology_class=best_mention.entity_class,
        proposition_schema=schema,
    )


def fact_aware_temporal_shift(text: str, rng: random.Random) -> Optional[PerturbationOutcome]:
    """Shift a temporal reference by changing a year, month, or relative time phrase."""
    candidate = _choose_temporal_candidate(text)
    if not candidate:
        return None

    sentence, source, target, kind = candidate
    replacement = target
    operator = "temporal_flip"

    if kind == "year":
        digits = re.sub(r"[^0-9০-৯]", "", source)
        try:
            year = int(bn_to_en_number(digits))
        except ValueError:
            return None
        delta = rng.choice([-2, -1, 1, 2])
        shifted = max(1900, min(2099, year + delta))
        replacement = en_to_bn_number(str(shifted)) if re.search(r"[০-৯]", source) else str(shifted)
        operator = "year_shift"
    elif kind == "month":
        operator = "month_shift"
    else:
        operator = "relative_time_flip"

    span = _find_phrase_span(sentence.text, source)
    if not span:
        return None

    start, end, matched = span
    new_sentence = _replace_span(sentence.text, start, end, replacement)
    new_text = _replace_span(text, sentence.start, sentence.end, new_sentence)
    confidence = 0.90 if kind != "year" else 0.79
    if kind == "relative_time":
        confidence = 0.92

    schema = {
        "family": "temporal_shift",
        "operator": operator,
        "reasoning_type": PERTURBATION_REASONING["temporal_shift"],
        "sentence_index": sentence.index,
        "changed_span_original": matched,
        "changed_span_replacement": replacement,
        "changed_span_role": "time",
        "ontology_class": "temporal",
        "confidence": round(confidence, 4),
    }

    return PerturbationOutcome(
        text=new_text,
        family="temporal_shift",
        operator=operator,
        reasoning_type=PERTURBATION_REASONING["temporal_shift"],
        difficulty=_difficulty_from_confidence("temporal_shift", confidence),
        changed_span_original=matched,
        changed_span_replacement=replacement,
        changed_span_role="time",
        sentence_index=sentence.index,
        proposition_confidence=confidence,
        ontology_class="temporal",
        proposition_schema=schema,
    )


def fact_aware_causal_inversion(text: str, rng: random.Random) -> Optional[PerturbationOutcome]:
    """Invert a causal relation by changing the connector and outcome wording."""
    candidate = _choose_causal_candidate(text)
    if not candidate:
        return None

    sentence, source, target, outcome_pair = candidate
    new_sentence = sentence.text
    changed_original: List[str] = []
    changed_replacement: List[str] = []
    operator = "causal_outcome_flip"

    if outcome_pair:
        outcome_source, outcome_target = outcome_pair
        span = _find_phrase_span(new_sentence, outcome_source)
        if span:
            start, end, matched = span
            new_sentence = _replace_span(new_sentence, start, end, outcome_target)
            changed_original.append(matched)
            changed_replacement.append(outcome_target)

    if not changed_original and source:
        span = _find_phrase_span(new_sentence, source)
        if span:
            start, end, matched = span
            new_sentence = _replace_span(new_sentence, start, end, target)
            changed_original.append(matched)
            changed_replacement.append(target)
            operator = "causal_connector_flip"

    if not changed_original:
        return None

    new_text = _replace_span(text, sentence.start, sentence.end, new_sentence)
    confidence = 0.88 if len(changed_original) > 1 else 0.76

    schema = {
        "family": "causal_inversion",
        "operator": operator,
        "reasoning_type": PERTURBATION_REASONING["causal_inversion"],
        "sentence_index": sentence.index,
        "changed_span_original": " || ".join(changed_original),
        "changed_span_replacement": " || ".join(changed_replacement),
        "changed_span_role": "causal_relation",
        "ontology_class": "causal",
        "confidence": round(confidence, 4),
    }

    return PerturbationOutcome(
        text=new_text,
        family="causal_inversion",
        operator=operator,
        reasoning_type=PERTURBATION_REASONING["causal_inversion"],
        difficulty=_difficulty_from_confidence("causal_inversion", confidence),
        changed_span_original=" || ".join(changed_original),
        changed_span_replacement=" || ".join(changed_replacement),
        changed_span_role="causal_relation",
        sentence_index=sentence.index,
        proposition_confidence=confidence,
        ontology_class="causal",
        proposition_schema=schema,
    )

# =============================================================================
# DATA LOADING AND SAMPLING
# =============================================================================

def load_beni_v2(max_samples: int = None) -> List[Dict]:
    """Load and filter BENI v2 data from zstandard compressed CSV."""
    logger.info(f"Loading BENI v2 from {BENI_V2_PATH}")
    
    articles = []
    
    with open(BENI_V2_PATH, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(f) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            csv_reader = csv.DictReader(text_stream)
            
            for row in csv_reader:
                # Filter for financial/economy articles
                category = row.get('category_original', '').strip()
                sector = row.get('industry_sector', '').strip()
                language = row.get('language', '').strip()
                
                # Must be Bengali economy article
                if (category in FINANCIAL_CATEGORIES and 
                    language == 'bn' and
                    row.get('text_clean', '').strip()):
                    
                    # Additional sector filtering (optional, more inclusive)
                    articles.append({
                        'article_id': row['article_id'],
                        'newspaper': row['newspaper'],
                        'publication_date': row['publication_date'],
                        'headline': row.get('headline', ''),
                        'text': row.get('text_clean', row.get('text', '')),
                        'industry_sector': sector
                    })
                    
                    if max_samples and len(articles) >= max_samples:
                        break
    
    logger.info(f"Loaded {len(articles)} financial articles from BENI v2")
    return articles

def sample_articles(articles: List[Dict], n: int, seed: int = 42) -> List[Dict]:
    """Randomly sample n articles."""
    rng = random.Random(seed)
    if len(articles) <= n:
        return list(articles)
    return rng.sample(articles, n)

# =============================================================================
# MULTI-HOP PLANNING AND VALIDATION
# =============================================================================

def extract_financial_propositions(text: str) -> Dict[str, List[Dict[str, object]]]:
    """Extract proposition candidates that can be perturbed safely."""
    proposition_graph: Dict[str, List[Dict[str, object]]] = {
        family: []
        for family in FACT_AWARE_PERTURBATION_TYPES
    }
    seen = set()
    sentence_pattern = re.compile(r"[^\u0964।!?]+(?:[।!?]+|$)")
    numeric_pattern = re.compile(r"[০-৯0-9]+(?:[.,][০-৯0-9]+)?%?")

    for sentence_index, sentence_match in enumerate(sentence_pattern.finditer(text)):
        sentence_text = sentence_match.group().strip()
        if not sentence_text:
            continue

        # Numerical propositions
        for match in numeric_pattern.finditer(sentence_text):
            span_text = match.group().strip()
            if not span_text or _is_year_like(span_text):
                continue
            sentence_span = SentenceSpan(
                index=sentence_index,
                start=sentence_match.start(),
                end=sentence_match.end(),
                text=sentence_text,
            )
            score = _score_numeric_candidate(sentence_span, match.start(), match.end(), span_text)
            if score <= 0:
                continue
            key = ("numerical_fact_change", sentence_index, match.start(), match.end(), span_text)
            if key in seen:
                continue
            seen.add(key)
            proposition_graph["numerical_fact_change"].append(
                {
                    "family": "numerical_fact_change",
                    "sentence_index": sentence_index,
                    "span_start": match.start(),
                    "span_end": match.end(),
                    "span_text": span_text,
                    "replacement_text": "",
                    "role": "value",
                    "ontology_class": "numeric_value",
                    "score": float(score),
                }
            )

        # Policy propositions
        for source, target in sorted(POLICY_REWRITE_PAIRS, key=lambda item: len(item[0]), reverse=True):
            span = _find_phrase_span(sentence_text, source)
            if not span:
                continue
            start, end, matched = span
            key = ("policy_reversal", sentence_index, start, end, matched)
            if key in seen:
                continue
            seen.add(key)
            proposition_graph["policy_reversal"].append(
                {
                    "family": "policy_reversal",
                    "sentence_index": sentence_index,
                    "span_start": start,
                    "span_end": end,
                    "span_text": matched,
                    "replacement_text": target,
                    "role": "policy_action",
                    "ontology_class": "policy_action",
                    "score": 0.90,
                }
            )

        # Entity propositions
        for mention in _match_entity_mentions(sentence_text):
            key = ("entity_replacement", sentence_index, mention.start, mention.end, mention.text)
            if key in seen:
                continue
            seen.add(key)
            score = 0.70 + min(0.20, len(mention.base_text) / 100.0)
            proposition_graph["entity_replacement"].append(
                {
                    "family": "entity_replacement",
                    "sentence_index": sentence_index,
                    "span_start": mention.start,
                    "span_end": mention.end,
                    "span_text": mention.text,
                    "replacement_text": "",
                    "role": "entity",
                    "ontology_class": mention.entity_class,
                    "score": round(score, 4),
                }
            )

        # Temporal propositions
        for source, target in RELATIVE_TIME_REWRITE:
            if source in sentence_text:
                span = _find_phrase_span(sentence_text, source)
                if not span:
                    continue
                start, end, matched = span
                key = ("temporal_shift", sentence_index, start, end, matched)
                if key in seen:
                    continue
                seen.add(key)
                proposition_graph["temporal_shift"].append(
                    {
                        "family": "temporal_shift",
                        "sentence_index": sentence_index,
                        "span_start": start,
                        "span_end": end,
                        "span_text": matched,
                        "replacement_text": target,
                        "role": "time",
                        "ontology_class": "temporal",
                        "score": 0.82,
                    }
                )

        year_match = re.search(r"(?<!\d)(?:19\d{2}|20\d{2}|[০-৯]{4})(?!\d)", sentence_text)
        if year_match:
            span_text = year_match.group()
            key = ("temporal_shift", sentence_index, year_match.start(), year_match.end(), span_text)
            if key not in seen:
                seen.add(key)
                proposition_graph["temporal_shift"].append(
                    {
                        "family": "temporal_shift",
                        "sentence_index": sentence_index,
                        "span_start": year_match.start(),
                        "span_end": year_match.end(),
                        "span_text": span_text,
                        "replacement_text": "",
                        "role": "time",
                        "ontology_class": "temporal",
                        "score": 0.75,
                    }
                )

        for month in MONTH_NAMES:
            if month in sentence_text:
                span = _find_phrase_span(sentence_text, month)
                if not span:
                    continue
                start, end, matched = span
                key = ("temporal_shift", sentence_index, start, end, matched)
                if key in seen:
                    continue
                seen.add(key)
                proposition_graph["temporal_shift"].append(
                    {
                        "family": "temporal_shift",
                        "sentence_index": sentence_index,
                        "span_start": start,
                        "span_end": end,
                        "span_text": matched,
                        "replacement_text": MONTH_REWRITE[month],
                        "role": "time",
                        "ontology_class": "temporal",
                        "score": 0.80,
                    }
                )

        # Causal propositions
        for source, target in sorted(CAUSAL_REWRITE_PAIRS, key=lambda item: len(item[0]), reverse=True):
            if source in sentence_text:
                span = _find_phrase_span(sentence_text, source)
                if not span:
                    continue
                start, end, matched = span
                key = ("causal_inversion", sentence_index, start, end, matched, target)
                if key in seen:
                    continue
                seen.add(key)
                proposition_graph["causal_inversion"].append(
                    {
                        "family": "causal_inversion",
                        "sentence_index": sentence_index,
                        "span_start": start,
                        "span_end": end,
                        "span_text": matched,
                        "replacement_text": target,
                        "role": "causal_connector",
                        "ontology_class": "causal",
                        "score": 0.86,
                    }
                )

        for source, target in OUTCOME_REWRITE_PAIRS:
            if source in sentence_text:
                span = _find_phrase_span(sentence_text, source)
                if not span:
                    continue
                start, end, matched = span
                key = ("causal_inversion", sentence_index, start, end, matched, target)
                if key in seen:
                    continue
                seen.add(key)
                proposition_graph["causal_inversion"].append(
                    {
                        "family": "causal_inversion",
                        "sentence_index": sentence_index,
                        "span_start": start,
                        "span_end": end,
                        "span_text": matched,
                        "replacement_text": target,
                        "role": "causal_outcome",
                        "ontology_class": "causal",
                        "score": 0.88,
                    }
                )

    for family, items in proposition_graph.items():
        items.sort(key=lambda item: (-item["score"], item["sentence_index"], item["span_start"], item["span_end"]))

    return proposition_graph


def _select_plan_difficulty(rng: random.Random, target_difficulty: Optional[str] = None) -> str:
    if target_difficulty in DIFFICULTY_TO_HOPS:
        return target_difficulty
    labels = list(DIFFICULTY_WEIGHTS.keys())
    weights = [DIFFICULTY_WEIGHTS[label] for label in labels]
    return rng.choices(labels, weights=weights, k=1)[0]


def _select_difficulty_for_family(family: str, rng: random.Random) -> str:
    """Select a target difficulty using the family-aware sampling policy."""
    family_weights = FAMILY_DIFFICULTY_WEIGHTS.get(family)
    if family_weights:
        labels = list(family_weights.keys())
        weights = [family_weights[label] for label in labels]
        return rng.choices(labels, weights=weights, k=1)[0]
    return _select_plan_difficulty(rng)


def _select_balanced_difficulty_for_family(
    family: str,
    rng: random.Random,
    difficulty_counts: Counter,
    difficulty_targets: Dict[str, float],
) -> str:
    """Select a difficulty while nudging the sampler toward the target mix."""
    family_weights = FAMILY_DIFFICULTY_WEIGHTS.get(family)
    if not family_weights:
        return _select_plan_difficulty(rng)

    labels = list(family_weights.keys())
    weights: List[float] = []
    for label in labels:
        target = float(difficulty_targets.get(label, 0.0))
        if target <= 0:
            balance_factor = 1.0
        else:
            remaining_ratio = max(target - float(difficulty_counts.get(label, 0)), 0.0) / target
            balance_factor = 0.5 + (1.5 * remaining_ratio)
        weights.append(family_weights[label] * balance_factor)

    if not any(weights):
        return _select_plan_difficulty(rng)

    return rng.choices(labels, weights=weights, k=1)[0]


def _choose_family_sequence(
    primary_family: str,
    proposition_graph: Dict[str, List[Dict[str, object]]],
    hop_count: int,
    rng: random.Random,
) -> Optional[List[str]]:
    candidate_counts = {family: len(items) for family, items in proposition_graph.items() if items}
    if candidate_counts.get(primary_family, 0) == 0:
        return None

    available = [family for family in FACT_AWARE_PERTURBATION_TYPES if candidate_counts.get(family, 0) > 0]
    if primary_family not in available:
        available.insert(0, primary_family)

    family_sequence = [primary_family]

    while len(family_sequence) < hop_count:
        pool = [family for family in available if family_sequence.count(family) < max(1, candidate_counts.get(family, 1))]
        if not pool:
            pool = available[:]

        if not pool:
            return None

        pool.sort(
            key=lambda family: (
                family_sequence.count(family),
                -candidate_counts.get(family, 0),
                FAMILY_PRIORITY.get(family, 99),
            )
        )
        chosen = pool[0]
        if chosen != family_sequence[-1] or len(pool) == 1:
            family_sequence.append(chosen)
        else:
            family_sequence.append(pool[0])

    return family_sequence


def build_perturbation_plan(
    text: str,
    primary_family: str,
    rng: random.Random,
    target_difficulty: Optional[str] = None,
) -> Optional[PerturbationPlan]:
    """Build a multi-hop perturbation plan from extracted propositions."""
    proposition_graph = extract_financial_propositions(text)
    if primary_family not in proposition_graph or not proposition_graph[primary_family]:
        return None

    difficulty = _select_plan_difficulty(rng, target_difficulty)
    hop_count = DIFFICULTY_TO_HOPS[difficulty]
    family_sequence = _choose_family_sequence(primary_family, proposition_graph, hop_count, rng)
    if not family_sequence:
        return None

    candidate_summary = {
        family: len(items)
        for family, items in proposition_graph.items()
        if items
    }
    planned_operations = [
        {
            "step": idx + 1,
            "family": family,
            "candidate_count": candidate_summary.get(family, 0),
        }
        for idx, family in enumerate(family_sequence)
    ]

    return PerturbationPlan(
        primary_family=primary_family,
        difficulty=difficulty,
        hop_count=hop_count,
        family_sequence=family_sequence,
        proposition_graph=proposition_graph,
        candidate_summary=candidate_summary,
        planned_operations=planned_operations,
    )


def _apply_operation_sequence(
    text: str,
    plan: PerturbationPlan,
    rng: random.Random,
) -> Optional[Tuple[str, List[PerturbationOutcome], List[str]]]:
    perturbation_funcs = {
        "numerical_fact_change": fact_aware_numerical_fact_change,
        "policy_reversal": fact_aware_policy_reversal,
        "entity_replacement": fact_aware_entity_replacement,
        "temporal_shift": fact_aware_temporal_shift,
        "causal_inversion": fact_aware_causal_inversion,
    }

    current_text = text
    outcomes: List[PerturbationOutcome] = []
    applied_families: List[str] = []

    search_order = list(plan.family_sequence)
    for family in FACT_AWARE_PERTURBATION_TYPES:
        if family not in search_order:
            search_order.append(family)

    for family in search_order:
        if len(outcomes) >= plan.hop_count:
            break
        func = perturbation_funcs.get(family)
        if func is None:
            continue
        outcome = func(current_text, rng)
        if outcome is None or outcome.text == current_text:
            continue
        outcomes.append(outcome)
        applied_families.append(family)
        current_text = outcome.text

    if len(outcomes) < plan.hop_count:
        return None

    return current_text, outcomes, applied_families


def _normalize_for_similarity(text: str) -> str:
    text = re.sub(r"[।!?;:,\-—()\"'“”‘’]", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _score_semantic_similarity(original: str, perturbed: str) -> float:
    norm_original = _normalize_for_similarity(original)
    norm_perturbed = _normalize_for_similarity(perturbed)
    if not norm_original or not norm_perturbed:
        return 0.0

    char_ratio = SequenceMatcher(None, norm_original, norm_perturbed).ratio()
    original_tokens = set(norm_original.split())
    perturbed_tokens = set(norm_perturbed.split())
    if original_tokens and perturbed_tokens:
        jaccard = len(original_tokens & perturbed_tokens) / len(original_tokens | perturbed_tokens)
    else:
        jaccard = char_ratio

    return round(min(1.0, 0.65 * char_ratio + 0.35 * jaccard), 4)


def _score_fluency(text: str) -> float:
    score = 1.0
    word_count = len(text.split())
    if word_count < 5:
        score -= 0.45
    if len(text) < 40:
        score -= 0.25

    special_chars = len(re.findall(r"[^\w\u0980-\u09FF\s।]", text))
    if text:
        score -= min(0.35, (special_chars / len(text)) * 1.8)

    if re.search(r"(.)\1{3,}", text):
        score -= 0.25
    if re.search(r"\s{3,}", text):
        score -= 0.10
    if text.count("(") != text.count(")"):
        score -= 0.05

    return round(max(0.0, min(1.0, score)), 4)


@lru_cache(maxsize=1)
def _load_nli_verifier():
    if not ENABLE_NLI_VERIFIER:
        return None

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional dependency path
        logger.warning(f"Skipping NLI verifier: {exc}")
        return None

    model_cfgs = VALIDATION_CONFIG.get("models", [])
    model_id = VALIDATION_CONFIG.get("nli_model_id")
    if not model_id and model_cfgs:
        model_id = model_cfgs[0].get("model_id")
    if not model_id:
        return None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(device)
    model.eval()
    return tokenizer, model, device


def _nli_contradiction_probability(original: str, perturbed: str) -> Optional[float]:
    verifier = _load_nli_verifier()
    if verifier is None:
        return None

    tokenizer, model, device = verifier
    try:
        import torch
    except Exception:  # pragma: no cover - optional dependency path
        return None

    inputs = tokenizer(
        original,
        perturbed,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    label2id = getattr(getattr(model, "config", None), "label2id", {}) or {}
    contra_idx = None
    for label_name, idx in label2id.items():
        if str(label_name).lower() == "contradiction":
            contra_idx = idx
            break
    if contra_idx is None:
        contra_idx = 0

    return float(probs[contra_idx].item())


def _score_contradiction(original: str, perturbed: str, outcomes: List[PerturbationOutcome]) -> float:
    if not outcomes:
        return 0.0

    family_weights = [FAMILY_CONTRADICTION_WEIGHTS.get(outcome.family, 0.20) for outcome in outcomes]
    confidence_weights = [outcome.proposition_confidence for outcome in outcomes]
    average_family_weight = sum(family_weights) / len(family_weights)
    average_confidence = sum(confidence_weights) / len(confidence_weights)
    score = 0.16 + (0.48 * average_family_weight) + (0.22 * average_confidence)
    score += 0.06 * max(0, len(outcomes) - 1)
    if any(outcome.family in {"policy_reversal", "numerical_fact_change", "causal_inversion"} for outcome in outcomes):
        score += 0.05
    if any(outcome.family == "temporal_shift" for outcome in outcomes):
        score += 0.02

    nli_score = _nli_contradiction_probability(original, perturbed)
    if nli_score is not None:
        score = 0.55 * score + 0.45 * nli_score

    return round(max(0.0, min(0.99, score)), 4)


def validate_generated_sample(
    original_text: str,
    perturbed_text: str,
    outcomes: List[PerturbationOutcome],
    plan: PerturbationPlan,
) -> ValidationResult:
    """Validate a generated perturbation with multiple quality signals."""
    issues: List[str] = []

    if not perturbed_text or perturbed_text == original_text:
        issues.append("identical_text")

    if len(outcomes) != plan.hop_count:
        issues.append("hop_count_mismatch")

    semantic_similarity = _score_semantic_similarity(original_text, perturbed_text)
    if semantic_similarity < SEMANTIC_SIMILARITY_MIN:
        issues.append("semantic_similarity_too_low")
    if semantic_similarity > SEMANTIC_SIMILARITY_MAX:
        issues.append("semantic_similarity_too_high")

    fluency_score = _score_fluency(perturbed_text)
    if fluency_score < FLUENCY_MIN_SCORE:
        issues.append("low_fluency")

    contradiction_score = _score_contradiction(original_text, perturbed_text, outcomes)
    threshold = CONTRADICTION_MIN_SCORE_BY_DIFFICULTY.get(
        plan.difficulty,
        CONTRADICTION_MIN_SCORE,
    )
    if contradiction_score < threshold:
        issues.append("weak_contradiction")

    passed = not issues
    return ValidationResult(
        passed=passed,
        contradiction_score=contradiction_score,
        semantic_similarity=semantic_similarity,
        fluency_score=fluency_score,
        issues=issues,
    )

# =============================================================================
# DATASET GENERATION
# =============================================================================

PERTURBATION_SUFFIX = {
    "numerical_fact_change": "num",
    "policy_reversal": "pol",
    "entity_replacement": "ent",
    "temporal_shift": "tmp",
    "causal_inversion": "cau",
}


def _select_difficulty_for_generation(rng: random.Random) -> str:
    labels = list(DIFFICULTY_WEIGHTS.keys())
    weights = [DIFFICULTY_WEIGHTS[label] for label in labels]
    return rng.choices(labels, weights=weights, k=1)[0]


def _build_perturbed_row(
    article: Dict,
    plan: PerturbationPlan,
    outcomes: List[PerturbationOutcome],
    validation: ValidationResult,
) -> Dict:
    primary = outcomes[0]
    applied_families = [outcome.family for outcome in outcomes]
    schema_json = json.dumps(primary.proposition_schema, ensure_ascii=False)
    proposition_graph_json = json.dumps(plan.proposition_graph, ensure_ascii=False)
    validation_json = json.dumps(asdict(validation), ensure_ascii=False)
    perturbation_plan_json = json.dumps(
        {
            "primary_family": plan.primary_family,
            "difficulty": plan.difficulty,
            "hop_count": plan.hop_count,
            "planned_families": plan.family_sequence,
            "applied_families": applied_families,
            "candidate_summary": plan.candidate_summary,
            "planned_operations": plan.planned_operations,
            "proposition_graph": plan.proposition_graph,
            "operations": [asdict(outcome) for outcome in outcomes],
            "validation": asdict(validation),
        },
        ensure_ascii=False,
    )

    return {
        "article_id": f"{article['article_id']}_pert_{PERTURBATION_SUFFIX[plan.primary_family]}_{len(outcomes)}",
        "original_id": article["article_id"],
        "newspaper": article["newspaper"],
        "publication_date": article["publication_date"],
        "headline": article["headline"],
        "text": outcomes[-1].text,
        "original_text": article["text"],
        "perturbation_type": plan.primary_family,
        "perturbation_mode": "multi_hop" if len(outcomes) > 1 else "single_hop",
        "perturbation_operator": "+".join(outcome.operator for outcome in outcomes),
        "fact_type": plan.primary_family,
        "primary_family": plan.primary_family,
        "planned_families": json.dumps(plan.family_sequence, ensure_ascii=False),
        "perturbation_families": json.dumps(applied_families, ensure_ascii=False),
        "hop_count": len(outcomes),
        "planned_hops": plan.hop_count,
        "reasoning_type": primary.reasoning_type,
        "operation_reasoning_types": json.dumps([outcome.reasoning_type for outcome in outcomes], ensure_ascii=False),
        "difficulty": plan.difficulty,
        "primary_operation_difficulty": primary.difficulty,
        "changed_span_original": primary.changed_span_original,
        "changed_span_replacement": primary.changed_span_replacement,
        "changed_span_role": primary.changed_span_role,
        "sentence_index": primary.sentence_index,
        "proposition_confidence": f"{primary.proposition_confidence:.4f}",
        "ontology_class": primary.ontology_class,
        "proposition_schema": schema_json,
        "proposition_graph": proposition_graph_json,
        "perturbation_plan": perturbation_plan_json,
        "validation_scores": validation_json,
        "validation_passed": validation.passed,
        "validation_issues": json.dumps(validation.issues, ensure_ascii=False),
        "industry_sector": article["industry_sector"],
        "label": 1,
    }


def apply_perturbation(
    article: Dict,
    perturbation_type: str,
    rng: random.Random,
    target_difficulty: Optional[str] = None,
) -> Optional[Dict]:
    """Apply a fact-aware perturbation plan to an article."""
    text = article["text"]

    for _ in range(MAX_GENERATION_ATTEMPTS):
        plan = build_perturbation_plan(text, perturbation_type, rng, target_difficulty)
        if plan is None:
            return None

        execution = _apply_operation_sequence(text, plan, rng)
        if execution is None:
            continue

        perturbed_text, outcomes, _ = execution
        if not outcomes or perturbed_text == text:
            continue

        validation = validate_generated_sample(text, perturbed_text, outcomes, plan)
        if not validation.passed:
            continue

        return _build_perturbed_row(article, plan, outcomes, validation)

    return None

def generate_dataset(articles: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Generate original and perturbed datasets using multi-hop fact-aware perturbations."""
    rng = random.Random(RANDOM_SEED)
    originals = []
    perturbed = []
    difficulty_counts = Counter()
    total_target = TARGET_PER_TYPE * len(FACT_AWARE_PERTURBATION_TYPES)
    difficulty_targets = {
        label: DIFFICULTY_WEIGHTS[label] * total_target
        for label in DIFFICULTY_WEIGHTS
    }

    # Create original samples
    for article in articles:
        originals.append({
            'article_id': article['article_id'],
            'original_id': article['article_id'],
            'newspaper': article['newspaper'],
            'publication_date': article['publication_date'],
            'headline': article['headline'],
            'text': article['text'],
            'original_text': article['text'],
            'perturbation_type': 'none',
            'perturbation_mode': 'original',
            'perturbation_operator': '',
            'fact_type': 'original',
            'primary_family': '',
            'planned_families': '',
            'perturbation_families': '',
            'operation_reasoning_types': '',
            'hop_count': '',
            'planned_hops': '',
            'reasoning_type': '',
            'difficulty': '',
            'primary_operation_difficulty': '',
            'changed_span_original': '',
            'changed_span_replacement': '',
            'changed_span_role': '',
            'sentence_index': '',
            'proposition_confidence': '',
            'ontology_class': '',
            'proposition_schema': '',
            'proposition_graph': '',
            'perturbation_plan': '',
            'validation_scores': '',
            'validation_passed': '',
            'validation_issues': '',
            'industry_sector': article['industry_sector'],
            'label': 0
        })

    shuffled_articles = list(articles)
    used_by_type = {ptype: set() for ptype in FACT_AWARE_PERTURBATION_TYPES}
    max_passes = 8

    for ptype in FACT_AWARE_PERTURBATION_TYPES:
        count = 0
        for pass_num in range(max_passes):
            if count >= TARGET_PER_TYPE:
                break
            rng.shuffle(shuffled_articles)
            for article in shuffled_articles:
                if count >= TARGET_PER_TYPE:
                    break
                if article['article_id'] in used_by_type[ptype]:
                    continue
                target_difficulty = _select_balanced_difficulty_for_family(
                    ptype,
                    rng,
                    difficulty_counts,
                    difficulty_targets,
                )
                result = apply_perturbation(article, ptype, rng, target_difficulty=target_difficulty)
                if result:
                    perturbed.append(result)
                    difficulty_counts[result["difficulty"]] += 1
                    used_by_type[ptype].add(article['article_id'])
                    count += 1
        if count < TARGET_PER_TYPE:
            logger.warning(
                f"Only generated {count}/{TARGET_PER_TYPE} samples for {ptype}"
            )

    logger.info(f"Generated {len(originals)} original + {len(perturbed)} perturbed samples")
    return originals, perturbed

# =============================================================================
# QUALITY FILTERING
# =============================================================================

def basic_quality_filter(article: Dict) -> bool:
    """Basic quality checks for perturbed articles."""
    text = article['text']
    original = article['original_text']
    
    # Check minimum length
    if len(text) < 50:
        return False
    
    # Check maximum length difference (perturbation shouldn't change length too much)
    length_ratio = len(text) / len(original) if len(original) > 0 else 0
    if length_ratio < 0.65 or length_ratio > 1.35:
        return False

    # Check if text actually changed
    if text == original:
        return False

    validation_scores_raw = article.get("validation_scores", "")
    if validation_scores_raw:
        try:
            validation_scores = json.loads(validation_scores_raw)
        except json.JSONDecodeError:
            validation_scores = {}
        if validation_scores:
            if not validation_scores.get("passed", False):
                return False
            difficulty = article.get("difficulty", "")
            threshold = CONTRADICTION_MIN_SCORE_BY_DIFFICULTY.get(difficulty, CONTRADICTION_MIN_SCORE)
            if validation_scores.get("semantic_similarity", 0.0) < SEMANTIC_SIMILARITY_MIN:
                return False
            if validation_scores.get("fluency_score", 0.0) < FLUENCY_MIN_SCORE:
                return False
            if validation_scores.get("contradiction_score", 0.0) < threshold:
                return False
            return True

    # Ensure fact-aware metadata is present and non-trivial
    if article.get("perturbation_type") in FACT_AWARE_PERTURBATION_TYPES:
        if not article.get("changed_span_original") or not article.get("changed_span_replacement"):
            return False
        if article.get("changed_span_original") == article.get("changed_span_replacement"):
            return False
        if not article.get("proposition_schema"):
            return False
        if not article.get("perturbation_plan"):
            return False

    return True


def assign_splits_by_original_id(
    rows: List[Dict],
    seed: int = RANDOM_SEED,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT,
) -> List[Dict]:
    """Assign train/validation/test splits at the source-article level."""
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio:.4f}")

    grouped_rows: Dict[str, List[Dict]] = {}
    for row in rows:
        original_id = row.get("original_id") or row.get("article_id")
        if not original_id:
            raise ValueError("Each row must contain original_id or article_id for split assignment")
        grouped_rows.setdefault(str(original_id), []).append(row)

    group_items = [(original_id, len(group_rows)) for original_id, group_rows in grouped_rows.items()]
    rng = random.Random(seed)
    rng.shuffle(group_items)
    group_items.sort(key=lambda item: item[1], reverse=True)

    targets = {
        "train": len(rows) * train_ratio,
        "validation": len(rows) * val_ratio,
        "test": len(rows) * test_ratio,
    }
    assigned_counts = {split: 0.0 for split in targets}
    group_to_split: Dict[str, str] = {}

    for original_id, group_size in group_items:
        split = min(
            targets.keys(),
            key=lambda split_name: (
                assigned_counts[split_name] / max(targets[split_name], 1.0),
                assigned_counts[split_name],
            ),
        )
        group_to_split[original_id] = split
        assigned_counts[split] += group_size

    assigned_rows: List[Dict] = []
    for row in rows:
        row_copy = dict(row)
        original_id = str(row_copy.get("original_id") or row_copy.get("article_id"))
        row_copy["split"] = group_to_split[original_id]
        assigned_rows.append(row_copy)

    return assigned_rows


def audit_original_id_split_leakage(rows: List[Dict]) -> Dict[str, object]:
    """Check whether any original_id appears in more than one split."""
    original_to_splits: Dict[str, set] = {}
    missing_split_rows = 0
    for row in rows:
        original_id = str(row.get("original_id") or row.get("article_id") or "")
        split = row.get("split", "")
        if not split:
            missing_split_rows += 1
            split = "unknown"
        if not original_id:
            continue
        original_to_splits.setdefault(original_id, set()).add(split)

    leaky_original_ids = {
        original_id: sorted(split_set)
        for original_id, split_set in original_to_splits.items()
        if len(split_set) > 1
    }
    split_row_counts = Counter(row.get("split", "unknown") for row in rows)
    split_group_counts = Counter(
        next(iter(split_set))
        for split_set in original_to_splits.values()
        if len(split_set) == 1
    )

    return {
        "has_leakage": bool(leaky_original_ids) or missing_split_rows > 0,
        "leaky_original_ids": len(leaky_original_ids),
        "leaky_original_id_examples": list(leaky_original_ids.items())[:10],
        "split_row_counts": dict(split_row_counts),
        "split_group_counts": dict(split_group_counts),
        "unique_original_ids": len(original_to_splits),
        "missing_split_rows": missing_split_rows,
    }

# =============================================================================
# SAVE DATASET
# =============================================================================

def save_dataset(originals: List[Dict], perturbed: List[Dict], output_dir: Path):
    """Save datasets to CSV files."""
    all_fields = []
    seen_fields = set()
    combined_rows = originals + perturbed
    for row in combined_rows:
        for field in row.keys():
            if field not in seen_fields:
                seen_fields.add(field)
                all_fields.append(field)

    def _write_csv(path: Path, rows: List[Dict]):
        if not rows:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_fields)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, '') for field in all_fields})

    # Save originals
    originals_path = output_dir / "finfact_bd_originals.csv"
    _write_csv(originals_path, originals)
    logger.info(f"Saved {len(originals)} originals to {originals_path}")
    
    # Save perturbed
    perturbed_path = output_dir / "finfact_bd_perturbed.csv"
    _write_csv(perturbed_path, perturbed)
    logger.info(f"Saved {len(perturbed)} perturbed to {perturbed_path}")
    
    # Save combined dataset
    combined_path = output_dir / "finfact_bd_combined.csv"
    _write_csv(combined_path, combined_rows)
    logger.info(f"Saved {len(combined_rows)} combined to {combined_path}")

    split_rows_present = any(row.get("split") for row in combined_rows)
    split_audit = None
    if split_rows_present:
        split_audit = audit_original_id_split_leakage(combined_rows)
        if split_audit["has_leakage"]:
            raise RuntimeError(
                "Split leakage detected across original_id groups: "
                f"{split_audit['leaky_original_ids']} overlapping groups"
            )

        for split_name in ("train", "validation", "test"):
            split_rows = [row for row in combined_rows if row.get("split") == split_name]
            split_path = output_dir / f"finfact_bd_{split_name}.csv"
            _write_csv(split_path, split_rows)
            logger.info(f"Saved {len(split_rows)} {split_name} rows to {split_path}")
    
    # Save metadata
    metadata = {
        'total_samples': len(originals) + len(perturbed),
        'original_count': len(originals),
        'perturbed_count': len(perturbed),
        'dataset_version': DATASET_VERSION,
        'release_tag': DATASET_RELEASE_TAG,
        'release_date': DATASET_RELEASE_DATE,
        'release_state': DATASET_RELEASE_STATE,
        'perturbation_types': FACT_AWARE_PERTURBATION_TYPES,
        'perturbation_distribution': {
            ptype: sum(1 for row in perturbed if row['perturbation_type'] == ptype)
            for ptype in FACT_AWARE_PERTURBATION_TYPES
        },
        'difficulty_distribution': dict(Counter(row.get('difficulty', 'unknown') for row in perturbed)),
        'hop_count_distribution': dict(Counter(str(row.get('hop_count', 'unknown')) for row in perturbed)),
        'difficulty_policy': {
            'global_weights': DIFFICULTY_WEIGHTS,
            'family_weights': FAMILY_DIFFICULTY_WEIGHTS,
            'balance_strategy': 'global_budget_adjusted_family_sampling',
        },
        'source': 'BENI v2',
        'language': 'bn',
        'domain': 'financial',
        'label_schema': {'0': 'original/real', '1': 'perturbed/fake'},
        'schema_version': '2.0',
        'perturbation_design': 'fact_aware',
    }
    if split_audit is not None:
        metadata['split_distribution'] = {
            split: split_audit['split_row_counts'].get(split, 0)
            for split in ("train", "validation", "test")
        }
        metadata['split_group_distribution'] = {
            split: split_audit['split_group_counts'].get(split, 0)
            for split in ("train", "validation", "test")
        }
        metadata['split_leakage'] = {
            'has_leakage': split_audit['has_leakage'],
            'leaky_original_ids': split_audit['leaky_original_ids'],
            'leaky_examples': split_audit['leaky_original_id_examples'],
            'unique_original_ids': split_audit['unique_original_ids'],
            'missing_split_rows': split_audit['missing_split_rows'],
        }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved metadata to {metadata_path}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main pipeline."""
    logger.info("Starting FinFact-BD dataset generation")
    
    # Step 1: Load and sample articles
    articles = load_beni_v2(max_samples=50000)  # Load more for better sampling
    sampled = sample_articles(articles, NUM_SAMPLES, RANDOM_SEED)
    logger.info(f"Sampled {len(sampled)} articles")
    
    # Step 2: Generate perturbations
    originals, perturbed = generate_dataset(sampled)
    
    # Step 3: Quality filter perturbed samples
    perturbed_filtered = [p for p in perturbed if basic_quality_filter(p)]
    logger.info(f"After quality filtering: {len(perturbed_filtered)} perturbed samples")
    
    # Step 4: Balance dataset
    # Take equal number of originals and perturbed
    n = min(len(originals), len(perturbed_filtered))
    originals_balanced = originals[:n]
    perturbed_balanced = perturbed_filtered[:n]
    logger.info(f"Balancing dataset to {n} originals and {n} perturbed samples")

    # Step 4b: Assign splits by original_id and audit for leakage
    combined_balanced = originals_balanced + perturbed_balanced
    combined_split = assign_splits_by_original_id(
        combined_balanced,
        seed=RANDOM_SEED,
        train_ratio=TRAIN_SPLIT,
        val_ratio=VAL_SPLIT,
        test_ratio=TEST_SPLIT,
    )
    split_audit = audit_original_id_split_leakage(combined_split)
    logger.info(
        "Split assignment complete: rows=%s, groups=%s, leakage=%s, distribution=%s",
        len(combined_split),
        split_audit["unique_original_ids"],
        split_audit["has_leakage"],
        split_audit["split_row_counts"],
    )
    if split_audit["has_leakage"]:
        raise RuntimeError(
            "Split leakage detected after original_id assignment: "
            f"{split_audit['leaky_original_ids']} overlapping groups"
        )

    originals_balanced = [row for row in combined_split if row["label"] == 0]
    perturbed_balanced = [row for row in combined_split if row["label"] == 1]
    
    # Step 5: Save
    save_dataset(originals_balanced, perturbed_balanced, OUTPUT_DIR)
    
    logger.info("FinFact-BD dataset generation complete!")
    logger.info(f"Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
