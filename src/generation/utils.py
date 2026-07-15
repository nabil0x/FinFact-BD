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
NUMBER_RE = re.compile(r"[0-9\u09e6-\u09ef]+(?:[.,][0-9\u09e6-\u09ef]+)?%?")
DIGIT_PATTERN = r"[0-9\u09e6-\u09ef]+"
DECIMAL_PATTERN = rf"{DIGIT_PATTERN}(?:[.,]{DIGIT_PATTERN})?"
BANGLA_DECIMAL_RE = re.compile(rf"({DIGIT_PATTERN})\s+দশমিক\s+({DIGIT_PATTERN})")
NUMERIC_TOKEN_RE = re.compile(rf"({DECIMAL_PATTERN})(?:\s*(?:শ|শত|শতাংশ|%))?")
SCALE_NUMERIC_RE = re.compile(
    rf"({DIGIT_PATTERN}\s+দশমিক\s+{DIGIT_PATTERN}|{DECIMAL_PATTERN})"
    r"\s*(কোটি|লাখ|হাজার|শ|শত|শতাংশ|ভাগ|%|ডলার|টাকা|টাকার|টি|জন)?"
)
MONEY_UNIT_RE = re.compile(r"(?:টাকা|টাকার|ডলার|মার্কিন ডলার)")
NUMERIC_SCALE_WORD_RE = re.compile(r"(?:কোটি|লাখ|হাজার)")
HIGH_MONEY_SCALE_RE = re.compile(r"(?:কোটি|লাখ)")
PERCENT_UNIT_RE = re.compile(r"(?:শতাংশ|ভাগ|%)")
COUNT_UNIT_RE = re.compile(r"(?:টি|টির|টা|টার|জন|দেশ|কারখানা|হাসপাতাল|পয়েন্ট|প্যাকেট)")
DATE_RE = re.compile(
    r"(?:19\d{2}|20\d{2}|[০-৯]{4}|"
    r"জানুয়ারি|ফেব্রুয়ারি|মার্চ|এপ্রিল|মে|জুন|জুলাই|আগস্ট|"
    r"সেপ্টেম্বর|অক্টোবর|নভেম্বর|ডিসেম্বর|আজ|গতকাল|আগামীকাল|"
    r"গত বছর|চলতি বছর|আগামী বছর|গত মাস|চলতি মাস|আগামী মাস)"
)

ENTITY_ROLE_GROUPS = {
    "development_finance": {
        "এডিবি", "বিশ্বব্যাংক", "বিশ্ব ব্যাংক", "আইএমএফ", "আইএফসি", "জাইকা",
        "এশীয় উন্নয়ন ব্যাংক", "এশিয়ান ডেভেলপমেন্ট ব্যাংক",
        "এশিয়ান ডেভেলপমেন্ট ব্যাংক",
    },
    "regulator": {
        "বাংলাদেশ ব্যাংক", "কেন্দ্রীয় ব্যাংক", "কেন্দ্রীয় ব্যাংক", "এনবিআর",
        "জাতীয় রাজস্ব বোর্ড", "জাতীয় রাজস্ব বোর্ড", "বিএসইসি",
        "বাংলাদেশ সিকিউরিটিজ অ্যান্ড এক্সচেঞ্জ কমিশন",
        "বাংলাদেশ সিকিউরিটিজ এন্ড এক্সচেঞ্জ কমিশন", "পরিকল্পনা কমিশন",
        "বিটিআরসি", "বিইআরসি", "বিমা উন্নয়ন ও নিয়ন্ত্রণ কর্তৃপক্ষ",
        "আইডিআরএ", "রাজস্ব বোর্ড", "দুর্নীতি দমন কমিশন", "দুদক",
        "বাংলাদেশ টেলিযোগাযোগ নিয়ন্ত্রণ কমিশন",
        "বাংলাদেশ টেলিযোগাযোগ নিয়ন্ত্রণ কমিশন", "এক্সাইজ ও ভ্যাট কমিশন",
        "কর কমিশন", "সিকিউরিটিজ অ্যান্ড এক্সচেঞ্জ কমিশন", "কাস্টম কমিশন",
    },
    "ministry_government": {
        "সরকার", "বাংলাদেশ সরকার", "অর্থ মন্ত্রণালয়", "অর্থ মন্ত্রণালয়",
        "বাণিজ্য মন্ত্রণালয়", "বাণিজ্য মন্ত্রণালয়", "শিল্প মন্ত্রণালয়",
        "শিল্প মন্ত্রণালয়", "কৃষি মন্ত্রণালয়", "কৃষি মন্ত্রণালয়",
        "পরিকল্পনা মন্ত্রণালয়", "পরিকল্পনা মন্ত্রণালয়",
        "বিমান পরিবহন ও পর্যটন মন্ত্রণালয়", "বিমান পরিবহন ও পর্যটন মন্ত্রণালয়",
        "বেসামরিক বিমান পরিবহন ও পর্যটন মন্ত্রণালয়",
        "বস্ত্র ও পাট মন্ত্রণালয়", "বস্ত্র ও পাট মন্ত্রণালয়",
        "নৌপরিবহন মন্ত্রণালয়", "প্রবাসী কল্যাণ ও বৈদেশিক কর্মসংস্থান মন্ত্রণালয়",
        "জনপ্রশাসন মন্ত্রণালয়", "শ্রম ও কর্মসংস্থান মন্ত্রণালয়",
        "শ্রম মন্ত্রণালয়", "স্বরাষ্ট্র মন্ত্রণালয়", "খাদ্য মন্ত্রণালয়",
        "রেলপথ মন্ত্রণালয়", "জ্বালানি ও খনিজ সম্পদ মন্ত্রণালয়",
        "পররাষ্ট্র মন্ত্রণালয়", "পল্লী উন্নয়ন ও সমবায় মন্ত্রণালয়",
        "নৌ-পরিবহন মন্ত্রণালয়", "মৎস্য ও প্রাণিসম্পদ মন্ত্রণালয়",
        "পানিসম্পদ মন্ত্রণালয়", "বন ও জলবায়ু পরিবর্তন মন্ত্রণালয়",
        "সড়ক পরিবহন ও সেতু মন্ত্রণালয়", "প্রাথমিক ও গণশিক্ষা মন্ত্রণালয়",
        "স্বাস্থ্য ও পরিবার কল্যাণ মন্ত্রণালয়", "ভূমি মন্ত্রণালয়",
        "স্বাস্থ্য মন্ত্রণালয়", "প্রতিরক্ষা মন্ত্রণালয়",
        "গৃহায়ণ ও গণপূর্ত মন্ত্রণালয়", "গৃহায়ন ও গণপূর্ত মন্ত্রণালয়",
        "বিজ্ঞান ও প্রযুক্তি মন্ত্রণালয়", "শিক্ষা মন্ত্রণালয়",
        "ধর্ম মন্ত্রণালয়", "দুর্যোগ ব্যবস্থাপনা ও ত্রাণ মন্ত্রণালয়",
        "দুর্যোগ ব্যবস্থাপনা ও ত্রাণ মন্ত্রণালয়", "জ্বালানি মন্ত্রণালয়",
        "ডাক ও টেলিযোগাযোগ মন্ত্রণালয়", "পরিবেশ ও বন মন্ত্রণালয়",
        "মহিলা ও শিশু বিষয়ক মন্ত্রণালয়",
    },
    "market": {
        "ডিএসই", "ঢাকা স্টক এক্সচেঞ্জ", "সিএসই", "চট্টগ্রাম স্টক এক্সচেঞ্জ",
        "নিউইয়র্ক মার্কেন্টাইল এক্সচেঞ্জ",
    },
    "state_owned_bank": {
        "সোনালী ব্যাংক", "জনতা ব্যাংক", "রূপালী ব্যাংক", "অগ্রণী ব্যাংক",
        "বেসিক ব্যাংক", "বাংলাদেশ কৃষি ব্যাংক", "কৃষি ব্যাংক",
        "রাজশাহী কৃষি উন্নয়ন ব্যাংক", "কর্মসংস্থান ব্যাংক",
        "প্রবাসী কল্যাণ ব্যাংক", "পল্লী সঞ্চয় ব্যাংক",
    },
    "commercial_bank": {
        "সিটি ব্যাংক", "ব্র্যাক ব্যাংক", "ইসলামী ব্যাংক", "ইসলামি ব্যাংক",
        "ইসলামী ব্যাংক বাংলাদেশ লিমিটেড", "ডাচ-বাংলা ব্যাংক", "মার্কেন্টাইল ব্যাংক",
        "প্রাইম ব্যাংক", "ন্যাশনাল ব্যাংক", "পূবালী ব্যাংক", "আইএফআইসি ব্যাংক",
        "এক্সিম ব্যাংক", "এনসিসি ব্যাংক", "উত্তরা ব্যাংক", "ওয়ান ব্যাংক",
        "সাউথইস্ট ব্যাংক", "ঢাকা ব্যাংক", "এবি ব্যাংক", "প্রিমিয়ার ব্যাংক",
        "ইস্টার্ন ব্যাংক", "আল-আরাফাহ্ ইসলামী ব্যাংক", "আল-আরাফাহ ইসলামী ব্যাংক",
        "সোস্যাল ইসলামী ব্যাংক", "স্ট্যান্ডার্ড চার্টার্ড ব্যাংক",
        "ফার্স্ট সিকিউরিটি ইসলামী ব্যাংক", "স্ট্যান্ডার্ড ব্যাংক",
        "শাহ্জালাল ইসলামী ব্যাংক", "শাহজালাল ইসলামী ব্যাংক",
        "ইউনাইটেড কমার্শিয়াল ব্যাংক", "এনআরবিসি ব্যাংক", "এনআরবি ব্যাংক",
        "ট্রাস্ট ব্যাংক", "গ্রামীণ ব্যাংক", "মিডল্যান্ড ব্যাংক",
        "মিউচুয়াল ট্রাস্ট ব্যাংক", "মধুমতি ব্যাংক", "এসবিএসি ব্যাংক",
        "এইচএসবিসি", "ব্যাংক এশিয়া", "ব্যাংক এশিয়া", "যমুনা ব্যাংক",
        "ডাচ্-বাংলা ব্যাংক", "ফারমার্স ব্যাংক", "মেঘনা ব্যাংক",
        "পদ্মা ব্যাংক", "বাংলাদেশ ডেভেলপমেন্ট ব্যাংক", "দি সিটি ব্যাংক",
        "সোনালী ব্যাংক লিমিটেড", "মার্কেন্টাইল ব্যাংক লিমিটেড",
        "ইস্টার্ন ব্যাংক লিমিটেড", "ইউনাইটেড কমার্শিয়াল ব্যাংক লিমিটেড",
        "সোস্যাল ইসলামী ব্যাংক লিমিটেড", "স্ট্যান্ডার্ড ব্যাংক লিমিটেড",
        "মিউচুয়াল ট্রাস্ট ব্যাংক লিমিটেড", "প্রাইম ব্যাংক লিমিটেড",
        "জনতা ব্যাংক লিমিটেড", "সাউথ বাংলা অ্যাগ্রিকালচার অ্যান্ড কমার্স ব্যাংক",
        "সাউথ বাংলা এগ্রিকালচার অ্যান্ড কমার্স ব্যাংক", "সাউথ বাংলা ব্যাংক",
        "বাংলাদেশ কমার্স ব্যাংক", "ইউনিয়ন ব্যাংক", "স্টেট ব্যাংক",
        "উরি ব্যাংক", "পিপলস ব্যাংক", "সীমান্ত ব্যাংক", "ইউসিবি ব্যাংক",
        "কমিউনিটি ব্যাংক", "ব্রাক ব্যাংক", "শাহজালাল ব্যাংক",
        "বেঙ্গল কমার্শিয়াল ব্যাংক", "স্টান্ডার্ড চার্টার্ড ব্যাংক",
    },
    "non_bank_finance": {
        "আইডিএলসি", "আইডিএলসি ফাইন্যান্স", "আইপিডিসি", "আইপিডিসি ফাইন্যান্স",
        "লংকা-বাংলা ফাইন্যান্স", "লংকাবাংলা ফাইন্যান্স", "ডেল্টা ব্র্যাক হাউজিং",
        "ডিবিএইচ", "বিএইচবিএফসি", "বাংলাদেশ হাউস বিল্ডিং ফাইন্যান্স করপোরেশন",
        "পিকেএসএফ", "এসএমই ফাউন্ডেশন", "বিডি ফাইন্যান্স", "লঙ্কাবাংলা ফাইন্যান্স",
        "ইউনিয়ন ক্যাপিটাল", "ফারইস্ট ফাইন্যান্স", "প্রাইম ফাইন্যান্স",
        "জিএসপি ফাইন্যান্স", "মাইডাস ফাইন্যান্স", "ফার্স্ট ফাইন্যান্স",
        "প্রিমিয়াম সিকিউরিটিজ",
        "পিপলস লিজিং অ্যান্ড ফিন্যান্সিয়াল সার্ভিসেস লিমিটেড",
        "ফাস ফাইন্যান্স", "ন্যাশনাল হাউজিং ফাইন্যান্স",
        "ইসলামিক ফাইন্যান্স", "ইউনাইটেড ফাইন্যান্স",
    },
    "trade_body": {
        "এফবিসিসিআই", "বিজিএমইএ", "বিকেএমইএ", "ডিসিসিআই", "এমসিসিআই",
        "বিসিআই", "বিটিএমএ", "বিজিএপিএমইএ", "ইআরএফ",
    },
    "state_corporation": {
        "বিজেএমসি", "বিসিআইসি", "বিপিসি", "বিআরটিসি", "ডিপিডিসি",
        "বিএডিসি", "বাংলাদেশ পেট্রোলিয়াম করপোরেশন",
        "বাংলাদেশ পেট্রোলিয়াম করপোরেশন", "ক্ষুদ্র ও কুটির শিল্প করপোরেশন",
        "বাংলাদেশ ক্ষুদ্র ও কুটির শিল্প করপোরেশন", "বিসিক",
        "বাংলাদেশ বিদ্যুৎ উন্নয়ন বোর্ড", "পিডিবি",
        "বাংলাদেশ চিনি ও খাদ্য শিল্প করপোরেশন", "বাংলাদেশ ট্যুরিজম বোর্ড",
        "বিনিয়োগ বোর্ড", "ট্যারিফ কমিশন", "নির্বাচন কমিশন",
        "যুক্তরাষ্ট্রের ফেডারেল রিজার্ভ ব্যাংক", "ফেডারেল রিজার্ভ ব্যাংক",
        "রিজার্ভ ব্যাংক", "বাংলাদেশ পর্যটন করপোরেশন", "পর্যটন করপোরেশন",
        "ট্যুরিজম বোর্ড", "সিটি করপোরেশন",
        "বাংলাদেশ পানি উন্নয়ন বোর্ড", "পানি উন্নয়ন বোর্ড",
        "বাংলাদেশ পল্লী বিদ্যুতায়ন বোর্ড", "পল্লী বিদ্যুতায়ন বোর্ড",
        "তিতাস গ্যাস ট্রান্সমিশন অ্যান্ড ডিস্ট্রিবিউশন কোম্পানি",
        "তিতাস গ্যাস", "ডেসকো", "ওয়াসা", "বাংলাদেশ পাটকল করপোরেশন",
        "বাংলাদেশ শিপিং করপোরেশন", "ঢাকা উত্তর সিটি করপোরেশন",
        "ঢাকা দক্ষিণ সিটি করপোরেশন", "ঢাকা উত্তর ও দক্ষিণ সিটি করপোরেশন",
        "চট্টগ্রাম সিটি করপোরেশন", "বাংলাদেশ কেমিক্যাল ইন্ডাস্ট্রিজ করপোরেশন",
        "ইনভেস্টমেন্ট করপোরেশন", "কর্ণফুলী গ্যাস ডিস্ট্রিবিউশন কোম্পানি লিমিটেড",
        "ট্রেডিং করপোরেশন", "ট্রেডিং করপোরেশন অব বাংলাদেশ",
    },
    "insurance": {
        "ইউনাইটেড ইন্স্যুরেন্স", "পাইওনিয়ার ইন্স্যুরেন্স",
        "বাংলাদেশ ন্যাশনাল ইন্স্যুরেন্স", "প্রগতি ইন্স্যুরেন্স",
        "গ্লোবাল ইন্স্যুরেন্স", "রূপালী ইন্স্যুরেন্স", "রিপাবলিক ইন্স্যুরেন্স",
        "নিটল ইন্স্যুরেন্স", "এশিয়া প্যাসিফিক জেনারেল ইন্স্যুরেন্স",
        "প্রভাতী ইন্স্যুরেন্স", "কন্টিনেন্টাল ইন্স্যুরেন্স",
        "রিলায়েন্স ইনস্যুরেন্স", "রিলায়েন্স ইন্স্যুরেন্স",
        "বাংলাদেশ ইনস্যুরেন্স", "বাংলাদেশ ইন্স্যুরেন্স",
        "সেন্ট্রাল ইন্স্যুরেন্স", "মার্কেন্টাইল ইন্স্যুরেন্স",
        "সোনার বাংলা ইন্স্যুরেন্স", "ন্যাশনাল লাইফ ইনস্যুরেন্স",
        "ফেডারেল ইন্স্যুরেন্স", "রূপালী লাইফ ইন্স্যুরেন্স",
        "গ্রিন ডেল্টা ইনস্যুরেন্স", "গ্রিন ডেল্টা ইন্স্যুরেন্স",
        "ইস্টার্ন ইনস্যুরেন্স", "এশিয়া ইন্স্যুরেন্স",
        "এশিয়া প্যাসিফিক ইনস্যুরেন্স", "এশিয়া প্যাসেফিক ইন্স্যুরেন্স",
        "ঢাকা ইন্স্যুরেন্স", "স্ট্যান্ডার্ড ইন্স্যুরেন্স",
        "প্রাইম লাইফ ইন্স্যুরেন্স", "তাকাফুল ইন্স্যুরেন্স",
        "এক্সপ্রেস ইন্স্যুরেন্স", "বাংলাদেশ জেনারেল ইন্স্যুরেন্স",
        "ইস্টার্ন ইন্স্যুরেন্স", "পিপলস ইন্স্যুরেন্স", "রিপাবলিক ইনস্যুরেন্স",
    },
    "mobile_finance": {"বিকাশ", "নগদ", "রকেট", "উপায়", "উপায়"},
    "telecom": {
        "গ্রামীণফোন", "রবি", "বাংলালিংক", "টেলিটক", "এয়ারটেল", "এয়ারটেল",
        "এডিএন টেলিকম", "টেলিনর গ্রুপ",
    },
    "company_group": {
        "মাহিন্দ্রা", "এস আলম গ্রুপ", "এস আলম গ্র“প", "বসুন্ধরা গ্রুপ",
        "বেক্সিমকো", "বেক্সিমকো লিমিটেড", "বিএসআরএম", "বিএসআরএম লিমিটেড",
        "আরএসআরএম", "আরএসআরএম লিমিটেড", "কেডিএস", "কেডিএস গ্রুপ",
        "প্রাণ-আরএফএল", "আরএফএল", "স্কয়ার", "স্কয়ার", "ওয়ালটন", "ওয়ালটন",
        "সামিট গ্রুপ", "মেঘনা গ্রুপ", "আকিজ গ্রুপ", "প্রাণ-আরএফএল গ্রুপ",
        "এসিআই লিমিটেড", "বিল অ্যান্ড মেলিন্ডা গেটস ফাউন্ডেশন",
        "বেক্সিমকো ফার্মা", "প্রাণ গ্রুপ", "যমুনা গ্রুপ", "এডিসন গ্রুপ",
        "সিটি গ্রুপ", "ফেয়ার গ্রুপ", "ডিবিএল গ্রুপ", "রূপায়ণ গ্রুপ",
        "আনোয়ার গ্রুপ", "নিটল-নিলয় গ্রুপ", "ম্যারিকো বাংলাদেশ লিমিটেড",
        "ইস্ট কোস্ট গ্রুপ", "লাফার্জ সুরমা সিমেন্ট লিমিটেড",
        "বসুন্ধরা এলপি গ্যাস লিমিটেড", "বার্জার পেইন্টস বাংলাদেশ লিমিটেড",
        "আইএসএন লিমিটেড", "ইনটেক লিমিটেড", "প্রাণ এগ্রো লিমিটেড",
        "আবুল খায়ের গ্রুপ",
    },
    "research_training": {
        "বিআইবিএম", "বাংলাদেশ ইনস্টিটিউট অব ব্যাংক ম্যানেজমেন্ট",
        "বাংলাদেশ ইন্সটিটিউট অব ব্যাংক ম্যানেজমেন্ট",
        "বাংলাদেশ ইনস্টিটিউট অব ব্যাংক", "বাংলাদেশ ইন্সটিটিউট অব ব্যাংক",
        "বিআইডিএস", "বাংলাদেশ উন্নয়ন গবেষণা প্রতিষ্ঠান", "ইসলামিক ফাউন্ডেশন",
        "পলিসি এক্সচেঞ্জ", "শক্তি ফাউন্ডেশন",
    },
}

ENTITY_TERMS = set().union(*ENTITY_ROLE_GROUPS.values())

GENERIC_ENTITY_PREFIXES = {
    "এ", "এই", "এতে", "এদিন", "এ সময়", "এ বিষয়ে", "এ বিষয়ে", "তবে",
    "ফলে", "আর", "তিনি", "যা", "বর্তমানে", "দেশের", "বিভিন্ন", "অনুষ্ঠানে",
    "জানতে চাইলে", "এ ছাড়া", "এছাড়া", "প্রধান", "অপর", "দুই", "সরকারি",
    "বেসরকারি", "বাণিজ্যিক", "রাষ্ট্রায়ত্ত", "রাষ্ট্রায়ত্ত", "মোবাইল",
    "মার্চেন্ট", "এজেন্ট", "ইন্টারনেট", "অ্যান্ড", "এন্ড", "ও", "এবং",
    "শেয়ারবাজার", "শেয়ারবাজার", "পুঁজিবাজার", "শীর্ষ", "সম্প্রতি",
    "এর", "এর ফলে", "কিন্তু", "এসব", "লেনদেন", "লেনদেন হওয়া",
    "লেনদেনে", "একই", "একই সঙ্গে", "অন্যান্য", "যেসব", "কোনো",
    "কোনো কোনো", "আমাদের", "পুঁজিবাজারে", "সংশ্লিষ্ট", "বাংলাদেশের",
    "অন্যদিকে", "অনেক", "দিনশেষে", "এখন", "অনলাইন", "কারণ", "নতুন",
    "সব", "যারা", "দর", "দর বৃদ্ধির", "দাম", "দাম কমার", "অন্য",
    "এর মধ্যে", "এরপর", "তাই", "এদিকে", "এর আগে", "আবার",
    "তালিকাভুক্ত", "পাশাপাশি", "গতকাল", "পরে", "সভায়", "বীমা",
    "ডিজিটাল", "একটি", "নিয়ন্ত্রক", "নিয়ন্ত্রক", "অ্যাসোসিয়েশন",
    "অ্যাসোসিয়েশন", "অথচ", "ওই", "বহুজাতিক", "যদিও", "করে",
    "এসময়", "এ সময়", "ঘোষণা", "মূলত", "রাজধানীর", "আলোচ্য",
    "তফসিলি", "তা ছাড়া", "এটি", "সভাপতিত্ব", "শেয়ারবাজারে",
    "অর্থাৎ", "অফশোর", "স্কুল", "লেনদেনকৃত", "ব্যাংকের", "সাধারণত",
    "অধিকাংশ", "সঞ্চয়পত্রের", "নামে", "খাতের", "যেখানে", "বিদেশি",
    "সে", "দুর্বল", "ক্যাটাগরিভুক্ত", "উল্টো", "গত", "প্রতি",
    "একইভাবে", "নির্বাহীদের", "এছাড়া", "আমরা", "পুরো", "সম্মেলনে",
    "অতিথি", "অতিথি হিসেবে", "হয়েছে", "এক্ষেত্রে", "তারা", "এরই মধ্যে",
    "বেশির ভাগ", "যে কারণে", "এমন", "এমন পরিস্থিতিতে", "এজন্য",
    "বাংলাদেশে", "গভর্নর", "জানতে", "সেখানে", "লাখ", "করেছে",
    "যে", "সুতরাং", "সংশোধিত", "এ-সংক্রান্ত", "আরো", "কয়েকটি",
    "কিছু", "তিন", "র", "উপস্থিত", "প্রচলিত", "যাতে", "বর্তমান",
    "প্রতিটি", "বড়",
}

GENERIC_ENTITY_CANDIDATES = {
    "ব্যাংক", "কোম্পানি", "কম্পানি", "গ্রুপ", "গ্র“প", "বোর্ড", "কমিশন",
    "ফাউন্ডেশন", "ফাইন্যান্স", "লিমিটেড", "লি.", "মন্ত্রণালয়", "মন্ত্রণালয়",
    "কর্পোরেট", "টেলিকম", "অপারেটর", "স্টক এক্সচেঞ্জ", "বাংলাদেশ লিমিটেড",
    "কোম্পানি লিমিটেড", "ম্যানেজমেন্ট লিমিটেড", "ব্যাংক কোম্পানি",
    "ব্যাংক কম্পানি", "বিদেশি ব্যাংক", "তফসিলি ব্যাংক", "বিশেষায়িত ব্যাংক",
    "ডিজিটাল ব্যাংক", "অফশোর ব্যাংক", "মোবাইল ব্যাংক", "বাংলাদেশ মার্চেন্ট ব্যাংক",
    "কমার্শিয়াল ব্যাংক", "ইন্স্যুরেন্স কোম্পানি", "ইনস্যুরেন্স কোম্পানি",
    "তামাক কোম্পানি", "নির্বাচন বোর্ড", "বন্ড কমিশন", "যুগ্ম কমিশন",
    "গ্রিন ব্যাংক", "এসএমই ব্যাংক", "অ্যাসেট ম্যানেজমেন্ট কম্পানি",
    "ভেঞ্চার ক্যাপিটাল", "কোম্পানী লিমিটেড",
}

ORG_TOKEN_PATTERN = r"[\u0980-\u09FFA-Za-z&().\-]+"
ORG_SUFFIX_RE = re.compile(
    rf"(?:{ORG_TOKEN_PATTERN}\s+){{0,5}}"
    r"(?:ব্যাংক|মন্ত্রণালয়|মন্ত্রণালয়|বোর্ড|কমিশন|কর্পোরেশন|করপোরেশন|"
    r"কোম্পানি|কম্পানি|গ্রুপ|গ্র“প|লিমিটেড|লি\.|এক্সচেঞ্জ|ফাউন্ডেশন|"
    r"ইন্স্যুরেন্স|ইনস্যুরেন্স|ইনসিওরেন্স|ফাইন্যান্স|সিকিউরিটিজ|ক্যাপিটাল|টেলিকম)"
)

POLICY_TERMS = {
    "সুদের হার", "নীতিগত সুদের হার", "কর", "ভ্যাট", "শুল্ক",
    "সাবসিডি", "দাম", "মূল্য", "নীতিমালা", "নির্দেশনা",
    "আরোপ", "প্রত্যাহার", "বাড়িয়েছে", "কমিয়েছে", "বৃদ্ধি", "হ্রাস",
}

CAUSAL_TERMS = {
    "কারণে", "ফলে", "সুতরাং", "অতএব", "যেহেতু", "এর ফলে", "এ কারণে",
}

TEMPORAL_TERMS = {
    "জানুয়ারি", "ফেব্রুয়ারি", "মার্চ", "এপ্রিল", "মে", "জুন", "জুলাই", "আগস্ট",
    "সেপ্টেম্বর", "অক্টোবর", "নভেম্বর", "ডিসেম্বর", "অর্থবছর", "বছর", "মাস",
    "সপ্তাহ", "দিন", "আজ", "গতকাল", "আগামীকাল", "চলতি", "গত", "আগামী",
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


def normalize_digits(text: str) -> str:
    table = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")
    return text.translate(table)


def _numeric_float(raw: str) -> float | None:
    normalized = normalize_digits(raw).replace(" ", "").replace(",", ".")
    if "দশমিক" in raw:
        parts = [normalize_digits(part.strip()) for part in raw.split("দশমিক", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            normalized = f"{parts[0]}.{parts[1]}"
    try:
        return float(normalized)
    except ValueError:
        return None


def numeric_values(text: str) -> List[float]:
    values: List[float] = []
    occupied: List[range] = []
    for match in BANGLA_DECIMAL_RE.finditer(text):
        value = _numeric_float(match.group())
        if value is None:
            continue
        values.append(value)
        occupied.append(range(match.start(), match.end()))
    for match in NUMERIC_TOKEN_RE.finditer(text):
        if any(match.start() in span or match.end() - 1 in span for span in occupied):
            continue
        value = _numeric_float(match.group(1))
        if value is None:
            continue
        suffix = match.group(0)
        if "শতাংশ" not in suffix and "%" not in suffix and re.search(r"শ(?:[’'\u2019]|\s|টি|টার|$)", suffix):
            value *= 100
        values.append(value)
    return values


def scaled_numeric_values(text: str) -> List[float]:
    values: List[float] = []
    for match in SCALE_NUMERIC_RE.finditer(text):
        value = _numeric_float(match.group(1))
        if value is None:
            continue
        unit = match.group(2) or ""
        if unit == "কোটি":
            value *= 10_000_000
        elif unit == "লাখ":
            value *= 100_000
        elif unit == "হাজার":
            value *= 1_000
        elif unit in {"শ", "শত"}:
            value *= 100
        values.append(value)
    return values or numeric_values(text)


def significant_numeric_scale_change(original: str, rewritten: str, min_factor: float = 5.0) -> bool:
    original_values = scaled_numeric_values(original)
    rewritten_values = scaled_numeric_values(rewritten)
    if not original_values or not rewritten_values:
        return False
    original_value = original_values[0]
    rewritten_value = rewritten_values[0]
    if original_value == rewritten_value:
        return False
    if original_value == 0:
        return rewritten_value != 0
    ratio = abs(rewritten_value / original_value)
    return ratio > min_factor or ratio < (1.0 / min_factor)


def numeric_unit_mismatch_reason(target: str, replacement: str) -> str | None:
    target_profile = _numeric_unit_profile(target)
    replacement_profile = _numeric_unit_profile(replacement)
    if not target_profile["has_numeric"] or not replacement_profile["has_numeric"]:
        return None
    if target_profile["count"] and not target_profile["money"] and replacement_profile["percent"]:
        return None
    if target_profile["percent"] != replacement_profile["percent"]:
        return "numeric_unit_mismatch_percent"
    if target_profile["money"] != replacement_profile["money"]:
        if target_profile["money"] or replacement_profile["money"]:
            return "numeric_unit_mismatch_money"
    return None


def _numeric_unit_profile(text: str) -> Dict[str, bool]:
    has_numeric = bool(numeric_values(text))
    money = bool(MONEY_UNIT_RE.search(text))
    scale = bool(NUMERIC_SCALE_WORD_RE.search(text))
    return {
        "has_numeric": has_numeric,
        "money": money,
        "percent": bool(PERCENT_UNIT_RE.search(text)),
        "count": bool(COUNT_UNIT_RE.search(text)) and not money,
        "scale": scale,
        "high_money_scale": bool(HIGH_MONEY_SCALE_RE.search(text)),
        "bare_money": money and not scale,
    }


def numeric_values_equivalent(left: str, right: str) -> bool:
    left_values = scaled_numeric_values(left)
    right_values = scaled_numeric_values(right)
    return bool(left_values and right_values and left_values == right_values)


def is_temporal_span(text: str) -> bool:
    normalized = normalize_digits(text)
    if any(term in text for term in TEMPORAL_TERMS):
        return True
    if re.search(r"(?:19|20)\d{2}(?:[-–](?:\d{2}|(?:19|20)\d{2}))?", normalized):
        return True
    return bool(extract_dates(text))


def extract_dates(text: str) -> List[str]:
    return [match.group() for match in DATE_RE.finditer(text)]


def extract_terms(text: str, terms: Iterable[str]) -> List[str]:
    found = [term for term in terms if term and term in text]
    return sorted(set(found), key=lambda value: (text.find(value), value))


def span_occurs_as_term(text: str, span: str) -> bool:
    span = span.strip()
    if not span:
        return False
    escaped = re.escape(span)
    pattern = rf"(?<![\u0980-\u09FF0-9\u09e6-\u09ef]){escaped}(?![\u0980-\u09FF0-9\u09e6-\u09ef])"
    return re.search(pattern, text) is not None or span in text.split()


def extract_entities(text: str) -> List[str]:
    entities = extract_terms(text, ENTITY_TERMS)
    suffix_matches = [_clean_entity_candidate(match.group(0)) for match in ORG_SUFFIX_RE.finditer(text)]
    suffix_matches = [match for match in suffix_matches if match and _looks_like_entity_candidate(match)]
    return sorted(set(entities + suffix_matches), key=lambda value: (text.find(value), value))


def _clean_entity_candidate(text: str) -> str:
    candidate = " ".join(text.strip(" :;,.।()[]{}").split())
    known = _known_entity_inside(candidate)
    if known:
        return known
    candidate = re.sub(r"^(?:অর্থনৈতিক|স্টাফ|নিজস্ব|কূটনৈতিক|বিশেষ|বাণিজ্যিক)\s+", "", candidate)
    known = _known_entity_inside(candidate)
    if known:
        return known
    return candidate.strip(" :;,.।")


def _known_entity_inside(candidate: str) -> str:
    matches = [term for term in ENTITY_TERMS if term in candidate]
    if not matches:
        return ""
    return max(matches, key=len)


def _looks_like_entity_candidate(candidate: str) -> bool:
    if len(candidate) < 4:
        return False
    tokens = candidate.split()
    if not tokens:
        return False
    if re.match(rf"^(?:{DIGIT_PATTERN}|[0-9])", tokens[0]):
        return False
    if re.match(r"^[\u09be-\u09cc\u09cd]", tokens[0]):
        return False
    if candidate in GENERIC_ENTITY_CANDIDATES:
        return False
    if candidate not in ENTITY_TERMS and any(char in candidate for char in "()"):
        return False
    if tokens[0] in GENERIC_ENTITY_PREFIXES or " ".join(tokens[:2]) in GENERIC_ENTITY_PREFIXES:
        return False
    if candidate not in ENTITY_TERMS and len(tokens) == 1:
        return False
    return True


def entity_role(entity: str) -> str | None:
    compact = entity.replace(" ", "")
    for role, values in ENTITY_ROLE_GROUPS.items():
        compact_values = {value.replace(" ", "") for value in values}
        if entity in values or compact in compact_values:
            return role
    return None


def entities_are_same_role(left: str, right: str) -> bool:
    left_role = entity_role(left)
    right_role = entity_role(right)
    return bool(left_role and right_role and left_role == right_role)


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


def target_sentence(text: str, sentence_index: int) -> str:
    spans = sentence_spans(text)
    if 0 <= sentence_index < len(spans):
        return spans[sentence_index].text
    return ""


def replace_first_exact(text: str, target: str, replacement: str) -> str:
    if not target:
        return text
    index = text.find(target)
    if index < 0:
        return text
    return text[:index] + replacement + text[index + len(target) :]


def replace_all_exact(text: str, target: str, replacement: str) -> str:
    if not target:
        return text
    return text.replace(target, replacement)


def artifact_reasons(text: str) -> List[str]:
    reasons: List[str] = []
    suspicious_fragments = ("প্রব্যঙ্গ", "নির্নয়ন", "ন঵", "অনুষ্")
    if "\ufffd" in text or "�" in text:
        reasons.append("replacement_character")
    if re.search(r"[\u0980-\u09FF]{1,2}্(?:\s|$)", text):
        reasons.append("dangling_halant_fragment")
    if any(fragment in text for fragment in suspicious_fragments):
        reasons.append("suspicious_bangla_fragment")
    if re.search(r"(?<![\u0980-\u09FF])কনসা(?![\u0980-\u09FF])", text):
        reasons.append("suspicious_bangla_fragment")
    if re.search(r"([\u0980-\u09FF]{3,}(?:\s+[\u0980-\u09FF]{3,}){0,2})\s+\1", text):
        reasons.append("repeated_bangla_fragment")
    return reasons


def has_text_artifacts(text: str) -> bool:
    return bool(artifact_reasons(text))


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
