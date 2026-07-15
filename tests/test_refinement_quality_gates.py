from __future__ import annotations

import pytest

from src.generation.claim_extraction import HeuristicClaimExtractor
from src.generation.claim_selection import ClaimRanker, ClaimRankingConfig
from src.generation.metadata import Article, Claim, RankedClaim, RewritePlan
from src.generation.perturbation_planner import build_planner
from src.generation.rewrite_generator import RewriteGenerator
from src.generation.verifier import CompositeVerifier, IntendedChangeVerifier


class StaticGenerator:
    model_name = "static-aya"
    model_revision = "test"

    def __init__(self, output: str) -> None:
        self.output = output

    def generate_batch(self, prompts, temperatures, seeds, max_new_tokens):
        return [self.output for _ in prompts]


def numerical_plan(target_span: str = "১শ’টির বেশি", replacement: str = "১০০টির বেশি") -> RewritePlan:
    claim = Claim(
        sentence_index=0,
        sentence="বিশ্বের ১শ’টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।",
        claim_type="numerical",
        entities=["মাহিন্দ্রা"],
        numbers=["১", "২০"],
        policies=[],
        dates=[],
        confidence=0.9,
    )
    return RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Change the target numeric fact.",
        edit_scope="target_sentence",
        expected_change="The country count changes.",
        verification_constraints={"preserve_all_other_sentences": True},
        target_span=target_span,
        replacement=replacement,
    )


def test_intended_change_rejects_numeric_surface_normalization():
    article = Article(
        article_id="a1",
        headline="মাহিন্দ্রা ব্যবসা করে",
        text="বিশ্বের ১শ’টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।",
    )
    rewritten = "বিশ্বের ১০০টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।"
    report = CompositeVerifier([IntendedChangeVerifier()]).verify(article, rewritten, numerical_plan())

    assert report.passed is False
    assert report.reasons == ["numeric_value_unchanged"]


def test_intended_change_accepts_real_numeric_change():
    article = Article(
        article_id="a1",
        headline="মাহিন্দ্রা ব্যবসা করে",
        text="বিশ্বের ১শ’টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।",
    )
    rewritten = "বিশ্বের ৯০টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।"
    report = CompositeVerifier([IntendedChangeVerifier()]).verify(article, rewritten, numerical_plan(replacement="৯০টির বেশি"))

    assert report.passed is True


def test_rewrite_generator_rejects_artifact_output():
    article = Article(
        article_id="a1",
        headline="বাংলাদেশ ব্যাংক সুদের হার বাড়িয়েছে",
        text="বাংলাদেশ ব্যাংক নীতিগত সুদের হার ১০ শতাংশ বাড়িয়েছে। বাজার স্থিতিশীল আছে।",
    )
    claim = HeuristicClaimExtractor().extract(article)[0]
    selected = RankedClaim(claim, 0.9, 0.9, 0.9, 0.9, 0.1, 0.9)
    plan = RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Change ১০ শতাংশ to ৭ শতাংশ.",
        edit_scope="target_sentence",
        expected_change="The rate changes.",
        verification_constraints={},
        target_span="১০ শতাংশ",
        replacement="৭ শতাংশ",
    )

    with pytest.raises(RuntimeError, match="text artifacts"):
        RewriteGenerator(StaticGenerator("বাংলাদেশ ব্যাংক নীতিগত সুদের হার ৭ শতাংশ বাড়িয়েছে �")).rewrite(
            article,
            plan,
            temperature=0.0,
            seed=1,
            attempt=1,
        )


def test_llm_planner_rejects_non_term_target_span_inside_word():
    article = Article(
        article_id="a1",
        headline="রবি ফোন এনেছে",
        text="গ্রাহকদের জন্য আকর্ষণীয় বান্ডেল অফারসহ এলজি স্মার্টফোন হ্যান্ডসেট বাজারে এনেছে মোবাইল ফোন অপারটের রবি।",
    )
    claim = HeuristicClaimExtractor(min_confidence=0.1).extract(article)[0]
    selected = ClaimRanker(ClaimRankingConfig(min_overall_score=0.1, max_risk_score=1.0)).select(article, [claim])
    assert selected is not None

    class BadPlanner:
        model_name = "bad-qwen"

        def generate_text(self, prompt, temperature, seed, max_new_tokens):
            return """
            {
              "family": "policy_reversal",
              "target_span": "কর",
              "replacement": "ট্যাক্স",
              "locality": "target_sentence",
              "edit_instruction": "Replace কর with ট্যাক্স.",
              "expected_change": "The policy term changes.",
              "verification_constraints": {}
            }
            """

    planner = build_planner({"backend": "llm_json", "allowed_families": ["policy_reversal"]}, model=BadPlanner())
    with pytest.raises(ValueError, match="target_span"):
        planner.create_plan(selected)
