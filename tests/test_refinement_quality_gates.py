from __future__ import annotations

import pytest

from src.generation.claim_extraction import HeuristicClaimExtractor
from src.generation.claim_selection import ClaimRanker, ClaimRankingConfig
from src.generation.metadata import Article, Claim, RankedClaim, RewritePlan
from src.generation.perturbation_planner import build_planner, validate_rewrite_plan
from src.generation.rewrite_generator import RewriteGenerator
from src.generation.utils import ENTITY_TERMS, artifact_reasons, entities_are_same_role, extract_entities
from src.generation.verifier import CompositeVerifier, ContradictionVerifier, IntendedChangeVerifier, TextQualityArtifactVerifier


class StaticGenerator:
    model_name = "static-aya"
    model_revision = "test"

    def __init__(self, output: str) -> None:
        self.output = output

    def generate_batch(self, prompts, temperatures, seeds, max_new_tokens):
        return [self.output for _ in prompts]


class ExplodingGenerator:
    model_name = "exploding-generator"
    model_revision = "test"

    def generate_batch(self, prompts, temperatures, seeds, max_new_tokens):
        raise AssertionError("controlled rewrite should not call the generation model")


class LowContradictionNLI:
    model_name = "low-nli"

    def contradiction_score(self, premise, hypothesis):
        return 0.01


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

    rewritten = "বিশ্বের ১০০০টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।"
    report = CompositeVerifier([IntendedChangeVerifier()]).verify(article, rewritten, numerical_plan(replacement="১০۰۰টির বেশি"))

    rewritten = "বিশ্বের 1000টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।"
    report = CompositeVerifier([IntendedChangeVerifier()]).verify(article, rewritten, numerical_plan(replacement="1000টির বেশি"))

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
        edit_instruction="Change ১০ শতাংশ to 100 শতাংশ.",
        edit_scope="target_sentence",
        expected_change="The rate changes.",
        verification_constraints={},
        target_span="১০ শতাংশ",
        replacement="100 শতাংশ",
    )

    plan = RewritePlan(
        family="policy_reversal",
        target_claim=claim,
        edit_instruction="Create an artifact output for test.",
        edit_scope="target_sentence",
        expected_change="The rate changes.",
        verification_constraints={},
        target_span="",
        replacement="",
    )

    with pytest.raises(RuntimeError, match="text artifacts"):
        RewriteGenerator(StaticGenerator("বাংলাদেশ ব্যাংক নীতিগত সুদের হার 100 শতাংশ বাড়িয়েছে �")).rewrite(
            article,
            plan,
            temperature=0.0,
            seed=1,
            attempt=1,
        )


def test_controlled_numeric_rewrite_uses_exact_scale_replacement_without_model_call():
    article = Article(
        article_id="a1",
        headline="সিটি ব্যাংকের অনুদান",
        text="সিটি ব্যাংক ডিএমপিকে ৫০ লাখ টাকার অনুদান প্রদান করেছে। বাজার স্থিতিশীল আছে।",
    )
    claim = Claim(
        sentence_index=0,
        sentence="সিটি ব্যাংক ডিএমপিকে ৫০ লাখ টাকার অনুদান প্রদান করেছে।",
        claim_type="numerical",
        entities=["সিটি ব্যাংক"],
        numbers=["৫০"],
        policies=[],
        dates=[],
        confidence=0.9,
    )
    plan = RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Inflate the donation amount.",
        edit_scope="target_sentence",
        expected_change="The donation amount is much larger.",
        verification_constraints={},
        target_span="৫০ লাখ টাকার",
        replacement="৫ কোটি টাকার",
    )

    rewritten = RewriteGenerator(ExplodingGenerator()).rewrite(article, plan, temperature=0.4, seed=1, attempt=1)
    report = CompositeVerifier([IntendedChangeVerifier()]).verify(article, rewritten.rewritten_article, plan)

    assert rewritten.rewritten_article == "সিটি ব্যাংক ডিএমপিকে ৫ কোটি টাকার অনুদান প্রদান করেছে। বাজার স্থিতিশীল আছে।"
    assert report.passed is True


def test_controlled_policy_rewrite_uses_exact_replacement_without_model_call():
    article = Article(
        article_id="a1",
        headline="সুদের বাধা",
        text="অবশেষে বহুল প্রত্যাশিত ব্যাংক ঋণের সুদের হার সিঙ্গেল ডিজিট ও সরল সুদ চালুর বাধা কাটল।",
    )
    claim = Claim(
        sentence_index=0,
        sentence=article.text,
        claim_type="policy",
        entities=["ব্যাংক"],
        numbers=[],
        policies=["বাধা কাটল"],
        dates=[],
        confidence=0.9,
    )
    plan = RewritePlan(
        family="policy_reversal",
        target_claim=claim,
        edit_instruction="Reverse the policy barrier direction.",
        edit_scope="target_sentence",
        expected_change="The implementation barrier increases instead of being removed.",
        verification_constraints={},
        target_span="সরল সুদ চালুর বাধা কাটল",
        replacement="সরল সুদ চালুর বাধা আরও বেড়েছে",
    )

    rewritten = RewriteGenerator(ExplodingGenerator()).rewrite(article, plan, temperature=0.4, seed=1, attempt=1)

    assert rewritten.rewritten_article == (
        "অবশেষে বহুল প্রত্যাশিত ব্যাংক ঋণের সুদের হার সিঙ্গেল ডিজিট ও "
        "সরল সুদ চালুর বাধা আরও বেড়েছে।"
    )


def test_artifact_detector_allows_valid_consultancy_word():
    assert "suspicious_bangla_fragment" not in artifact_reasons("ফ্রি ভিসা ও মেডিকেল কনসালটেন্সি সুবিধা পাবেন।")
    assert "suspicious_bangla_fragment" in artifact_reasons("ফ্রি ভিসা ও মেডিকেল কনসা।")


def test_numerical_plan_allows_strong_scale_replacements_and_rejects_count_percentages():
    claim = Claim(
        sentence_index=0,
        sentence="উদ্বোধন উপলক্ষে মোড়কজাত ৫৬ টাকার চিনির প্যাকেট সরবরাহ করা হয়।",
        claim_type="numerical",
        entities=[],
        numbers=["৫৬"],
        policies=[],
        dates=[],
        confidence=0.9,
    )
    bad_money_plan = RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Inflate the price.",
        edit_scope="target_sentence",
        expected_change="The packet price changes.",
        verification_constraints={},
        target_span="৫৬ টাকার",
        replacement="৫ কোটি টাকার",
    )
    money_to_percent_plan = RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Change the price unit incorrectly.",
        edit_scope="target_sentence",
        expected_change="The packet price changes.",
        verification_constraints={},
        target_span="৫৬ টাকার",
        replacement="১০০ শতাংশ",
    )
    good_money_plan = RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Inflate the price.",
        edit_scope="target_sentence",
        expected_change="The packet price changes.",
        verification_constraints={},
        target_span="৫৬ টাকার",
        replacement="৫৬০ টাকার",
    )
    count_claim = Claim(
        sentence_index=0,
        sentence="বিশ্বের ১শ’টির বেশি দেশে ব্যবসা পরিচালনাকারী কোম্পানি।",
        claim_type="numerical",
        entities=[],
        numbers=["১"],
        policies=[],
        dates=[],
        confidence=0.9,
    )
    good_count_plan = RewritePlan(
        family="numerical_fact",
        target_claim=count_claim,
        edit_instruction="Change the country count.",
        edit_scope="target_sentence",
        expected_change="The country count changes.",
        verification_constraints={},
        target_span="১শ’টির বেশি",
        replacement="২শ’টির বেশি",
    )
    stronger_count_plan = RewritePlan(
        family="numerical_fact",
        target_claim=count_claim,
        edit_instruction="Change the country count.",
        edit_scope="target_sentence",
        expected_change="The country count changes.",
        verification_constraints={},
        target_span="১শ’টির বেশি",
        replacement="৫শ’টির বেশি",
    )
    count_to_percent_plan = RewritePlan(
        family="numerical_fact",
        target_claim=count_claim,
        edit_instruction="Change the country count into a percentage.",
        edit_scope="target_sentence",
        expected_change="The country count changes.",
        verification_constraints={},
        target_span="১শ’টির বেশি",
        replacement="১০০ শতাংশ",
    )
    scaled_percent_plan = RewritePlan(
        family="numerical_fact",
        target_claim=count_claim,
        edit_instruction="Change the country count into an impossible scaled percentage.",
        edit_scope="target_sentence",
        expected_change="The country count changes.",
        verification_constraints={},
        target_span="১শ’টির বেশি",
        replacement="১ কোটি শতাংশ",
    )

    validate_rewrite_plan(bad_money_plan)
    validate_rewrite_plan(good_count_plan)
    validate_rewrite_plan(stronger_count_plan)
    validate_rewrite_plan(good_money_plan)
    with pytest.raises(ValueError, match="incompatible units"):
        validate_rewrite_plan(money_to_percent_plan)
    with pytest.raises(ValueError, match="count_percentage"):
        validate_rewrite_plan(count_to_percent_plan)
    with pytest.raises(ValueError, match="count_percentage"):
        validate_rewrite_plan(scaled_percent_plan)


def test_numeric_contradiction_verifier_accepts_strong_scale_rule_when_nli_is_low():
    article = Article(
        article_id="a1",
        headline="মাহিন্দ্রা ব্যবসা করে",
        text="বিশ্বের ১শ’টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।",
    )
    plan = numerical_plan(target_span="১শ’টির বেশি", replacement="২শ’টির বেশি")
    rewritten = "বিশ্বের ২শ’টির বেশি দেশে ব্যবসা পরিচালনাকারী প্রায় ২০ বিলিয়ন ডলারের কোম্পানি মাহিন্দ্রা।"

    result = ContradictionVerifier(LowContradictionNLI()).verify(article, rewritten, plan)

    assert result.passed is True
    assert result.details["rule"] == "deterministic_numeric_scale_contradiction"


def test_temporal_contradiction_verifier_accepts_strong_time_shift_when_nli_is_low():
    article = Article(
        article_id="a1",
        headline="গার্মেন্টস খাতে পরিবর্তন",
        text="তৈরি পোশাক খাতের প্রায় ২০ কারখানা বন্ধ হয়েছে চলতি জুলাই মাসে।",
    )
    claim = Claim(
        sentence_index=0,
        sentence="তৈরি পোশাক খাতের প্রায় ২০ কারখানা বন্ধ হয়েছে চলতি জুলাই মাসে।",
        claim_type="temporal",
        entities=[],
        numbers=[],
        policies=[],
        dates=["জুলাই"],
        confidence=0.9,
    )
    plan = RewritePlan(
        family="temporal_shift",
        target_claim=claim,
        edit_instruction="Shift the reporting period.",
        edit_scope="target_sentence",
        expected_change="The time anchor changes.",
        verification_constraints={},
        target_span="চলতি জুলাই মাসে",
        replacement="গত ডিসেম্বর মাসে",
    )
    rewritten = "তৈরি পোশাক খাতের প্রায় ২০ কারখানা বন্ধ হয়েছে গত ডিসেম্বর মাসে।"

    result = ContradictionVerifier(LowContradictionNLI()).verify(article, rewritten, plan)

    assert result.passed is True
    assert result.details["rule"] == "deterministic_temporal_shift_contradiction"


def test_target_artifact_verifier_ignores_unchanged_non_target_artifacts():
    article = Article(
        article_id="a1",
        headline="সিটি ব্যাংকের অনুদান",
        text="সিটি ব্যাংক ডিএমপিকে ৫০ লাখ টাকার অনুদান প্রদান করেছে। বাজার স্থিতিশীল বাজার স্থিতিশীল আছে।",
    )
    claim = Claim(
        sentence_index=0,
        sentence="সিটি ব্যাংক ডিএমপিকে ৫০ লাখ টাকার অনুদান প্রদান করেছে।",
        claim_type="numerical",
        entities=["সিটি ব্যাংক"],
        numbers=["৫০"],
        policies=[],
        dates=[],
        confidence=0.9,
    )
    plan = RewritePlan(
        family="numerical_fact",
        target_claim=claim,
        edit_instruction="Inflate the donation amount.",
        edit_scope="target_sentence",
        expected_change="The donation amount changes.",
        verification_constraints={},
        target_span="৫০ লাখ টাকার",
        replacement="৫ কোটি টাকার",
    )
    rewritten = "সিটি ব্যাংক ডিএমপিকে ৫ কোটি টাকার অনুদান প্রদান করেছে। বাজার স্থিতিশীল বাজার স্থিতিশীল আছে।"

    result = TextQualityArtifactVerifier().verify(article, rewritten, plan)

    assert result.passed is True


def test_entity_replacement_rejects_same_role_peer():
    claim = Claim(
        sentence_index=0,
        sentence="এডিবি করোনা মোকাবেলায় বাংলাদেশের জন্য ৩ লাখ ডলারের জরুরি সহায়তা অনুমোদন করেছে।",
        claim_type="entity",
        entities=["এডিবি"],
        numbers=["৩"],
        policies=[],
        dates=[],
        confidence=0.9,
    )
    article = Article(article_id="a1", headline="এডিবির সহায়তা", text=claim.sentence)
    plan = RewritePlan(
        family="entity_replacement",
        target_claim=claim,
        edit_instruction="Replace the entity.",
        edit_scope="target_sentence",
        expected_change="The aid provider changes.",
        verification_constraints={},
        target_span="এডিবি",
        replacement="বিশ্ব ব্যাংক",
    )
    rewritten = "বিশ্ব ব্যাংক করোনা মোকাবেলায় বাংলাদেশের জন্য ৩ লাখ ডলারের জরুরি সহায়তা অনুমোদন করেছে।"

    report = CompositeVerifier([IntendedChangeVerifier()]).verify(article, rewritten, plan)

    assert report.passed is False
    assert report.reasons == ["entity_same_role_replacement"]


def test_causal_inversion_accepts_opposite_effect_not_sentence_flip():
    claim = Claim(
        sentence_index=0,
        sentence="আমদানি ব্যয় বেড়ে যাওয়ার কারণে বৈদেশিক মুদ্রার রিজার্ভে চাপ সৃষ্টি হয়েছে।",
        claim_type="causal",
        entities=[],
        numbers=[],
        policies=[],
        dates=[],
        confidence=0.9,
    )
    article = Article(article_id="a1", headline="রিজার্ভে চাপ", text=claim.sentence)
    plan = RewritePlan(
        family="causal_inversion",
        target_claim=claim,
        edit_instruction="Invert the economic effect.",
        edit_scope="target_sentence",
        expected_change="Import cost growth is claimed to increase reserves.",
        verification_constraints={},
        target_span="রিজার্ভে চাপ সৃষ্টি হয়েছে",
        replacement="রিজার্ভ বেড়েছে",
    )
    rewritten = "আমদানি ব্যয় বেড়ে যাওয়ার কারণে বৈদেশিক মুদ্রার রিজার্ভ বেড়েছে।"

    report = CompositeVerifier([IntendedChangeVerifier()]).verify(article, rewritten, plan)

    assert report.passed is True


def test_expanded_entity_lexicon_covers_dataset_financial_entities():
    text = (
        "পূবালী ব্যাংক, বিএসইসি, বিজিএমইএ, এস আলম গ্রুপ, পিকেএসএফ "
        "এবং চট্টগ্রাম স্টক এক্সচেঞ্জ নতুন সিদ্ধান্ত জানিয়েছে। "
        "বিআইবিএম, বেক্সিমকো ফার্মা, এশীয় উন্নয়ন ব্যাংক, সাউথ বাংলা ব্যাংক, "
        "দুর্নীতি দমন কমিশন, বাংলাদেশ পর্যটন করপোরেশন, তিতাস গ্যাস, "
        "প্রগতি ইন্স্যুরেন্স এবং শিক্ষা মন্ত্রণালয়ও বিবৃতি দিয়েছে।"
    )
    entities = set(extract_entities(text))

    assert len(ENTITY_TERMS) >= 300
    assert {
        "পূবালী ব্যাংক",
        "বিএসইসি",
        "বিজিএমইএ",
        "এস আলম গ্রুপ",
        "পিকেএসএফ",
        "চট্টগ্রাম স্টক এক্সচেঞ্জ",
        "বিআইবিএম",
        "বেক্সিমকো ফার্মা",
        "এশীয় উন্নয়ন ব্যাংক",
        "সাউথ বাংলা ব্যাংক",
        "দুর্নীতি দমন কমিশন",
        "বাংলাদেশ পর্যটন করপোরেশন",
        "তিতাস গ্যাস",
        "প্রগতি ইন্স্যুরেন্স",
        "শিক্ষা মন্ত্রণালয়",
    } <= entities
    assert entities_are_same_role("সিটি ব্যাংক", "পূবালী ব্যাংক") is True
    assert entities_are_same_role("এডিবি", "বিএসইসি") is False


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
