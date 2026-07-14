from collections import Counter
import random
import json

from src.generation import perturbation_pipeline as p


class _MaxWeightRandom:
    def choices(self, population, weights, k):
        best = max(range(len(population)), key=lambda i: weights[i])
        return [population[best]]


def test_numerical_fact_change_changes_value():
    original = "বাংলাদেশ ব্যাংক সুদের হার ১০ শতাংশ থেকে ৮ শতাংশে কমিয়েছে।"
    out = p.fact_aware_numerical_fact_change(original, random.Random(42))

    assert out is not None
    assert out.text != original
    assert out.changed_span_original != out.changed_span_replacement
    assert out.proposition_schema["family"] == "numerical_fact_change"


def test_policy_reversal_flips_direction():
    original = "বাংলাদেশ ব্যাংক নীতিগত সুদের হার বাড়িয়েছে।"
    out = p.fact_aware_policy_reversal(original, random.Random(42))

    assert out is not None
    assert out.text != original
    assert "কমিয়েছে" in out.text or "বাড়িয়েছে" in out.text
    assert out.proposition_schema["family"] == "policy_reversal"


def test_entity_replacement_stays_in_class():
    original = "বাংলাদেশ ব্যাংক আজ নতুন নির্দেশনা দিয়েছে।"
    out = p.fact_aware_entity_replacement(original, random.Random(42))

    assert out is not None
    assert out.text != original
    assert out.ontology_class == "central_bank"
    assert out.changed_span_original == "বাংলাদেশ ব্যাংক"
    assert out.changed_span_replacement != out.changed_span_original


def test_temporal_shift_changes_date():
    original = "প্রতিবেদনটি ২০২৫ সালে প্রকাশিত হয়েছে।"
    out = p.fact_aware_temporal_shift(original, random.Random(42))

    assert out is not None
    assert out.text != original
    assert out.changed_span_role == "time"
    assert out.proposition_schema["family"] == "temporal_shift"


def test_causal_inversion_flips_outcome():
    original = "কর কমানোর ফলে বাজারে চাহিদা বেড়েছে।"
    out = p.fact_aware_causal_inversion(original, random.Random(42))

    assert out is not None
    assert out.text != original
    assert "কমেছে" in out.text
    assert out.proposition_schema["family"] == "causal_inversion"


def test_build_perturbation_plan_supports_multi_hop():
    text = (
        "বাংলাদেশ ব্যাংক জুন মাসে নীতিগত সুদের হার ১০ শতাংশে বাড়িয়েছে, "
        "ফলে ঋণের খরচ বেড়েছে এবং প্রতিবেদনটি ২০২৫ সালে প্রকাশিত হয়েছে।"
    )

    plan = p.build_perturbation_plan(text, "policy_reversal", random.Random(42), target_difficulty="hard")

    assert plan is not None
    assert plan.primary_family == "policy_reversal"
    assert plan.difficulty == "hard"
    assert plan.hop_count == 3
    assert len(plan.family_sequence) == 3
    assert plan.family_sequence[0] == "policy_reversal"
    assert len(plan.proposition_graph["policy_reversal"]) >= 1
    assert len(plan.proposition_graph["causal_inversion"]) >= 1
    assert len(plan.proposition_graph["temporal_shift"]) >= 1


def test_apply_perturbation_adds_multi_hop_metadata():
    article = {
        "article_id": "a1",
        "newspaper": "Test",
        "publication_date": "2025-01-01",
        "headline": "headline",
        "text": (
            "বাংলাদেশ ব্যাংক জুন মাসে নীতিগত সুদের হার ১০ শতাংশে বাড়িয়েছে, "
            "ফলে ঋণের খরচ বেড়েছে এবং প্রতিবেদনটি ২০২৫ সালে প্রকাশিত হয়েছে।"
        ),
        "industry_sector": "banking",
    }

    out = p.apply_perturbation(article, "policy_reversal", random.Random(42), target_difficulty="hard")

    assert out is not None
    assert out["perturbation_type"] == "policy_reversal"
    assert out["difficulty"] == "hard"
    assert out["hop_count"] == 3
    assert out["perturbation_mode"] == "multi_hop"
    assert out["validation_passed"] is True
    assert out["proposition_schema"]
    assert out["perturbation_plan"]

    validation = json.loads(out["validation_scores"])
    assert validation["passed"] is True
    assert validation["contradiction_score"] >= 0.55

    plan = json.loads(out["perturbation_plan"])
    assert len(plan["operations"]) == 3
    assert plan["validation"]["passed"] is True
    assert p.basic_quality_filter(out)


def test_generate_dataset_balances_all_families():
    article_text = (
        "বাংলাদেশ ব্যাংক জুন মাসে নীতিগত সুদের হার ১০ শতাংশে বাড়িয়েছে, "
        "ফলে ঋণের খরচ বেড়েছে এবং প্রতিবেদনটি ২০২৫ সালে প্রকাশিত হয়েছে।"
    )
    articles = [
        {
            "article_id": f"a{i}",
            "newspaper": "Test",
            "publication_date": "2025-01-01",
            "headline": f"headline {i}",
            "text": article_text,
            "industry_sector": "banking",
        }
        for i in range(5)
    ]

    original_target = p.TARGET_PER_TYPE
    try:
        p.TARGET_PER_TYPE = 1
        originals, perturbed = p.generate_dataset(articles)
    finally:
        p.TARGET_PER_TYPE = original_target

    assert len(originals) == 5
    assert len(perturbed) == 5
    assert {row["perturbation_type"] for row in perturbed} == set(p.FACT_AWARE_PERTURBATION_TYPES)
    assert all(row["validation_passed"] is True for row in perturbed)
    assert all(int(row["hop_count"]) >= 1 for row in perturbed)


def test_assign_splits_keeps_original_ids_together():
    rows = [
        {"article_id": "a1", "original_id": "a1", "label": 0},
        {"article_id": "a1_pert", "original_id": "a1", "label": 1},
        {"article_id": "a2", "original_id": "a2", "label": 0},
        {"article_id": "a2_pert", "original_id": "a2", "label": 1},
        {"article_id": "a3", "original_id": "a3", "label": 0},
    ]

    assigned = p.assign_splits_by_original_id(rows, seed=42)
    by_original = {}
    for row in assigned:
        by_original.setdefault(row["original_id"], set()).add(row["split"])

    assert all(len(splits) == 1 for splits in by_original.values())
    audit = p.audit_original_id_split_leakage(assigned)
    assert audit["has_leakage"] is False
    assert audit["leaky_original_ids"] == 0
    assert sum(audit["split_row_counts"].values()) == len(rows)


def test_audit_detects_manual_split_leakage():
    rows = [
        {"article_id": "a1", "original_id": "a1", "split": "train"},
        {"article_id": "a1_pert", "original_id": "a1", "split": "test"},
    ]

    audit = p.audit_original_id_split_leakage(rows)
    assert audit["has_leakage"] is True
    assert audit["leaky_original_ids"] == 1


def test_family_difficulty_policy_prefers_expected_band():
    rng = _MaxWeightRandom()
    assert p._select_difficulty_for_family("policy_reversal", rng) == "easy"
    assert p._select_difficulty_for_family("numerical_fact_change", rng) == "medium"
    assert p._select_difficulty_for_family("causal_inversion", rng) == "hard"


def test_balanced_difficulty_policy_tracks_global_budget():
    rng = _MaxWeightRandom()
    difficulty_counts = Counter({"easy": 9, "medium": 6, "hard": 12})
    difficulty_targets = {"easy": 9.0, "medium": 12.0, "hard": 9.0}

    assert (
        p._select_balanced_difficulty_for_family(
            "causal_inversion",
            rng,
            difficulty_counts,
            difficulty_targets,
        )
        == "medium"
    )


def test_basic_quality_filter_honors_difficulty_thresholds():
    base_row = {
        "text": "বাংলাদেশ ব্যাংক নীতিগত সুদের হার কমিয়েছে এবং বাজারে নতুন নির্দেশনা দিয়েছে।",
        "original_text": "বাংলাদেশ ব্যাংক নীতিগত সুদের হার বাড়িয়েছে এবং বাজারে নতুন নির্দেশনা দিয়েছে।",
        "perturbation_type": "policy_reversal",
        "changed_span_original": "বাড়িয়েছে",
        "changed_span_replacement": "কমিয়েছে",
        "proposition_schema": "{}",
        "perturbation_plan": "{}",
        "validation_scores": json.dumps({
            "passed": True,
            "contradiction_score": 0.44,
            "semantic_similarity": 0.86,
            "fluency_score": 0.95,
            "issues": [],
        }),
    }

    high_similarity_easy_row = dict(base_row, difficulty="easy")
    easy_row = dict(base_row, difficulty="easy")
    hard_row = dict(base_row, difficulty="hard")

    high_similarity_easy_row["validation_scores"] = json.dumps({
        "passed": True,
        "contradiction_score": 0.44,
        "semantic_similarity": 0.995,
        "fluency_score": 0.95,
        "issues": [],
    })

    assert p.basic_quality_filter(high_similarity_easy_row) is True
    assert p.basic_quality_filter(easy_row) is True
    assert p.basic_quality_filter(hard_row) is False
