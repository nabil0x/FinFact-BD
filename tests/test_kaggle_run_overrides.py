from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import kaggle_run  # noqa: E402


def test_resolved_pipeline_config_applies_role_model_overrides(tmp_path, monkeypatch):
    monkeypatch.setattr(kaggle_run, "LOG_DIR", tmp_path / "logs")
    args = Namespace(
        config="configs/rewrite_pipeline.yaml",
        planner_preset="qwen25-3b",
        planner_model=None,
        planner_revision="test-revision",
        planner_load_in_4bit=False,
        planner_use_chat_template=False,
        extractor_enabled=None,
        extractor_model="Qwen/Qwen2.5-3B-Instruct",
        extractor_revision="main",
        extractor_load_in_4bit=False,
        extractor_use_chat_template=True,
        generator_model="CohereLabs/aya-expanse-8b",
        generator_revision="branch-a",
        generator_load_in_4bit=True,
        generator_use_chat_template=False,
        embedding_model="intfloat/multilingual-e5-base",
        embedding_prefix="passage: ",
        nli_model="MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7",
        nli_revision="branch-b",
        fluency_model="csebuetnlp/banglabert",
        fluency_revision="dev",
        fluency_backend="hf_electra_discriminator",
        plan_repair_attempts=3,
        plan_review_attempts=2,
    )

    resolved = Path(kaggle_run.resolved_pipeline_config(args, "pilot", tmp_path / "rewrite_generation_pilot10"))
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))

    assert resolved.parent == tmp_path / "logs"
    assert data["models"]["planner"]["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert data["models"]["planner"]["revision"] == "test-revision"
    assert data["models"]["planner"]["load_in_4bit"] is False
    assert data["models"]["planner"]["use_chat_template"] is False
    assert data["models"]["extractor"]["enabled"] is True
    assert data["models"]["extractor"]["model_name"] == "Qwen/Qwen2.5-3B-Instruct"
    assert data["models"]["extractor"]["load_in_4bit"] is False
    assert data["models"]["generator"]["model_name"] == "CohereLabs/aya-expanse-8b"
    assert data["models"]["generator"]["revision"] == "branch-a"
    assert data["models"]["generator"]["load_in_4bit"] is True
    assert data["models"]["generator"]["use_chat_template"] is False
    assert data["models"]["embedding"]["model_name"] == "intfloat/multilingual-e5-base"
    assert data["models"]["embedding"]["prefix"] == "passage: "
    assert data["models"]["nli"]["model_name"] == "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    assert data["models"]["nli"]["revision"] == "branch-b"
    assert data["models"]["fluency"]["model_name"] == "csebuetnlp/banglabert"
    assert data["models"]["fluency"]["revision"] == "dev"
    assert data["models"]["fluency"]["backend"] == "hf_electra_discriminator"
    assert data["planner"]["plan_repair_attempts"] == 3
    assert data["planner"]["plan_review_attempts"] == 2
