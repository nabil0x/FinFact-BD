#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from kaggle_output_inspector import inspect_output
from kaggle_failure_audit import audit_failures
from kaggle_metrics import summarize_metrics

DEFAULT_CONFIG = "configs/rewrite_pipeline.yaml"
DEFAULT_INPUT = "data/finfact_bd/finfact_bd_originals.csv"
LOG_DIR = Path("logs")

OUTPUT_DIRS = {
    "smoke": "data/generated/rewrite_generation_smoke",
    "stress1k": "data/generated/rewrite_generation_stress1k",
    "pilot": "data/generated/rewrite_generation_pilot",
    "full": "data/generated/rewrite_generation_full",
}

logger = logging.getLogger(__name__)

PLANNER_PRESETS = {
    "qwen3-8b": {
        "model_name": "Qwen/Qwen3-8B",
        "revision": "main",
        "load_in_4bit": True,
        "use_chat_template": True,
        "chat_template_kwargs": {"enable_thinking": False},
    },
    "qwen25-3b": {
        "model_name": "Qwen/Qwen2.5-3B-Instruct",
        "revision": "main",
        "load_in_4bit": True,
        "use_chat_template": True,
        "chat_template_kwargs": {},
    },
}


def _any_non_none(*values: object) -> bool:
    return any(value is not None for value in values)


def _apply_updates(target: dict[str, Any], **updates: Any) -> bool:
    changed = False
    for key, value in updates.items():
        if value is not None:
            target[key] = value
            changed = True
    return changed


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Could not locate project root")


def configure_runtime(disable_xet: bool = True) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if Path("/kaggle").exists():
        os.environ.setdefault("HF_HOME", "/kaggle/temp/huggingface")
    if disable_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    if os.environ.get("HF_TOKEN"):
        logger.info("HF_TOKEN already set")
        return
    try:
        from kaggle_secrets import UserSecretsClient
    except ImportError:
        logger.warning("HF_TOKEN is not set and kaggle_secrets is unavailable")
        return
    try:
        os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
        logger.info("Loaded HF_TOKEN from Kaggle Secrets")
    except Exception as exc:
        logger.warning("Could not read HF_TOKEN from Kaggle Secrets: %s", exc)


def run_command(cmd: list[str], log_path: Optional[Path] = None, append: bool = False) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Running: %s", " ".join(cmd))
    handle = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(log_path, "a" if append else "w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if handle is not None:
                handle.write(line)
                handle.flush()
        code = process.wait()
    finally:
        if handle is not None:
            handle.close()
    if code != 0:
        raise SystemExit(f"Command failed with exit code {code}: {' '.join(cmd)}")


def python_cmd(*args: str) -> list[str]:
    return [sys.executable, *args]


def git_pull() -> None:
    if not Path(".git").exists():
        logger.info("Skipping git pull because .git is absent")
        return
    run_command(["git", "pull", "--ff-only"], LOG_DIR / "git_pull.log")


def setup(args: argparse.Namespace) -> None:
    if not args.no_pull:
        git_pull()
    if not args.no_pip_upgrade:
        run_command(python_cmd("-m", "pip", "install", "--upgrade", "pip"), LOG_DIR / "pip_upgrade.log")
    run_command(python_cmd("-m", "pip", "install", "-r", "requirements.txt"), LOG_DIR / "pip_install.log")


def check(_: argparse.Namespace) -> None:
    run_command(python_cmd("-m", "compileall", "-q", "src", "scripts", "tests"), LOG_DIR / "compileall.log")
    run_command(python_cmd("-m", "pytest", "-q"), LOG_DIR / "pytest.log")
    validate_config(DEFAULT_CONFIG)


def validate_config(path: str) -> None:
    import yaml

    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    required = [
        ("paths", "input_csv"),
        ("paths", "output_dir"),
        ("input", "id_column"),
        ("models", "extractor"),
        ("models", "planner"),
        ("models", "generator"),
        ("models", "embedding"),
        ("models", "nli"),
        ("models", "fluency"),
    ]
    missing = [f"{section}.{key}" for section, key in required if section not in cfg or key not in cfg[section]]
    if missing:
        raise SystemExit(f"Missing config keys: {missing}")
    logger.info("Config OK: %s", path)


def add_model_override_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--planner-preset", choices=sorted(PLANNER_PRESETS))
    parser.add_argument("--planner-model")
    parser.add_argument("--planner-revision")
    parser.add_argument("--planner-load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--planner-use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--extractor-enabled", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--extractor-model")
    parser.add_argument("--extractor-revision")
    parser.add_argument("--extractor-load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--extractor-use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--generator-model")
    parser.add_argument("--generator-revision")
    parser.add_argument("--generator-load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--generator-use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--embedding-model")
    parser.add_argument("--embedding-prefix")
    parser.add_argument("--nli-model")
    parser.add_argument("--nli-revision")
    parser.add_argument("--fluency-model")
    parser.add_argument("--fluency-revision")
    parser.add_argument("--fluency-backend")
    parser.add_argument("--plan-repair-attempts", type=int)
    parser.add_argument("--plan-review-attempts", type=int)


def resolved_pipeline_config(args: argparse.Namespace, mode: str, output_dir: Path) -> str:
    planner_preset = getattr(args, "planner_preset", None)
    planner_model = getattr(args, "planner_model", None)
    planner_revision = getattr(args, "planner_revision", None)
    planner_load_in_4bit = getattr(args, "planner_load_in_4bit", None)
    planner_use_chat_template = getattr(args, "planner_use_chat_template", None)
    extractor_enabled = getattr(args, "extractor_enabled", None)
    extractor_model = getattr(args, "extractor_model", None)
    extractor_revision = getattr(args, "extractor_revision", None)
    extractor_load_in_4bit = getattr(args, "extractor_load_in_4bit", None)
    extractor_use_chat_template = getattr(args, "extractor_use_chat_template", None)
    generator_model = getattr(args, "generator_model", None)
    generator_revision = getattr(args, "generator_revision", None)
    generator_load_in_4bit = getattr(args, "generator_load_in_4bit", None)
    generator_use_chat_template = getattr(args, "generator_use_chat_template", None)
    embedding_model = getattr(args, "embedding_model", None)
    embedding_prefix = getattr(args, "embedding_prefix", None)
    nli_model = getattr(args, "nli_model", None)
    nli_revision = getattr(args, "nli_revision", None)
    fluency_model = getattr(args, "fluency_model", None)
    fluency_revision = getattr(args, "fluency_revision", None)
    fluency_backend = getattr(args, "fluency_backend", None)
    plan_repair_attempts = getattr(args, "plan_repair_attempts", None)
    plan_review_attempts = getattr(args, "plan_review_attempts", None)
    if not _any_non_none(
        planner_preset,
        planner_model,
        planner_revision,
        planner_load_in_4bit,
        planner_use_chat_template,
        extractor_enabled,
        extractor_model,
        extractor_revision,
        extractor_load_in_4bit,
        extractor_use_chat_template,
        generator_model,
        generator_revision,
        generator_load_in_4bit,
        generator_use_chat_template,
        embedding_model,
        embedding_prefix,
        nli_model,
        nli_revision,
        fluency_model,
        fluency_revision,
        fluency_backend,
        plan_repair_attempts,
        plan_review_attempts,
    ):
        return args.config

    import yaml

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root() / config_path
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    models = config.setdefault("models", {})
    changed_roles: list[str] = []

    planner_model_config = models.setdefault("planner", {})
    planner_touched = False
    if planner_preset:
        planner_model_config.update(PLANNER_PRESETS[planner_preset])
        planner_touched = True
    planner_touched |= _apply_updates(
        planner_model_config,
        model_name=planner_model,
        revision=planner_revision,
        load_in_4bit=planner_load_in_4bit,
        use_chat_template=planner_use_chat_template,
    )
    if planner_touched:
        changed_roles.append(f"planner={planner_model_config.get('model_name')}")

    extractor_model_config = models.setdefault("extractor", {})
    extractor_touched = False
    if extractor_enabled is not None:
        extractor_model_config["enabled"] = extractor_enabled
        extractor_touched = True
    if _any_non_none(extractor_model, extractor_revision, extractor_load_in_4bit, extractor_use_chat_template):
        if extractor_enabled is None:
            extractor_model_config["enabled"] = True
        extractor_touched |= _apply_updates(
            extractor_model_config,
            model_name=extractor_model,
            revision=extractor_revision,
            load_in_4bit=extractor_load_in_4bit,
            use_chat_template=extractor_use_chat_template,
        )
    if extractor_touched:
        changed_roles.append(f"extractor={extractor_model_config.get('model_name', 'heuristic')}")

    generator_model_config = models.setdefault("generator", {})
    if _apply_updates(
        generator_model_config,
        model_name=generator_model,
        revision=generator_revision,
        load_in_4bit=generator_load_in_4bit,
        use_chat_template=generator_use_chat_template,
    ):
        changed_roles.append(f"generator={generator_model_config.get('model_name')}")

    embedding_model_config = models.setdefault("embedding", {})
    if _apply_updates(embedding_model_config, model_name=embedding_model, prefix=embedding_prefix):
        changed_roles.append(f"embedding={embedding_model_config.get('model_name')}")

    nli_model_config = models.setdefault("nli", {})
    if _apply_updates(nli_model_config, model_name=nli_model, revision=nli_revision):
        changed_roles.append(f"nli={nli_model_config.get('model_name')}")

    fluency_model_config = models.setdefault("fluency", {})
    if _apply_updates(
        fluency_model_config,
        model_name=fluency_model,
        revision=fluency_revision,
        backend=fluency_backend,
    ):
        changed_roles.append(f"fluency={fluency_model_config.get('model_name')}")

    planner_config = config.setdefault("planner", {})
    planner_rules_touched = False
    if plan_repair_attempts is not None:
        planner_config["plan_repair_attempts"] = plan_repair_attempts
        planner_rules_touched = True
    if plan_review_attempts is not None:
        planner_config["plan_review_attempts"] = plan_review_attempts
        planner_rules_touched = True
    if planner_rules_touched:
        changed_roles.append(
            "planner_rules="
            f"{planner_config.get('plan_repair_attempts')}/{planner_config.get('plan_review_attempts')}"
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe_mode = mode.replace("/", "_")
    safe_output = output_dir.name or "output"
    resolved_text = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    resolved_hash = hashlib.sha1(resolved_text.encode("utf-8")).hexdigest()[:8]
    resolved_path = LOG_DIR / f"resolved_{safe_mode}_{safe_output}_{resolved_hash}_config.yaml"
    resolved_path.write_text(resolved_text, encoding="utf-8")
    logger.info(
        "Using resolved config %s overrides=%s",
        resolved_path,
        ", ".join(changed_roles) if changed_roles else "none",
    )
    return str(resolved_path)


def gpu(_: argparse.Namespace) -> None:
    run_command(["nvidia-smi"], LOG_DIR / "nvidia_smi.log")
    run_command(
        python_cmd(
            "-c",
            "import torch; print('torch:', torch.__version__); "
            "print('cuda:', torch.cuda.is_available()); "
            "print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')",
        ),
        LOG_DIR / "torch_cuda.log",
    )


def preflight(args: argparse.Namespace) -> None:
    configure_runtime(disable_xet=not args.enable_xet)
    config_path = resolved_pipeline_config(args, "preflight", Path("preflight"))
    stages = ["metadata", "download", "load"] if args.stage == "all" else [args.stage]
    for stage in stages:
        cmd = python_cmd("scripts/kaggle_preflight.py", "--config", config_path, "--log-level", args.log_level)
        if stage == "download":
            cmd.append("--download")
        if stage == "load":
            cmd.append("--load")
        if not args.enable_xet and stage in {"download", "load"}:
            cmd.append("--disable-xet")
        run_command(cmd, LOG_DIR / f"model_{stage}_preflight.log")


def pipeline(args: argparse.Namespace, mode: str) -> None:
    configure_runtime(disable_xet=not args.enable_xet)
    output_dir = Path(args.output_dir or OUTPUT_DIRS[mode])
    if args.clean:
        shutil.rmtree(output_dir, ignore_errors=True)
    config_path = resolved_pipeline_config(args, mode, output_dir)
    cmd = python_cmd(
        "scripts/run_rewrite_pipeline.py",
        "--config",
        config_path,
        "--input",
        args.input,
        "--output-dir",
        str(output_dir),
        "--seed",
        str(args.seed),
        "--log-level",
        args.log_level,
    )
    if args.num_samples is not None:
        cmd.extend(["--num-samples", str(args.num_samples)])
    run_command(cmd, LOG_DIR / f"rewrite_{mode}.log", append=args.append_log)


def smoke(args: argparse.Namespace) -> None:
    pipeline(args, "smoke")


def pilot(args: argparse.Namespace) -> None:
    pipeline(args, "pilot")


def stress1k(args: argparse.Namespace) -> None:
    pipeline(args, "stress1k")


def full(args: argparse.Namespace) -> None:
    pipeline(args, "full")


def resume(args: argparse.Namespace) -> None:
    if args.clean:
        raise SystemExit("resume does not support --clean; use full --clean to intentionally restart")
    pipeline(args, "full")


def inspect(args: argparse.Namespace) -> None:
    inspect_output(Path(args.output_dir), args.preview, args.fast, args.skip_workbook)


def metrics(args: argparse.Namespace) -> None:
    summary = summarize_metrics(Path(args.output_dir), Path(args.log) if args.log else None)
    if args.write:
        path = Path(args.output_dir) / "metrics_summary.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def failure_audit(args: argparse.Namespace) -> None:
    summary = audit_failures(Path(args.output_dir), args.preview)
    if args.write:
        path = Path(args.output_dir) / "failure_audit.json"
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def all_smoke(args: argparse.Namespace) -> None:
    if not args.skip_setup:
        setup(argparse.Namespace(no_pull=args.no_pull, no_pip_upgrade=args.no_pip_upgrade))
    check(args)
    preflight_kwargs = dict(vars(args))
    preflight_kwargs["stage"] = "all"
    preflight(argparse.Namespace(**preflight_kwargs))
    smoke_kwargs = dict(vars(args))
    smoke_kwargs.update(
        {
            "output_dir": OUTPUT_DIRS["smoke"],
            "clean": True,
            "append_log": False,
        }
    )
    smoke(argparse.Namespace(**smoke_kwargs))
    inspect(argparse.Namespace(output_dir=OUTPUT_DIRS["smoke"], preview=3, fast=True, skip_workbook=False))


def add_pipeline_args(parser: argparse.ArgumentParser, default_output: str, default_samples: Optional[int]) -> None:
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument("--num-samples", type=int, default=default_samples)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--append-log", action="store_true")
    parser.add_argument("--enable-xet", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    add_model_override_args(parser)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle workflow runner for FinFact-BD.")
    sub = parser.add_subparsers(dest="command", required=True)

    setup_parser = sub.add_parser("setup", help="Pull repo and install requirements.")
    setup_parser.add_argument("--no-pull", action="store_true")
    setup_parser.add_argument("--no-pip-upgrade", action="store_true")
    setup_parser.set_defaults(func=setup)

    check_parser = sub.add_parser("check", help="Run compile, tests, and config validation.")
    check_parser.set_defaults(func=check)

    gpu_parser = sub.add_parser("gpu", help="Print GPU and torch CUDA status.")
    gpu_parser.set_defaults(func=gpu)

    preflight_parser = sub.add_parser("preflight", help="Run model metadata/download/load checks.")
    preflight_parser.add_argument("--stage", default="all", choices=["metadata", "download", "load", "all"])
    preflight_parser.add_argument("--config", default=DEFAULT_CONFIG)
    preflight_parser.add_argument("--enable-xet", action="store_true")
    preflight_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    add_model_override_args(preflight_parser)
    preflight_parser.set_defaults(func=preflight)

    smoke_parser = sub.add_parser("smoke", help="Run clean five-sample smoke generation.")
    add_pipeline_args(smoke_parser, OUTPUT_DIRS["smoke"], 5)
    smoke_parser.set_defaults(func=smoke, clean=True)

    pilot_parser = sub.add_parser("pilot", help="Run pilot generation.")
    add_pipeline_args(pilot_parser, OUTPUT_DIRS["pilot"], 100)
    pilot_parser.set_defaults(func=pilot, clean=True)

    stress_parser = sub.add_parser("stress1k", help="Run clean 1k stress generation.")
    add_pipeline_args(stress_parser, OUTPUT_DIRS["stress1k"], 1000)
    stress_parser.set_defaults(func=stress1k, clean=True)

    full_parser = sub.add_parser("full", help="Run full generation. Does not clean by default.")
    add_pipeline_args(full_parser, OUTPUT_DIRS["full"], None)
    full_parser.set_defaults(func=full)

    resume_parser = sub.add_parser("resume", help="Resume full generation from checkpoint.")
    add_pipeline_args(resume_parser, OUTPUT_DIRS["full"], None)
    resume_parser.set_defaults(func=resume, append_log=True, clean=False)

    inspect_parser = sub.add_parser("inspect", help="Inspect exported dataset/checkpoint/workbook.")
    inspect_parser.add_argument("--output-dir", default=OUTPUT_DIRS["smoke"])
    inspect_parser.add_argument("--preview", type=int, default=3)
    inspect_parser.add_argument("--fast", action="store_true", help="Skip workbook parsing and count CSV rows by line count.")
    inspect_parser.add_argument("--skip-workbook", action="store_true", help="Skip human_validation.xlsx preview.")
    inspect_parser.set_defaults(func=inspect)

    metrics_parser = sub.add_parser("metrics", help="Summarize throughput, retry, memory, and verifier timing metrics.")
    metrics_parser.add_argument("--output-dir", default=OUTPUT_DIRS["smoke"])
    metrics_parser.add_argument("--log")
    metrics_parser.add_argument("--write", action="store_true")
    metrics_parser.set_defaults(func=metrics)

    audit_parser = sub.add_parser("failure-audit", help="Audit failed rewrite attempts and verifier scores.")
    audit_parser.add_argument("--output-dir", default=OUTPUT_DIRS["smoke"])
    audit_parser.add_argument("--preview", type=int, default=10)
    audit_parser.add_argument("--write", action="store_true")
    audit_parser.set_defaults(func=failure_audit)

    all_parser = sub.add_parser("all-smoke", help="Setup, test, preflight, smoke, and inspect.")
    all_parser.add_argument("--skip-setup", action="store_true")
    all_parser.add_argument("--no-pull", action="store_true")
    all_parser.add_argument("--no-pip-upgrade", action="store_true")
    all_parser.add_argument("--config", default=DEFAULT_CONFIG)
    all_parser.add_argument("--input", default=DEFAULT_INPUT)
    all_parser.add_argument("--num-samples", type=int, default=5)
    all_parser.add_argument("--seed", type=int, default=42)
    all_parser.add_argument("--enable-xet", action="store_true")
    all_parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    add_model_override_args(all_parser)
    all_parser.set_defaults(func=all_smoke)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.chdir(project_root())
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
