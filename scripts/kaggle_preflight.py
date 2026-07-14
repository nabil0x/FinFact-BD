#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSpec:
    role: str
    model_name: str
    revision: str = "main"
    backend: str = ""
    load_in_4bit: bool = False


def project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in [current.parent] + list(current.parents):
        if (parent / "src").is_dir() and (parent / "configs").is_dir():
            return parent
    raise RuntimeError("Could not locate project root")


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_specs(config: Dict[str, Any]) -> List[ModelSpec]:
    models = config["models"]
    specs: List[ModelSpec] = []
    extractor_cfg = dict(models.get("extractor", {}))
    if extractor_cfg.get("enabled", False):
        specs.append(_spec("extractor", extractor_cfg))
    planner_cfg = dict(models.get("planner", {}))
    if planner_cfg.get("share_with") != "extractor" and planner_cfg.get("enabled", False):
        specs.append(_spec("planner", planner_cfg))
    specs.append(_spec("generator", dict(models["generator"])))
    specs.append(_spec("embedding", dict(models["embedding"])))
    specs.append(_spec("nli", dict(models["nli"])))
    specs.append(_spec("fluency", dict(models["fluency"])))
    return specs


def _spec(role: str, cfg: Dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        role=role,
        model_name=str(cfg["model_name"]),
        revision=str(cfg.get("revision", "main")),
        backend=str(cfg.get("backend", "")),
        load_in_4bit=bool(cfg.get("load_in_4bit", False)),
    )


def unique_repos(specs: Iterable[ModelSpec]) -> List[ModelSpec]:
    seen: set[tuple[str, str]] = set()
    unique: List[ModelSpec] = []
    for spec in specs:
        key = (spec.model_name, spec.revision)
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def metadata_check(specs: List[ModelSpec], token: Optional[str]) -> None:
    from huggingface_hub import hf_hub_download, whoami

    failures = 0
    if token:
        account = whoami(token=token).get("name", "unknown")
        logger.info("HF account: %s", account)
    else:
        logger.warning("HF_TOKEN is not set; gated models will fail")

    for spec in unique_repos(specs):
        start = time.monotonic()
        try:
            path = hf_hub_download(
                repo_id=spec.model_name,
                filename="config.json",
                revision=spec.revision,
                token=token,
            )
        except Exception as exc:
            failures += 1
            logger.error("metadata failed role=%s repo=%s error=%s", spec.role, spec.model_name, exc)
            continue
        logger.info("metadata ok role=%s repo=%s path=%s elapsed=%.1fs", spec.role, spec.model_name, path, elapsed(start))
    if failures:
        raise RuntimeError(f"metadata preflight failed for {failures} model(s)")


def download_check(specs: List[ModelSpec], token: Optional[str]) -> None:
    from huggingface_hub import snapshot_download

    failures = 0
    allow_patterns = [
        "*.json",
        "*.txt",
        "*.model",
        "*.spm",
        "tokenizer*",
        "*.safetensors",
        "*.bin",
    ]
    ignore_patterns = [
        "*.h5",
        "*.msgpack",
        "*.onnx",
        "*.ot",
        "*.tflite",
        "flax_model*",
        "tf_model*",
    ]
    for spec in unique_repos(specs):
        start = time.monotonic()
        logger.info("download start role=%s repo=%s", spec.role, spec.model_name)
        try:
            path = snapshot_download(
                repo_id=spec.model_name,
                revision=spec.revision,
                token=token,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
        except Exception as exc:
            failures += 1
            logger.error("download failed role=%s repo=%s error=%s", spec.role, spec.model_name, exc)
            continue
        logger.info(
            "download ok role=%s repo=%s cache=%s size=%s elapsed=%.1fs",
            spec.role,
            spec.model_name,
            path,
            directory_size(Path(path)),
            elapsed(start),
        )
    if failures:
        raise RuntimeError(f"download preflight failed for {failures} model(s)")


def load_check(specs: List[ModelSpec], device: str) -> None:
    failures = 0
    for spec in specs:
        start = time.monotonic()
        logger.info("load start role=%s repo=%s backend=%s", spec.role, spec.model_name, spec.backend)
        log_gpu("before load")
        ok = True
        try:
            if spec.role in {"extractor", "planner", "generator"} or spec.backend == "hf_causal_lm":
                load_causal_lm(spec, device)
            elif spec.role == "embedding":
                load_embedding(spec, device)
            elif spec.role == "nli":
                load_nli(spec, device)
            elif spec.role == "fluency" and spec.backend == "hf_electra_discriminator":
                load_electra(spec, device)
            else:
                logger.warning("No load check implemented for role=%s backend=%s", spec.role, spec.backend)
        except Exception as exc:
            failures += 1
            ok = False
            logger.error("load failed role=%s repo=%s error=%s", spec.role, spec.model_name, exc)
        finally:
            cleanup_cuda()
        log_gpu("after release")
        if ok:
            logger.info("load ok role=%s repo=%s elapsed=%.1fs", spec.role, spec.model_name, elapsed(start))
    if failures:
        raise RuntimeError(f"load preflight failed for {failures} model(s)")


def load_causal_lm(spec: ModelSpec, device: str) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.model_name, revision=spec.revision, trust_remote_code=True)
    kwargs: Dict[str, Any] = {"revision": spec.revision, "trust_remote_code": True}
    if spec.load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"
    else:
        kwargs["torch_dtype"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(spec.model_name, **kwargs)
    if not spec.load_in_4bit:
        target = resolve_device(device)
        model = model.to(target)
    model.eval()
    logger.info("loaded causal lm role=%s pad_token=%s", spec.role, tokenizer.pad_token_id)
    del model, tokenizer


def load_embedding(spec: ModelSpec, device: str) -> None:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(spec.model_name, device=resolve_device(device))
    vectors = model.encode(["passage: preflight"], normalize_embeddings=True, show_progress_bar=False)
    logger.info("embedding ok dim=%d", len(vectors[0]))
    del model


def load_nli(spec: ModelSpec, device: str) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    target = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_name, revision=spec.revision)
    model = AutoModelForSequenceClassification.from_pretrained(spec.model_name, revision=spec.revision).to(target)
    model.eval()
    inputs = tokenizer("A claim changed.", "A claim did not change.", return_tensors="pt").to(target)
    with torch.no_grad():
        logits = model(**inputs).logits
    logger.info("nli ok logits_shape=%s", tuple(logits.shape))
    del model, tokenizer, inputs, logits


def load_electra(spec: ModelSpec, device: str) -> None:
    import torch
    from normalizer import normalize
    from transformers import AutoModelForPreTraining, AutoTokenizer

    target = resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(spec.model_name, revision=spec.revision)
    model = AutoModelForPreTraining.from_pretrained(spec.model_name, revision=spec.revision).to(target)
    model.eval()
    inputs = tokenizer(normalize("বাংলা ভাষার প্রিফ্লাইট পরীক্ষা।"), return_tensors="pt").to(target)
    with torch.no_grad():
        logits = model(**inputs).logits
    logger.info("electra ok logits_shape=%s", tuple(logits.shape))
    del model, tokenizer, inputs, logits


def cleanup_cuda() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        return


def resolve_device(device: str) -> str:
    if device != "cuda":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def log_gpu(label: str) -> None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            check=False,
            text=True,
            capture_output=True,
        )
    except FileNotFoundError:
        logger.info("gpu %s: nvidia-smi unavailable", label)
        return
    output = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    logger.info("gpu %s: %s", label, output or "unavailable")


def directory_size(path: Path) -> str:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return human_bytes(total)


def human_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024.0:
            return f"{value:.1f}{unit}"
        value /= 1024.0
    return f"{value:.1f}PB"


def elapsed(start: float) -> float:
    return time.monotonic() - start


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle model access, download, and load preflight.")
    parser.add_argument("--config", default="configs/rewrite_pipeline.yaml")
    parser.add_argument("--download", action="store_true", help="Download full model artifacts into the HF cache.")
    parser.add_argument("--load", action="store_true", help="Load each configured model sequentially.")
    parser.add_argument("--disable-xet", action="store_true", help="Set HF_HUB_DISABLE_XET=1 before downloads.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s [%(levelname)s] %(message)s")
    if args.disable_xet:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    root = project_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    specs = model_specs(load_yaml(config_path))
    token = os.environ.get("HF_TOKEN")

    logger.info("config=%s", config_path)
    logger.info("roles=%s", [(spec.role, spec.model_name) for spec in specs])
    try:
        metadata_check(specs, token)
        if args.download:
            download_check(specs, token)
        if args.load:
            load_check(specs, args.device)
    except RuntimeError as exc:
        logger.error(str(exc))
        raise SystemExit(1) from exc
    logger.info("preflight complete")


if __name__ == "__main__":
    main()
