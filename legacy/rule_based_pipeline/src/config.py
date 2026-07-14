#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def expand_config_refs(obj: Any, root: Dict[str, Any] | None = None) -> Any:
    if root is None:
        root = obj
    if isinstance(obj, str):
        if "${" in obj and "}" in obj:
            import re

            def replace_ref(match: re.Match[str]) -> str:
                value: Any = root
                for part in match.group(1).split("."):
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)
                return str(value)

            return re.sub(r"\$\{([^}]+)\}", replace_ref, obj)
        return obj
    if isinstance(obj, dict):
        return {key: expand_config_refs(value, root) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand_config_refs(value, root) for value in obj]
    return obj


def load_config(config_path: str | None = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as handle:
        return expand_config_refs(yaml.safe_load(handle))


_config: Dict[str, Any] | None = None


def get_config() -> Dict[str, Any]:
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_data_dir() -> Path:
    return Path(get_config()["paths"]["data_dir"])


def get_output_dir() -> Path:
    return Path(get_config()["paths"]["output_dir"])


def get_beni_v2_path() -> Path:
    return Path(get_config()["paths"]["beni_v2"])
