#!/usr/bin/env python3
"""
FinFact-BD Configuration Loader
Loads configuration from YAML file with environment variable expansion.
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict

# Default config path
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "default.yaml"


def expand_config_refs(obj: Any, root: Dict[str, Any] = None) -> Any:
    """Recursively expand ${section.key} references in config values."""
    if root is None:
        root = obj
    
    if isinstance(obj, str):
        # Expand ${section.key} patterns
        if "${" in obj and "}" in obj:
            import re
            def replace_ref(match):
                ref_path = match.group(1)
                parts = ref_path.split(".")
                value = root
                for part in parts:
                    if isinstance(value, dict) and part in value:
                        value = value[part]
                    else:
                        return match.group(0)  # Return original if not found
                return str(value)
            
            obj = re.sub(r'\$\{([^}]+)\}', replace_ref, obj)
        return obj
    elif isinstance(obj, dict):
        return {k: expand_config_refs(v, root) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [expand_config_refs(item, root) for item in obj]
    return obj


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Expand config references (e.g., ${paths.data_dir})
    config = expand_config_refs(config)
    
    # Convert string paths to Path objects
    if 'paths' in config:
        for key, value in config['paths'].items():
            if isinstance(value, str):
                config['paths'][key] = Path(value)
    
    return config


# Global config instance
_config = None


def get_config() -> Dict[str, Any]:
    """Get global config instance (lazy loading)."""
    global _config
    if _config is None:
        _config = load_config()
    return _config


# Convenience accessors
def get_data_dir() -> Path:
    """Get data directory path."""
    return Path(get_config()['paths']['data_dir'])


def get_output_dir() -> Path:
    """Get output directory path."""
    return Path(get_config()['paths']['output_dir'])


def get_beni_v2_path() -> Path:
    """Get BENI v2 dataset path."""
    return Path(get_config()['paths']['beni_v2'])
