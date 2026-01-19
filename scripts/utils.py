from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


def load_config(path: str | Path = "scripts/config.yaml") -> Dict:
    # Load YAML configuration into a dict.
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def ensure_dirs(paths: Iterable[str | Path]) -> None:
    # Create directories if they do not exist.
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Dict) -> None:
    # Write a JSON payload to disk.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def list_files(root: str | Path, pattern: str) -> List[Path]:
    # Return sorted files matching a glob under root.
    root_path = Path(root)
    return sorted(root_path.glob(pattern))


def require_files(files: Iterable[Path]) -> Tuple[bool, List[str]]:
    # Check for missing files and return status plus list.
    missing = [str(f) for f in files if not f.exists()]
    return (len(missing) == 0, missing)
