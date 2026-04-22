from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def normalize_path(path_value: str) -> str:
    return str(Path(path_value).expanduser().resolve())


def load_config(config_path: Path) -> Dict[str, object]:
    default_config: Dict[str, object] = {"videos": {}, "last_calibration": None}
    if not config_path.exists():
        return default_config

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default_config

    if not isinstance(payload, dict):
        return default_config

    config_data = dict(payload)

    videos = config_data.get("videos", {})
    if not isinstance(videos, dict):
        videos = {}
    config_data["videos"] = videos

    if "last_calibration" not in config_data:
        config_data["last_calibration"] = None

    return config_data


def save_config(config_path: Path, config_data: Dict[str, object]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config_data, indent=2), encoding="utf-8")
