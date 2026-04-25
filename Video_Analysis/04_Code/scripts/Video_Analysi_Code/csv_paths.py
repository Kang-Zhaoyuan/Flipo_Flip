from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from Video_Analysi_Code.path_registry import (
    plots_dir,
    processed_data_dir,
    raw_data_dir,
    resolve_config_path,
)


RAW_CSV_SUFFIX = "_raw_data"


def load_last_calibration_video_path(config_path: Path) -> Optional[Path]:
    if not config_path.exists():
        return None

    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(payload, dict):
        return None

    last_calibration = payload.get("last_calibration")
    if not isinstance(last_calibration, dict):
        return None

    raw_video_path = last_calibration.get("video_path")
    if not isinstance(raw_video_path, str) or not raw_video_path.strip():
        return None

    return Path(raw_video_path).expanduser()


def resolve_video_path(project_root: Path, explicit_video_path: Optional[str]) -> Path:
    if explicit_video_path:
        explicit_path = Path(explicit_video_path).expanduser()
        if explicit_path.exists() and explicit_path.is_file():
            return explicit_path.resolve()
        raise FileNotFoundError(
            f"Provided --video-path does not exist or is not a file: {explicit_path}"
        )

    config_path = resolve_config_path(project_root)
    fallback = load_last_calibration_video_path(config_path)
    if fallback is None:
        raise FileNotFoundError(
            "Cannot resolve source video path. Pass --video-path or populate "
            "last_calibration.video_path in config."
        )

    if fallback.exists() and fallback.is_file():
        return fallback.resolve()

    raise FileNotFoundError(
        f"Configured last_calibration video path is unavailable: {fallback}"
    )


def resolve_raw_csv_from_video(project_root: Path, video_path: Path) -> Tuple[Path, bool]:
    raw_csv_stem = f"{video_path.stem}{RAW_CSV_SUFFIX}"

    preferred_csv = raw_data_dir(project_root) / f"{raw_csv_stem}.csv"
    if preferred_csv.exists() and preferred_csv.is_file():
        return preferred_csv.resolve(), False

    legacy_csv = project_root / f"{raw_csv_stem}.csv"
    if legacy_csv.exists() and legacy_csv.is_file():
        return legacy_csv.resolve(), True

    raise FileNotFoundError(
        "CSV not found for source video. Tried: "
        f"{preferred_csv} and {legacy_csv}"
    )


def resolve_csv_path(
    project_root: Path,
    explicit_csv_path: Optional[str],
    explicit_video_path: Optional[str],
) -> Tuple[Path, Optional[Path], bool, bool]:
    if explicit_csv_path:
        csv_path = Path(explicit_csv_path).expanduser()
        if csv_path.exists() and csv_path.is_file():
            return csv_path.resolve(), None, False, True
        raise FileNotFoundError(
            f"Provided --csv-path does not exist or is not a file: {csv_path}"
        )

    video_path = resolve_video_path(project_root, explicit_video_path)
    csv_path, used_legacy_raw_dir = resolve_raw_csv_from_video(project_root, video_path)
    return csv_path, video_path, used_legacy_raw_dir, False


def raw_csv_output_path(project_root: Path, video_path: Path) -> Path:
    raw_csv_stem = f"{video_path.stem}{RAW_CSV_SUFFIX}"

    target_dir = raw_data_dir(project_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / f"{raw_csv_stem}.csv"


def default_processed_output_dir(project_root: Path) -> Path:
    target_dir = processed_data_dir(project_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def default_plot_output_path(project_root: Path, filename: str) -> Path:
    target_dir = plots_dir(project_root)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename
