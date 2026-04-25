from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


_NEW_LAYOUT_MARKERS = (
    "01_Data",
    "02_Videos",
    "03_Results",
    "04_Code",
    "05_Metadata",
)


def _iter_candidate_roots(anchor: Path) -> Iterable[Path]:
    yield anchor
    yield from anchor.parents


def find_project_root(start: Optional[Path] = None) -> Path:
    anchor = (start or Path(__file__)).resolve()
    if anchor.is_file():
        anchor = anchor.parent

    for candidate in _iter_candidate_roots(anchor):
        if all((candidate / marker).exists() for marker in _NEW_LAYOUT_MARKERS):
            return candidate

    # Backward-compatible marker set while migration is in progress.
    for candidate in _iter_candidate_roots(anchor):
        if (candidate / "Tracker_Video_Files").exists() and (candidate / "04_Code").exists():
            return candidate

    raise RuntimeError(
        "Cannot resolve project root from current file location. "
        "Expected the 01_Data..05_Metadata layout."
    )


def raw_data_dir(project_root: Path) -> Path:
    return project_root / "01_Data" / "Raw"


def processed_data_dir(project_root: Path) -> Path:
    return project_root / "01_Data" / "Processed"


def raw_footage_dir(project_root: Path) -> Path:
    return project_root / "02_Videos" / "Raw_Footage"


def experiments_dir(project_root: Path) -> Path:
    return project_root / "02_Videos" / "Experiments"


def plots_dir(project_root: Path) -> Path:
    return project_root / "03_Results" / "Plots"


def reports_dir(project_root: Path) -> Path:
    return project_root / "03_Results" / "Reports"


def code_root_dir(project_root: Path) -> Path:
    return project_root / "04_Code"


def scripts_dir(project_root: Path) -> Path:
    return project_root / "04_Code" / "scripts"


def metadata_dir(project_root: Path) -> Path:
    return project_root / "05_Metadata"


def logs_dir(project_root: Path) -> Path:
    return project_root / "05_Metadata" / "logs"


def color_profile_log_path(project_root: Path) -> Path:
    return logs_dir(project_root) / "flipo_color_profiles.json"


def references_dir(project_root: Path) -> Path:
    return project_root / "06_References"


def resolve_config_path(project_root: Path) -> Path:
    preferred = metadata_dir(project_root) / "config.json"
    if preferred.exists() and preferred.is_file():
        return preferred

    legacy = project_root / "config.json"
    if legacy.exists() and legacy.is_file():
        return legacy

    # Return preferred target so callers can write a new config there.
    return preferred
