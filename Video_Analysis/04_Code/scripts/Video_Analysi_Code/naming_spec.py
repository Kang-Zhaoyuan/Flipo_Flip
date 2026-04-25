from __future__ import annotations

import re
from pathlib import Path
from typing import Final


EXPECTED_TRIAL_FORMAT: Final[str] = "{Color}_D{Infill}_T{Thickness}_L{EdgeWidth}_W{Weight}"
RAW_CSV_SUFFIX: Final[str] = "_raw_data"

# Mirrors FLIPO_FLIP_NAMING_SPEC.md section 3 while keeping numeric groups strict.
TRIAL_STEM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^([A-Za-z]+)_D(\d+)_T(\d+(?:\.\d+)?)_L(\d+(?:\.\d+)?)_W(\d+(?:\.\d+)?)$"
)


def validate_trial_stem_or_raise(stem: str, *, source_label: str) -> str:
    if TRIAL_STEM_PATTERN.fullmatch(stem):
        return stem

    raise ValueError(
        f"Invalid Flipo trial name in {source_label}: '{stem}'. "
        f"Expected format: {EXPECTED_TRIAL_FORMAT}"
    )


def validate_video_filename_or_raise(video_path: Path) -> str:
    if not video_path.suffix:
        raise ValueError(f"Video file has no extension: {video_path}")

    return validate_trial_stem_or_raise(
        video_path.stem,
        source_label=f"video filename '{video_path.name}'",
    )


def raw_csv_stem_from_trial_stem(trial_stem: str) -> str:
    validated = validate_trial_stem_or_raise(
        trial_stem,
        source_label="trial stem used for raw CSV output",
    )
    return f"{validated}{RAW_CSV_SUFFIX}"


def trial_stem_from_raw_csv_stem(raw_csv_stem: str, *, source_label: str) -> str:
    if not raw_csv_stem.endswith(RAW_CSV_SUFFIX):
        raise ValueError(
            f"Invalid raw CSV stem in {source_label}: '{raw_csv_stem}'. "
            f"Expected suffix: {RAW_CSV_SUFFIX}"
        )

    trial_stem = raw_csv_stem[: -len(RAW_CSV_SUFFIX)]
    return validate_trial_stem_or_raise(trial_stem, source_label=source_label)


def validate_raw_csv_filename_or_raise(csv_path: Path) -> str:
    if csv_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected a CSV file, got: {csv_path}")

    return trial_stem_from_raw_csv_stem(
        csv_path.stem,
        source_label=f"raw CSV filename '{csv_path.name}'",
    )
