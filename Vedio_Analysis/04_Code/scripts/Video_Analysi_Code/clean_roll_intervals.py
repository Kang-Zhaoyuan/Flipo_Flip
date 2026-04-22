from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    _scripts_root = Path(__file__).resolve().parent.parent
    if str(_scripts_root) not in sys.path:
        sys.path.insert(0, str(_scripts_root))

from Video_Analysi_Code.csv_paths import (
    default_processed_output_dir,
    load_last_calibration_video_path as shared_load_last_calibration_video_path,
    resolve_csv_path as shared_resolve_csv_path,
    resolve_raw_csv_from_video as shared_resolve_raw_csv_from_video,
    resolve_video_path as shared_resolve_video_path,
)
from Video_Analysi_Code.calibration_io import load_config, save_config
from Video_Analysi_Code.path_registry import (
    find_project_root,
    plots_dir,
    raw_data_dir,
    resolve_config_path,
)

REQUIRED_BASE_COLUMNS = ["frame_index", "timestamp", "x", "y"]
ANGLE_COLUMNS = ["theta", "theta_unwrapped"]
CLEANED_SEGMENT_PATTERN = re.compile(r"(.+)_cleaned_segment_(\d+)\.csv$")


@dataclass
class AngleTrackPack:
    from_theta: np.ndarray
    from_theta_available: bool
    from_theta_unwrapped: np.ndarray
    from_theta_unwrapped_available: bool


@dataclass
class RankedSegment:
    rank: int
    source_index: int
    delta_theta_deg: float
    turns: float
    delta_source: str
    frame_start: int
    frame_end: int
    duration_s: float
    rows: int
    ranking_value: float
    ranking_source: str
    theta_source: str
    segment_df: pd.DataFrame


@dataclass
class FileProcessingResult:
    source_csv: Path
    accepted_segments: int
    rejected_segments: int
    removed_rows_by_y: int
    cleaned_csv_paths: List[Path]
    analysis_csv_path: Path
    summary_plot_path: Path
    segment_plot_paths: List[Path]
    failed: bool = False
    failure_reason: str = ""


@dataclass
class CleaningSelection:
    segments: List[pd.DataFrame]
    logs: List[str]
    removed_by_y: int
    rejected_segments: int
    min_roll_deg_used: float
    fallback_attempts: List[Tuple[float, int]]
    red_mode: bool


@dataclass
class EnergySourceSpec:
    source_slug: str
    thickness_mm: float
    mass_g: float
    height_mm: float
    width_mm: float
    y_top_mm: float
    y_low_mm: float
    gravity_m_s2: float
    inertia_kg_m2: float
    y_low_override_mm: Optional[float]
    config_source: str

    @property
    def mass_kg(self) -> float:
        return self.mass_g / 1000.0


@dataclass
class YMappingProfile:
    y_column: str
    y_bottom_ref: float
    q02_up_mm: float
    q98_up_mm: float
    y_low_mm: float
    y_top_mm: float


@dataclass
class SegmentEnergySummary:
    omega_source: str
    y_clamped_ratio: float
    y_min_mm: float
    y_max_mm: float
    potential_j_peak: float
    trans_j_peak: float
    rot_j_peak: float
    total_j_peak: float
    potential_j_mean: float
    trans_j_mean: float
    rot_j_mean: float
    total_j_mean: float
    total_j_final: float
    potential_j_kg_peak: float
    trans_j_kg_peak: float
    rot_j_kg_peak: float
    total_j_kg_peak: float
    potential_j_kg_mean: float
    trans_j_kg_mean: float
    rot_j_kg_mean: float
    total_j_kg_mean: float
    total_j_kg_final: float


ENERGY_SPECS_KEY = "energy_specs"
ENERGY_DEFAULTS_KEY = "defaults"
ENERGY_SOURCES_KEY = "sources"

DEFAULT_ENERGY_DEFAULTS: Dict[str, float] = {
    "height_mm": 50.0,
    "width_mm": 50.0,
    "mass_g": 1000.0,
    "y_top_mm": 25.0,
    "gravity_m_s2": 9.81,
}
THICKNESS_TOKEN_PATTERN = re.compile(r"(?:^|_)T(\d+(?:\.\d+)?)(?:_|$)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clean Flipo Flip raw CSV files, rank cleaned segments by delta-theta "
            "from theta_unwrapped, and generate analysis/plots automatically."
        )
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help=(
            "Explicit input CSV path. If omitted, CSV is resolved from --video-path "
            "or config.json:last_calibration.video_path."
        ),
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help="Video path used to resolve <video_stem>_raw_data.csv when --csv-path is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for cleaned CSV and analysis CSV outputs. Defaults to 01_Data/Processed.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Directory for generated plot images. Defaults to 03_Results/Plots.",
    )
    parser.add_argument(
        "--batch-all-raw",
        action="store_true",
        help="Process every CSV file in --raw-dir (or default 01_Data/Raw).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Raw CSV directory for batch mode. Defaults to 01_Data/Raw.",
    )
    parser.add_argument(
        "--replot-cleaned-only",
        action="store_true",
        help=(
            "Regenerate trajectory images only from existing *_cleaned_segment_XX.csv "
            "files without running cleaning/segmentation again."
        ),
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default=None,
        help=(
            "Directory containing existing cleaned segment CSV files for "
            "--replot-cleaned-only. Defaults to 01_Data/Processed."
        ),
    )
    parser.add_argument("--y-min", type=float, default=800.0, help="Hard lower Y bound.")
    parser.add_argument("--y-max", type=float, default=950.0, help="Hard upper Y bound.")
    parser.add_argument(
        "--jump-threshold-px",
        type=float,
        default=30.0,
        help="Split candidate segments when |delta_x| or |delta_y| exceeds this threshold.",
    )
    parser.add_argument(
        "--min-candidate-frames",
        type=int,
        default=5,
        help="Discard candidate segments shorter than this frame count.",
    )
    parser.add_argument(
        "--max-reverse-ratio",
        type=float,
        default=0.20,
        help="Reject segments if X max drawdown exceeds this fraction of X total displacement.",
    )
    parser.add_argument(
        "--min-roll-deg",
        type=float,
        default=180.0,
        help="Minimum conservative theta amplitude required for a rolling segment.",
    )
    parser.add_argument(
        "--auto-relax-roll-threshold",
        action="store_true",
        help=(
            "If no segment is accepted at --min-roll-deg, retry with lower thresholds "
            "from --relax-roll-seq until at least one segment is accepted."
        ),
    )
    parser.add_argument(
        "--relax-roll-seq",
        type=str,
        default="150,120,90,60,45",
        help=(
            "Comma-separated fallback min-roll-deg thresholds used when "
            "--auto-relax-roll-threshold is enabled."
        ),
    )
    parser.add_argument(
        "--upright-angle-tol-deg",
        type=float,
        default=15.0,
        help="Start posture tolerance around 0 deg.",
    )
    parser.add_argument(
        "--lying-angle-tol-deg",
        type=float,
        default=5.0,
        help="End posture tolerance around +90/-90 deg.",
    )
    parser.add_argument(
        "--y-high-tol",
        type=float,
        default=10.0,
        help="Start posture Y tolerance above local high reference (q10).",
    )
    parser.add_argument(
        "--y-low-tol",
        type=float,
        default=10.0,
        help="End posture Y tolerance below local low reference (q90).",
    )
    parser.add_argument(
        "--red-ignore-theta-mode",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "Red no-theta policy. auto=enable only for filenames starting with Red; "
            "on=force enable; off=disable."
        ),
    )
    parser.add_argument(
        "--red-start-y-max",
        type=float,
        default=838.0,
        help="Red y-only start gate: y <= red-start-y-max.",
    )
    parser.add_argument(
        "--red-end-y-min",
        type=float,
        default=935.0,
        help="Red y-only end gate: y >= red-end-y-min.",
    )
    parser.add_argument(
        "--red-wobble-smoothing-alpha",
        type=float,
        default=0.25,
        help="Low-pass smoothing factor for Red wobble phase theta updates (0..1).",
    )
    parser.add_argument(
        "--red-forward-dx-threshold",
        type=float,
        default=0.5,
        help="Minimum per-frame dx considered forward motion for Red theta unwrapping.",
    )
    parser.add_argument(
        "--red-cycle-reset-threshold-deg",
        type=float,
        default=35.0,
        help=(
            "If principal theta drops more than this amount during forward motion, "
            "advance Red theta unwrapped cycle by 180 deg."
        ),
    )
    return parser.parse_args()


def load_last_calibration_video_path(config_path: Path) -> Optional[Path]:
    return shared_load_last_calibration_video_path(config_path)


def resolve_video_path(project_root: Path, explicit_video_path: Optional[str]) -> Path:
    return shared_resolve_video_path(project_root, explicit_video_path)


def resolve_csv_from_video(project_root: Path, video_path: Path) -> Tuple[Path, bool]:
    return shared_resolve_raw_csv_from_video(project_root, video_path)


def resolve_csv_path(
    project_root: Path,
    explicit_csv_path: Optional[str],
    explicit_video_path: Optional[str],
) -> Tuple[Path, Optional[Path], bool, bool]:
    return shared_resolve_csv_path(project_root, explicit_csv_path, explicit_video_path)


def _coerce_finite_float(value: object, fallback: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(fallback)

    if not np.isfinite(numeric):
        return float(fallback)
    return float(numeric)


def _coerce_positive_float(value: object, fallback: float) -> float:
    numeric = _coerce_finite_float(value, fallback)
    if numeric <= 0.0:
        return float(fallback)
    return numeric


def _coerce_optional_positive_float(value: object) -> Optional[float]:
    if value is None:
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return float(numeric)


def infer_thickness_mm_from_source_slug(source_slug: str) -> Optional[float]:
    match = THICKNESS_TOKEN_PATTERN.search(source_slug)
    if not match:
        return None

    token = match.group(1)
    if "." in token:
        inferred = _coerce_optional_positive_float(token)
        return inferred

    digits = re.sub(r"\D+", "", token)
    if not digits:
        return None

    numeric = float(digits)
    if len(digits) >= 4:
        # Convention: T1776 means 177.6 mm.
        numeric /= 10.0

    if not np.isfinite(numeric) or numeric <= 0.0:
        return None
    return float(numeric)


def _prompt_required_positive_float(message: str) -> float:
    while True:
        try:
            raw = input(message).strip()
        except EOFError as exc:
            raise RuntimeError(
                "Interactive input is required to set thickness_mm for first-time source."
            ) from exc

        if not raw:
            print("Value is required.")
            continue

        try:
            value = float(raw)
        except ValueError:
            print(f"Invalid number: {raw}")
            continue

        if not np.isfinite(value) or value <= 0.0:
            print("Value must be a positive finite number.")
            continue

        return float(value)


def _ensure_energy_specs_sections(
    config_data: Dict[str, object],
) -> Tuple[Dict[str, object], Dict[str, float], Dict[str, object], bool]:
    dirty = False

    energy_specs_obj = config_data.get(ENERGY_SPECS_KEY)
    if not isinstance(energy_specs_obj, dict):
        energy_specs_obj = {}
        config_data[ENERGY_SPECS_KEY] = energy_specs_obj
        dirty = True

    defaults_obj = energy_specs_obj.get(ENERGY_DEFAULTS_KEY)
    if not isinstance(defaults_obj, dict):
        defaults_obj = {}
        energy_specs_obj[ENERGY_DEFAULTS_KEY] = defaults_obj
        dirty = True

    normalized_defaults: Dict[str, float] = {}
    for key, default_value in DEFAULT_ENERGY_DEFAULTS.items():
        normalized_value = _coerce_positive_float(defaults_obj.get(key), default_value)
        normalized_defaults[key] = normalized_value
        if defaults_obj.get(key) != normalized_value:
            defaults_obj[key] = normalized_value
            dirty = True

    sources_obj = energy_specs_obj.get(ENERGY_SOURCES_KEY)
    if not isinstance(sources_obj, dict):
        sources_obj = {}
        energy_specs_obj[ENERGY_SOURCES_KEY] = sources_obj
        dirty = True

    return energy_specs_obj, normalized_defaults, sources_obj, dirty


def resolve_energy_source_spec(
    project_root: Path,
    source_slug: str,
    source_csv_name: str,
) -> EnergySourceSpec:
    config_path = resolve_config_path(project_root)
    config_data = load_config(config_path)

    _, defaults, sources_obj, dirty = _ensure_energy_specs_sections(config_data)

    source_obj = sources_obj.get(source_slug)
    if not isinstance(source_obj, dict):
        source_obj = {}
        sources_obj[source_slug] = source_obj
        dirty = True

    source_entry: Dict[str, Any] = source_obj
    config_source = "config"

    thickness_mm = _coerce_optional_positive_float(source_entry.get("thickness_mm"))
    if thickness_mm is None:
        inferred_thickness = infer_thickness_mm_from_source_slug(source_slug)
        if inferred_thickness is not None:
            thickness_mm = inferred_thickness
            config_source = "inferred_from_name"
        else:
            thickness_mm = defaults["height_mm"]
            config_source = "defaults_fallback"
    if source_entry.get("thickness_mm") != thickness_mm:
        source_entry["thickness_mm"] = thickness_mm
        dirty = True

    mass_g = _coerce_positive_float(source_entry.get("mass_g"), defaults["mass_g"])
    if source_entry.get("mass_g") != mass_g:
        source_entry["mass_g"] = mass_g
        dirty = True

    height_mm = _coerce_positive_float(source_entry.get("height_mm"), defaults["height_mm"])
    if source_entry.get("height_mm") != height_mm:
        source_entry["height_mm"] = height_mm
        dirty = True

    width_mm = _coerce_positive_float(source_entry.get("width_mm"), defaults["width_mm"])
    if source_entry.get("width_mm") != width_mm:
        source_entry["width_mm"] = width_mm
        dirty = True

    y_top_mm = _coerce_positive_float(source_entry.get("y_top_mm"), defaults["y_top_mm"])
    if source_entry.get("y_top_mm") != y_top_mm:
        source_entry["y_top_mm"] = y_top_mm
        dirty = True

    gravity_m_s2 = _coerce_positive_float(
        source_entry.get("gravity_m_s2"), defaults["gravity_m_s2"]
    )
    if source_entry.get("gravity_m_s2") != gravity_m_s2:
        source_entry["gravity_m_s2"] = gravity_m_s2
        dirty = True

    y_low_override_mm = _coerce_optional_positive_float(source_entry.get("y_low_override_mm"))
    if source_entry.get("y_low_override_mm") != y_low_override_mm:
        source_entry["y_low_override_mm"] = y_low_override_mm
        dirty = True

    default_y_low_mm = float(thickness_mm) / 2.0
    y_low_mm = default_y_low_mm
    if y_low_override_mm is not None:
        y_low_mm = y_low_override_mm

    if y_top_mm <= y_low_mm + 1e-9:
        if y_low_override_mm is not None:
            y_low_mm = default_y_low_mm
            source_entry["y_low_override_mm"] = None
            y_low_override_mm = None
            dirty = True
        if y_top_mm <= y_low_mm + 1e-9:
            auto_span_mm = max(1.0, 0.05 * max(height_mm, thickness_mm))
            y_top_mm = y_low_mm + auto_span_mm
            source_entry["y_top_mm"] = y_top_mm
            dirty = True
            if config_source == "config":
                config_source = "config_auto_fixed"
            else:
                config_source = f"{config_source}+auto_fixed"

    mass_kg = mass_g / 1000.0
    thickness_m = thickness_mm / 1000.0
    height_m = height_mm / 1000.0
    inertia_kg_m2 = (mass_kg * (height_m ** 2 + thickness_m ** 2)) / 12.0

    if dirty:
        save_config(config_path, config_data)

    return EnergySourceSpec(
        source_slug=source_slug,
        thickness_mm=thickness_mm,
        mass_g=mass_g,
        height_mm=height_mm,
        width_mm=width_mm,
        y_top_mm=y_top_mm,
        y_low_mm=y_low_mm,
        gravity_m_s2=gravity_m_s2,
        inertia_kg_m2=inertia_kg_m2,
        y_low_override_mm=y_low_override_mm,
        config_source=config_source,
    )


def load_and_prepare_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing_required = [column for column in REQUIRED_BASE_COLUMNS if column not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    for column in REQUIRED_BASE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    for column in ANGLE_COLUMNS:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if len(df) < 2:
        raise ValueError("At least two rows are required.")

    if df["timestamp"].notna().sum() < 2:
        raise ValueError("Timestamp column has insufficient finite values.")

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def build_unwrapped_track(series: pd.Series) -> Tuple[np.ndarray, bool]:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() == 0:
        return np.full(len(series), np.nan, dtype=float), False

    filled = numeric.interpolate(method="linear", limit_direction="both").to_numpy(dtype=float)
    if not np.isfinite(filled).all():
        return np.full(len(series), np.nan, dtype=float), False

    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(filled)))
    return unwrapped.astype(float), True


def build_angle_tracks(segment_df: pd.DataFrame) -> AngleTrackPack:
    theta_track, theta_ok = build_unwrapped_track(segment_df["theta"])
    theta_unwrapped_track, theta_unwrapped_ok = build_unwrapped_track(segment_df["theta_unwrapped"])

    return AngleTrackPack(
        from_theta=theta_track,
        from_theta_available=theta_ok,
        from_theta_unwrapped=theta_unwrapped_track,
        from_theta_unwrapped_available=theta_unwrapped_ok,
    )


def conservative_amplitude_deg(angle_tracks: AngleTrackPack) -> float:
    amplitudes: List[float] = []

    if angle_tracks.from_theta_available:
        amplitudes.append(float(np.nanmax(angle_tracks.from_theta) - np.nanmin(angle_tracks.from_theta)))

    if angle_tracks.from_theta_unwrapped_available:
        amplitudes.append(
            float(
                np.nanmax(angle_tracks.from_theta_unwrapped)
                - np.nanmin(angle_tracks.from_theta_unwrapped)
            )
        )

    if not amplitudes:
        return float("nan")

    return float(min(amplitudes))


def angular_distance_deg(values: np.ndarray, target_deg: float) -> np.ndarray:
    return np.abs(((values - target_deg + 180.0) % 360.0) - 180.0)


def combine_conservative_masks(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    a_available: bool,
    b_available: bool,
) -> np.ndarray:
    if a_available and b_available:
        return mask_a & mask_b
    if a_available:
        return mask_a
    if b_available:
        return mask_b
    return np.zeros(mask_a.shape, dtype=bool)


def mask_upright(values: np.ndarray, tol_deg: float) -> np.ndarray:
    return np.isfinite(values) & (angular_distance_deg(values, 0.0) <= tol_deg)


def mask_lying(values: np.ndarray, tol_deg: float) -> np.ndarray:
    dist_pos_90 = angular_distance_deg(values, 90.0)
    dist_neg_90 = angular_distance_deg(values, -90.0)
    return np.isfinite(values) & ((dist_pos_90 <= tol_deg) | (dist_neg_90 <= tol_deg))


def apply_y_bounds(df: pd.DataFrame, y_min: float, y_max: float) -> pd.DataFrame:
    mask = (
        np.isfinite(df["x"].to_numpy(dtype=float))
        & np.isfinite(df["y"].to_numpy(dtype=float))
        & (df["y"].to_numpy(dtype=float) >= y_min)
        & (df["y"].to_numpy(dtype=float) <= y_max)
    )
    return df.loc[mask].copy().reset_index(drop=True)


def split_candidate_segments(
    filtered_df: pd.DataFrame,
    jump_threshold_px: float,
    min_candidate_frames: int,
) -> List[pd.DataFrame]:
    if filtered_df.empty:
        return []

    frame_diff = filtered_df["frame_index"].diff()
    frame_break = frame_diff.isna() | (frame_diff <= 0.0) | (frame_diff > 1.0)

    dx = filtered_df["x"].diff().abs().fillna(0.0)
    dy = filtered_df["y"].diff().abs().fillna(0.0)
    jump_break = (dx > jump_threshold_px) | (dy > jump_threshold_px)

    segment_start = frame_break | jump_break
    segment_start.iloc[0] = True
    segment_id = segment_start.cumsum()

    segments: List[pd.DataFrame] = []
    for _, part in filtered_df.groupby(segment_id, sort=False):
        if len(part) >= min_candidate_frames:
            segments.append(part.reset_index(drop=True))

    return segments


def x_direction_metrics(segment_df: pd.DataFrame) -> Tuple[float, float, float]:
    x_values = segment_df["x"].to_numpy(dtype=float)
    finite_x = x_values[np.isfinite(x_values)]

    if len(finite_x) < 2:
        return float("nan"), float("nan"), float("nan")

    total_displacement = float(np.nanmax(finite_x) - np.nanmin(finite_x))
    running_peak = np.maximum.accumulate(finite_x)
    max_drawdown = float(np.nanmax(running_peak - finite_x))
    net_decrease = float(max(0.0, finite_x[0] - finite_x[-1]))

    return total_displacement, max_drawdown, net_decrease


def passes_x_direction_rule(segment_df: pd.DataFrame, max_reverse_ratio: float) -> Tuple[bool, float, float, float]:
    total_disp, max_drawdown, net_decrease = x_direction_metrics(segment_df)

    if not np.isfinite(total_disp) or total_disp <= 1e-9:
        return False, total_disp, max_drawdown, net_decrease

    reverse_limit = max_reverse_ratio * total_disp
    # User rule: allow local reversals; reject only if net decrease exceeds threshold.
    passed = net_decrease <= reverse_limit
    return passed, total_disp, max_drawdown, net_decrease


def split_by_x_reversal(
    candidate_df: pd.DataFrame,
    max_reverse_ratio: float,
    min_candidate_frames: int,
    max_splits: int = 6,
) -> Tuple[List[pd.DataFrame], List[str], int]:
    queue: List[pd.DataFrame] = [candidate_df.reset_index(drop=True)]
    passed_segments: List[pd.DataFrame] = []
    logs: List[str] = []
    rejected_count = 0
    split_count = 0

    while queue:
        segment = queue.pop(0)
        x_ok, total_disp, max_drawdown, net_decrease = passes_x_direction_rule(
            segment,
            max_reverse_ratio=max_reverse_ratio,
        )
        if x_ok:
            passed_segments.append(segment)
            continue

        if split_count >= max_splits:
            logs.append(
                "X rule failed and split budget exhausted "
                + f"(total_disp={total_disp:.3f}, max_drawdown={max_drawdown:.3f}, "
                + f"net_decrease={net_decrease:.3f})."
            )
            rejected_count += 1
            continue

        if len(segment) < 2 * min_candidate_frames:
            logs.append(
                "X rule failed and segment too short to split safely "
                + f"(rows={len(segment)})."
            )
            rejected_count += 1
            continue

        x_values = segment["x"].to_numpy(dtype=float)
        if not np.isfinite(x_values).any():
            logs.append("X rule failed and segment has no finite x values.")
            rejected_count += 1
            continue

        peak_index = int(np.nanargmax(x_values))
        left_length = peak_index + 1
        right_length = len(segment) - peak_index - 1

        if left_length < min_candidate_frames or right_length < min_candidate_frames:
            logs.append(
                "X rule failed and peak split would create too-short segment "
                + f"(left_rows={left_length}, right_rows={right_length})."
            )
            rejected_count += 1
            continue

        split_count += 1
        peak_frame = int(segment["frame_index"].iloc[peak_index])
        logs.append(
            "X rule failed; split at peak frame "
            + f"{peak_frame} (total_disp={total_disp:.3f}, max_drawdown={max_drawdown:.3f})."
        )

        left = segment.iloc[: peak_index + 1].copy().reset_index(drop=True)
        right = segment.iloc[peak_index + 1 :].copy().reset_index(drop=True)
        queue = [left, right] + queue

    return passed_segments, logs, rejected_count


def trim_segment_by_posture(
    segment_df: pd.DataFrame,
    angle_tracks: AngleTrackPack,
    upright_angle_tol_deg: float,
    lying_angle_tol_deg: float,
    y_high_tol: float,
    y_low_tol: float,
) -> Optional[pd.DataFrame]:
    y_values = segment_df["y"].to_numpy(dtype=float)
    finite_y = y_values[np.isfinite(y_values)]
    if len(finite_y) == 0:
        return None

    y_high_ref = float(np.nanquantile(finite_y, 0.10))
    y_low_ref = float(np.nanquantile(finite_y, 0.90))

    start_y_mask = np.isfinite(y_values) & (y_values <= y_high_ref + y_high_tol)
    end_y_mask = np.isfinite(y_values) & (y_values >= y_low_ref - y_low_tol)

    upright_theta = mask_upright(angle_tracks.from_theta, upright_angle_tol_deg)
    upright_unwrapped = mask_upright(angle_tracks.from_theta_unwrapped, upright_angle_tol_deg)
    lying_theta = mask_lying(angle_tracks.from_theta, lying_angle_tol_deg)
    lying_unwrapped = mask_lying(angle_tracks.from_theta_unwrapped, lying_angle_tol_deg)

    start_angle_mask = combine_conservative_masks(
        upright_theta,
        upright_unwrapped,
        angle_tracks.from_theta_available,
        angle_tracks.from_theta_unwrapped_available,
    )
    end_angle_mask = combine_conservative_masks(
        lying_theta,
        lying_unwrapped,
        angle_tracks.from_theta_available,
        angle_tracks.from_theta_unwrapped_available,
    )

    start_mask = start_angle_mask & start_y_mask
    end_mask = end_angle_mask & end_y_mask

    start_candidates = np.flatnonzero(start_mask)
    if start_candidates.size == 0:
        return None

    start_index = int(start_candidates[0])
    end_candidates = np.flatnonzero(end_mask & (np.arange(len(segment_df)) >= start_index))
    if end_candidates.size == 0:
        return None

    end_index = int(end_candidates[-1])
    if end_index < start_index:
        return None

    trimmed = segment_df.iloc[start_index : end_index + 1].copy().reset_index(drop=True)
    if len(trimmed) < 2:
        return None

    return trimmed


def find_next_cleaned_index(output_dir: Path) -> int:
    pattern = re.compile(r"cleaned_data_(\d+)\.csv$")
    max_index = 0

    for path in output_dir.glob("cleaned_data_*.csv"):
        match = pattern.match(path.name)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return max_index + 1


def summarize_segment(segment_df: pd.DataFrame) -> str:
    first_frame = int(segment_df["frame_index"].iloc[0])
    last_frame = int(segment_df["frame_index"].iloc[-1])
    return f"frames {first_frame}-{last_frame}, rows={len(segment_df)}"


def clean_segments(
    df: pd.DataFrame,
    y_min: float,
    y_max: float,
    jump_threshold_px: float,
    min_candidate_frames: int,
    max_reverse_ratio: float,
    min_roll_deg: float,
    upright_angle_tol_deg: float,
    lying_angle_tol_deg: float,
    y_high_tol: float,
    y_low_tol: float,
) -> Tuple[List[pd.DataFrame], List[str], int, int]:
    logs: List[str] = []

    y_filtered = apply_y_bounds(df, y_min=y_min, y_max=y_max)
    removed_by_y = len(df) - len(y_filtered)
    logs.append(f"Step1 Y filter: kept {len(y_filtered)}/{len(df)} rows (removed {removed_by_y}).")

    candidates = split_candidate_segments(
        y_filtered,
        jump_threshold_px=jump_threshold_px,
        min_candidate_frames=min_candidate_frames,
    )
    logs.append(f"Step2 segmentation: {len(candidates)} candidate segments.")

    valid_segments: List[pd.DataFrame] = []
    rejected_segments = 0

    for index, candidate in enumerate(candidates, start=1):
        prefix = f"Candidate {index}: {summarize_segment(candidate)}"

        x_subsegments, x_logs, x_rejected = split_by_x_reversal(
            candidate,
            max_reverse_ratio=max_reverse_ratio,
            min_candidate_frames=min_candidate_frames,
        )
        for detail in x_logs:
            logs.append(prefix + " -> " + detail)

        rejected_segments += x_rejected
        if not x_subsegments:
            logs.append(prefix + " -> rejected by X direction (no valid forward subsegment).")
            continue

        for sub_index, x_valid_segment in enumerate(x_subsegments, start=1):
            sub_prefix = prefix + f" [sub{sub_index}]"

            tracks = build_angle_tracks(x_valid_segment)
            amplitude = conservative_amplitude_deg(tracks)
            if not np.isfinite(amplitude) or amplitude < min_roll_deg:
                logs.append(
                    sub_prefix
                    + f" -> rejected by theta amplitude ({amplitude:.3f} < {min_roll_deg:.3f})."
                )
                rejected_segments += 1
                continue

            trimmed = trim_segment_by_posture(
                x_valid_segment,
                angle_tracks=tracks,
                upright_angle_tol_deg=upright_angle_tol_deg,
                lying_angle_tol_deg=lying_angle_tol_deg,
                y_high_tol=y_high_tol,
                y_low_tol=y_low_tol,
            )
            if trimmed is None:
                logs.append(
                    sub_prefix + " -> rejected by posture trimming (no valid start/end frame)."
                )
                rejected_segments += 1
                continue

            logs.append(
                sub_prefix
                + f" -> accepted, amplitude={amplitude:.3f}, trimmed to {summarize_segment(trimmed)}."
            )
            valid_segments.append(trimmed)

    return valid_segments, logs, removed_by_y, rejected_segments


def resolve_red_mode(csv_path: Path, args: argparse.Namespace) -> bool:
    if args.red_ignore_theta_mode == "on":
        return True
    if args.red_ignore_theta_mode == "off":
        return False
    return csv_path.stem.lower().startswith("red")


def trim_segment_by_y_only(
    segment_df: pd.DataFrame,
    start_y_max: float,
    end_y_min: float,
) -> Optional[pd.DataFrame]:
    y_values = pd.to_numeric(segment_df["y"], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(y_values)
    if int(finite_mask.sum()) < 2:
        return None

    start_candidates = np.flatnonzero(finite_mask & (y_values <= start_y_max))
    if start_candidates.size == 0:
        return None

    start_index = int(start_candidates[0])
    end_candidates = np.flatnonzero(
        finite_mask
        & (y_values >= end_y_min)
        & (np.arange(len(segment_df), dtype=int) >= start_index)
    )
    if end_candidates.size == 0:
        return None

    end_index = int(end_candidates[-1])
    if end_index <= start_index:
        return None

    trimmed = segment_df.iloc[start_index : end_index + 1].copy().reset_index(drop=True)
    if len(trimmed) < 2:
        return None

    return trimmed


def clean_segments_red_no_theta(
    df: pd.DataFrame,
    y_min: float,
    y_max: float,
    jump_threshold_px: float,
    min_candidate_frames: int,
    max_reverse_ratio: float,
    start_y_max: float,
    end_y_min: float,
) -> Tuple[List[pd.DataFrame], List[str], int, int]:
    logs: List[str] = []

    y_filtered = apply_y_bounds(df, y_min=y_min, y_max=y_max)
    removed_by_y = len(df) - len(y_filtered)
    logs.append(f"Step1 Y filter: kept {len(y_filtered)}/{len(df)} rows (removed {removed_by_y}).")

    candidates = split_candidate_segments(
        y_filtered,
        jump_threshold_px=jump_threshold_px,
        min_candidate_frames=min_candidate_frames,
    )
    logs.append(f"Step2 segmentation: {len(candidates)} candidate segments.")
    logs.append(
        "Red no-theta mode: bypass theta amplitude/posture checks and use fixed y gates "
        + f"start<= {start_y_max:.3f}, end>= {end_y_min:.3f}."
    )

    valid_segments: List[pd.DataFrame] = []
    rejected_segments = 0

    for index, candidate in enumerate(candidates, start=1):
        prefix = f"Candidate {index}: {summarize_segment(candidate)}"

        x_subsegments, x_logs, x_rejected = split_by_x_reversal(
            candidate,
            max_reverse_ratio=max_reverse_ratio,
            min_candidate_frames=min_candidate_frames,
        )
        for detail in x_logs:
            logs.append(prefix + " -> " + detail)

        rejected_segments += x_rejected
        if not x_subsegments:
            logs.append(prefix + " -> rejected by X direction (no valid forward subsegment).")
            continue

        for sub_index, x_valid_segment in enumerate(x_subsegments, start=1):
            sub_prefix = prefix + f" [sub{sub_index}]"
            trimmed = trim_segment_by_y_only(
                x_valid_segment,
                start_y_max=start_y_max,
                end_y_min=end_y_min,
            )
            if trimmed is None:
                logs.append(
                    sub_prefix + " -> rejected by Red y-only trimming (no valid start/end frame)."
                )
                rejected_segments += 1
                continue

            logs.append(
                sub_prefix
                + " -> accepted (Red no-theta mode), "
                + f"trimmed to {summarize_segment(trimmed)}."
            )
            valid_segments.append(trimmed)

    return valid_segments, logs, removed_by_y, rejected_segments


def choose_red_theta_columns(segment_df: pd.DataFrame) -> Tuple[str, str]:
    if "y_mm" in segment_df.columns and pd.to_numeric(
        segment_df["y_mm"], errors="coerce"
    ).notna().sum() >= 2:
        y_col = "y_mm"
    else:
        y_col = "y"

    if "x_mm" in segment_df.columns and pd.to_numeric(
        segment_df["x_mm"], errors="coerce"
    ).notna().sum() >= 2:
        x_col = "x_mm"
    else:
        x_col = "x"

    return x_col, y_col


def reconstruct_theta_from_yx(
    segment_df: pd.DataFrame,
    wobble_alpha: float,
    forward_dx_threshold: float,
    cycle_reset_threshold_deg: float,
) -> Tuple[pd.DataFrame, str]:
    updated = segment_df.copy()
    x_col, y_col = choose_red_theta_columns(updated)

    y_values = pd.to_numeric(updated[y_col], errors="coerce").to_numpy(dtype=float)
    x_values = pd.to_numeric(updated[x_col], errors="coerce").to_numpy(dtype=float)

    y_up = to_up_positive_display_y(y_values)
    y_up_series = pd.Series(y_up)
    y_up_filled = y_up_series.interpolate(method="linear", limit_direction="both").ffill().bfill()
    if y_up_filled.notna().sum() < 2:
        updated["theta"] = np.nan
        updated["theta_unwrapped"] = np.nan
        return updated, "red_yx_reconstructed_unavailable"

    y_up_calc = y_up_filled.to_numpy(dtype=float)
    y_low = float(np.nanquantile(y_up_calc, 0.05))
    y_high = float(np.nanquantile(y_up_calc, 0.95))
    if y_high <= y_low + 1e-9:
        y_low = float(np.nanmin(y_up_calc))
        y_high = float(np.nanmax(y_up_calc))

    denom = max(1e-9, y_high - y_low)
    y_norm = (y_up_calc - y_low) / denom
    y_norm = np.clip(y_norm, -1.0, 1.0)
    theta_principal = np.rad2deg(np.arccos(y_norm))

    x_series = pd.Series(x_values)
    x_calc = x_series.interpolate(method="linear", limit_direction="both").ffill().bfill().to_numpy(dtype=float)
    dx = np.diff(x_calc, prepend=x_calc[0])

    theta_unwrapped = np.zeros(len(updated), dtype=float)
    theta_unwrapped[0] = float(theta_principal[0])
    phase_index = 0

    wobble_alpha = float(np.clip(wobble_alpha, 0.0, 1.0))

    for index in range(1, len(updated)):
        principal = float(theta_principal[index])
        previous = float(theta_unwrapped[index - 1])
        previous_in_phase = previous - 180.0 * phase_index

        forward_motion = bool(dx[index] >= forward_dx_threshold)
        if forward_motion and principal < (previous_in_phase - cycle_reset_threshold_deg):
            phase_index += 1

        target_unwrapped = principal + 180.0 * phase_index
        if forward_motion:
            if target_unwrapped < previous:
                target_unwrapped = previous
            theta_unwrapped[index] = target_unwrapped
        else:
            theta_unwrapped[index] = (1.0 - wobble_alpha) * previous + wobble_alpha * target_unwrapped

    theta_wrapped = ((theta_unwrapped + 180.0) % 360.0) - 180.0
    updated["theta"] = np.round(theta_wrapped, 3)
    updated["theta_unwrapped"] = np.round(theta_unwrapped, 3)

    timestamps = make_strictly_increasing(
        pd.to_numeric(updated["timestamp"], errors="coerce").to_numpy(dtype=float)
    )
    if len(updated) >= 2:
        updated["omega_rad_s"] = np.gradient(np.deg2rad(theta_unwrapped), timestamps)
    else:
        updated["omega_rad_s"] = np.nan

    return updated, "red_yx_reconstructed"


def parse_relax_roll_seq(raw_value: str) -> List[float]:
    thresholds: List[float] = []
    for token in raw_value.split(","):
        stripped = token.strip()
        if not stripped:
            continue

        try:
            threshold = float(stripped)
        except ValueError as exc:
            raise ValueError(f"Invalid --relax-roll-seq value: {stripped}") from exc

        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError(
                "Each --relax-roll-seq value must be a finite positive number. "
                + f"Got: {stripped}"
            )
        thresholds.append(threshold)

    unique_thresholds: List[float] = []
    seen: set[float] = set()
    for threshold in thresholds:
        if threshold in seen:
            continue
        seen.add(threshold)
        unique_thresholds.append(threshold)

    return unique_thresholds


def summarize_angle_quality(df: pd.DataFrame) -> List[str]:
    lines: List[str] = []
    total_rows = len(df)

    for column in ANGLE_COLUMNS:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        finite_count = int(finite.size)

        if finite_count >= 2:
            span_deg = float(np.nanmax(finite) - np.nanmin(finite))
            span_text = f"{span_deg:.3f}"
        else:
            span_text = "nan"

        lines.append(
            f" - {column}: finite={finite_count}/{total_rows}, span_deg={span_text}"
        )

    return lines


def run_cleaning_with_optional_roll_relax(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> CleaningSelection:
    base_segments, base_logs, base_removed_by_y, base_rejected_segments = clean_segments(
        df=df,
        y_min=args.y_min,
        y_max=args.y_max,
        jump_threshold_px=args.jump_threshold_px,
        min_candidate_frames=args.min_candidate_frames,
        max_reverse_ratio=args.max_reverse_ratio,
        min_roll_deg=args.min_roll_deg,
        upright_angle_tol_deg=args.upright_angle_tol_deg,
        lying_angle_tol_deg=args.lying_angle_tol_deg,
        y_high_tol=args.y_high_tol,
        y_low_tol=args.y_low_tol,
    )

    if base_segments or not args.auto_relax_roll_threshold:
        return CleaningSelection(
            segments=base_segments,
            logs=base_logs,
            removed_by_y=base_removed_by_y,
            rejected_segments=base_rejected_segments,
            min_roll_deg_used=args.min_roll_deg,
            fallback_attempts=[],
            red_mode=False,
        )

    fallback_attempts: List[Tuple[float, int]] = []
    for fallback_threshold in parse_relax_roll_seq(args.relax_roll_seq):
        if fallback_threshold >= args.min_roll_deg:
            continue

        segments, logs, removed_by_y, rejected_segments = clean_segments(
            df=df,
            y_min=args.y_min,
            y_max=args.y_max,
            jump_threshold_px=args.jump_threshold_px,
            min_candidate_frames=args.min_candidate_frames,
            max_reverse_ratio=args.max_reverse_ratio,
            min_roll_deg=fallback_threshold,
            upright_angle_tol_deg=args.upright_angle_tol_deg,
            lying_angle_tol_deg=args.lying_angle_tol_deg,
            y_high_tol=args.y_high_tol,
            y_low_tol=args.y_low_tol,
        )

        fallback_attempts.append((fallback_threshold, len(segments)))
        if segments:
            return CleaningSelection(
                segments=segments,
                logs=logs,
                removed_by_y=removed_by_y,
                rejected_segments=rejected_segments,
                min_roll_deg_used=fallback_threshold,
                fallback_attempts=fallback_attempts,
                red_mode=False,
            )

    return CleaningSelection(
        segments=base_segments,
        logs=base_logs,
        removed_by_y=base_removed_by_y,
        rejected_segments=base_rejected_segments,
        min_roll_deg_used=args.min_roll_deg,
        fallback_attempts=fallback_attempts,
        red_mode=False,
    )


def run_cleaning_with_mode(
    df: pd.DataFrame,
    args: argparse.Namespace,
    csv_path: Path,
) -> CleaningSelection:
    red_mode = resolve_red_mode(csv_path, args)
    if not red_mode:
        return run_cleaning_with_optional_roll_relax(df=df, args=args)

    segments, logs, removed_by_y, rejected_segments = clean_segments_red_no_theta(
        df=df,
        y_min=args.y_min,
        y_max=args.y_max,
        jump_threshold_px=args.jump_threshold_px,
        min_candidate_frames=args.min_candidate_frames,
        max_reverse_ratio=args.max_reverse_ratio,
        start_y_max=args.red_start_y_max,
        end_y_min=args.red_end_y_min,
    )

    return CleaningSelection(
        segments=segments,
        logs=logs,
        removed_by_y=removed_by_y,
        rejected_segments=rejected_segments,
        min_roll_deg_used=float("nan"),
        fallback_attempts=[],
        red_mode=True,
    )


def sanitize_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return cleaned or "source"


def resolve_optional_dir(
    project_root: Path,
    path_arg: Optional[str],
    default_dir: Path,
) -> Path:
    if path_arg:
        target_dir = Path(path_arg).expanduser()
        if not target_dir.is_absolute():
            target_dir = (project_root / target_dir).resolve()
    else:
        target_dir = default_dir

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def resolve_raw_input_dir(project_root: Path, raw_dir_arg: Optional[str]) -> Path:
    if raw_dir_arg:
        raw_dir = Path(raw_dir_arg).expanduser()
        if not raw_dir.is_absolute():
            raw_dir = (project_root / raw_dir).resolve()
    else:
        raw_dir = raw_data_dir(project_root)

    if not raw_dir.exists() or not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_dir}")

    return raw_dir


def discover_raw_csv_files(raw_dir: Path) -> List[Path]:
    csv_paths = sorted(path.resolve() for path in raw_dir.glob("*.csv") if path.is_file())
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under raw directory: {raw_dir}")
    return csv_paths


def make_strictly_increasing(time_values: np.ndarray) -> np.ndarray:
    if np.isfinite(time_values).sum() < 2:
        return np.arange(len(time_values), dtype=float)

    timestamps = pd.Series(time_values.astype(float)).interpolate(
        method="linear", limit_direction="both"
    )
    timestamps = timestamps.ffill().bfill().to_numpy(dtype=float)

    if not np.isfinite(timestamps).all():
        return np.arange(len(time_values), dtype=float)

    for index in range(1, len(timestamps)):
        if timestamps[index] <= timestamps[index - 1]:
            timestamps[index] = np.nextafter(timestamps[index - 1], np.inf)

    return timestamps


def choose_position_columns(segment_df: pd.DataFrame) -> Tuple[str, str]:
    for x_col, y_col in (("x_mm", "y_mm"), ("x", "y")):
        if x_col not in segment_df.columns or y_col not in segment_df.columns:
            continue

        x_values = pd.to_numeric(segment_df[x_col], errors="coerce").to_numpy(dtype=float)
        y_values = pd.to_numeric(segment_df[y_col], errors="coerce").to_numpy(dtype=float)
        valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
        if int(valid_mask.sum()) >= 2:
            return x_col, y_col

    raise ValueError("Cannot find usable position columns. Expected x_mm/y_mm or x/y.")


def compute_segment_speed(
    segment_df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        timestamps,
        x_raw,
        y_raw,
        _,
        _,
        speed,
    ) = compute_motion_kinematics(
        segment_df,
        x_values=pd.to_numeric(segment_df[x_col], errors="coerce").to_numpy(dtype=float),
        y_values=pd.to_numeric(segment_df[y_col], errors="coerce").to_numpy(dtype=float),
    )
    return timestamps, x_raw, y_raw, speed


def _interpolate_finite_series(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(float)).interpolate(
        method="linear",
        limit_direction="both",
    )
    return series.ffill().bfill().to_numpy(dtype=float)


def compute_motion_kinematics(
    segment_df: pd.DataFrame,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps = make_strictly_increasing(
        pd.to_numeric(segment_df["timestamp"], errors="coerce").to_numpy(dtype=float)
    )

    x_raw = np.asarray(x_values, dtype=float)
    y_raw = np.asarray(y_values, dtype=float)

    x_calc = _interpolate_finite_series(x_raw)
    y_calc = _interpolate_finite_series(y_raw)

    if not np.isfinite(x_calc).all() or not np.isfinite(y_calc).all():
        raise ValueError("Position interpolation failed for motion kinematics.")

    vx = np.gradient(x_calc, timestamps)
    vy = np.gradient(y_calc, timestamps)
    speed = np.hypot(vx, vy)

    return timestamps, x_raw, y_raw, x_calc, y_calc, speed


def to_up_positive_display_y(
    y_values: np.ndarray,
    y_bottom_ref: Optional[float] = None,
) -> np.ndarray:
    y_display = np.full(y_values.shape, np.nan, dtype=float)
    finite_mask = np.isfinite(y_values)
    if not np.any(finite_mask):
        return y_display

    # Convert OpenCV-style y-down coordinates to an up-positive display frame.
    if y_bottom_ref is None or not np.isfinite(y_bottom_ref):
        y_bottom_ref = float(np.nanmax(y_values[finite_mask]))
    y_display[finite_mask] = float(y_bottom_ref) - y_values[finite_mask]
    return y_display


def build_source_y_mapping_profile(
    source_df: pd.DataFrame,
    y_col: str,
    energy_spec: EnergySourceSpec,
) -> YMappingProfile:
    y_values = pd.to_numeric(source_df[y_col], errors="coerce").to_numpy(dtype=float)
    finite_mask = np.isfinite(y_values)
    if int(finite_mask.sum()) < 2:
        raise ValueError(
            "Cannot build Y mapping profile: insufficient finite source Y values."
        )

    y_bottom_ref = float(np.nanmax(y_values[finite_mask]))
    y_up = to_up_positive_display_y(y_values, y_bottom_ref=y_bottom_ref)
    finite_up = y_up[np.isfinite(y_up)]

    q02 = float(np.nanquantile(finite_up, 0.02))
    q98 = float(np.nanquantile(finite_up, 0.98))
    if q98 <= q02 + 1e-9:
        q02 = float(np.nanmin(finite_up))
        q98 = float(np.nanmax(finite_up))

    if q98 <= q02 + 1e-9:
        raise ValueError(
            "Cannot build Y mapping profile: source Y dynamic range is too small."
        )

    return YMappingProfile(
        y_column=y_col,
        y_bottom_ref=y_bottom_ref,
        q02_up_mm=q02,
        q98_up_mm=q98,
        y_low_mm=energy_spec.y_low_mm,
        y_top_mm=energy_spec.y_top_mm,
    )


def map_y_to_physical_height_mm(
    y_values: np.ndarray,
    profile: YMappingProfile,
) -> Tuple[np.ndarray, np.ndarray, float]:
    y_up = to_up_positive_display_y(y_values, y_bottom_ref=profile.y_bottom_ref)
    y_physical = np.full(y_up.shape, np.nan, dtype=float)

    finite_mask = np.isfinite(y_up)
    if not np.any(finite_mask):
        return y_up, y_physical, 0.0

    denom = max(1e-9, profile.q98_up_mm - profile.q02_up_mm)
    alpha = (y_up[finite_mask] - profile.q02_up_mm) / denom
    alpha_clipped = np.clip(alpha, 0.0, 1.0)
    y_physical[finite_mask] = profile.y_low_mm + alpha_clipped * (
        profile.y_top_mm - profile.y_low_mm
    )

    clamped_ratio = float(np.mean((alpha < 0.0) | (alpha > 1.0)))
    return y_up, y_physical, clamped_ratio


def resolve_theta_track_for_energy(segment_df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if "theta_unwrapped" in segment_df.columns:
        theta_unwrapped = pd.to_numeric(segment_df["theta_unwrapped"], errors="coerce").to_numpy(
            dtype=float
        )
        if np.isfinite(theta_unwrapped).sum() >= 2:
            return theta_unwrapped, "theta_unwrapped"

    tracks = build_angle_tracks(segment_df)
    if tracks.from_theta_unwrapped_available:
        return tracks.from_theta_unwrapped, "derived_theta_unwrapped"
    if tracks.from_theta_available:
        return tracks.from_theta, "derived_theta"

    return np.full(len(segment_df), np.nan, dtype=float), "unavailable"


def resolve_segment_omega_rad_s(
    segment_df: pd.DataFrame,
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, str]:
    omega_raw = np.full(len(segment_df), np.nan, dtype=float)
    has_raw = False
    if "omega_rad_s" in segment_df.columns:
        omega_raw = pd.to_numeric(segment_df["omega_rad_s"], errors="coerce").to_numpy(dtype=float)
        has_raw = np.isfinite(omega_raw).sum() >= 2

    theta_track_deg, theta_source = resolve_theta_track_for_energy(segment_df)
    omega_derived = np.full(len(segment_df), np.nan, dtype=float)
    has_derived = np.isfinite(theta_track_deg).sum() >= 2
    if has_derived:
        theta_calc = _interpolate_finite_series(theta_track_deg)
        if np.isfinite(theta_calc).all():
            omega_derived = np.gradient(np.deg2rad(theta_calc), timestamps)
        else:
            has_derived = False

    omega_combined = omega_raw.copy()
    if has_derived:
        missing_mask = ~np.isfinite(omega_combined)
        omega_combined[missing_mask] = omega_derived[missing_mask]

    if np.isfinite(omega_combined).sum() < 2 and has_derived:
        omega_combined = omega_derived.copy()

    if np.isfinite(omega_combined).sum() >= 1:
        omega_combined = _interpolate_finite_series(omega_combined)

    if np.isfinite(omega_combined).sum() == 0:
        omega_combined = np.zeros(len(segment_df), dtype=float)
        return omega_combined, "unavailable_assumed_zero"

    if has_raw and has_derived and np.isfinite(omega_raw).sum() < len(omega_raw):
        return omega_combined, f"omega_rad_s+{theta_source}"
    if has_raw:
        return omega_combined, "omega_rad_s"
    if has_derived:
        return omega_combined, f"derived_{theta_source}"
    return omega_combined, "unavailable_assumed_zero"


def _finite_max(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmax(finite))


def _finite_mean(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmean(finite))


def _finite_min(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.nanmin(finite))


def _last_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite[-1])


def compute_segment_energy(
    segment_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    y_mapping_profile: YMappingProfile,
    energy_spec: EnergySourceSpec,
) -> Tuple[pd.DataFrame, SegmentEnergySummary]:
    x_values = pd.to_numeric(segment_df[x_col], errors="coerce").to_numpy(dtype=float)
    y_values = pd.to_numeric(segment_df[y_col], errors="coerce").to_numpy(dtype=float)

    y_up, y_physical_raw, y_clamped_ratio = map_y_to_physical_height_mm(
        y_values,
        y_mapping_profile,
    )

    (
        timestamps,
        x_raw,
        y_physical_raw,
        _,
        y_physical_calc,
        speed_mm_s,
    ) = compute_motion_kinematics(
        segment_df,
        x_values=x_values,
        y_values=y_physical_raw,
    )

    omega_rad_s, omega_source = resolve_segment_omega_rad_s(segment_df, timestamps)

    speed_m_s = speed_mm_s / 1000.0
    y_m = y_physical_calc / 1000.0

    ep_j = energy_spec.mass_kg * energy_spec.gravity_m_s2 * np.maximum(y_m, 0.0)
    ek_trans_j = 0.5 * energy_spec.mass_kg * np.square(speed_m_s)
    ek_rot_j = 0.5 * energy_spec.inertia_kg_m2 * np.square(omega_rad_s)
    e_total_j = ep_j + ek_trans_j + ek_rot_j

    ep_j_per_kg = ep_j / energy_spec.mass_kg
    ek_trans_j_per_kg = ek_trans_j / energy_spec.mass_kg
    ek_rot_j_per_kg = ek_rot_j / energy_spec.mass_kg
    e_total_j_per_kg = e_total_j / energy_spec.mass_kg

    enriched = segment_df.copy()
    enriched["y_up_mm"] = y_up
    enriched["y_physical_mm"] = y_physical_raw
    enriched["y_physical_mm_interp"] = y_physical_calc
    enriched["v_mm_s"] = speed_mm_s
    enriched["v_m_s"] = speed_m_s
    enriched["omega_rad_s_energy"] = omega_rad_s
    enriched["ep_j"] = ep_j
    enriched["ek_trans_j"] = ek_trans_j
    enriched["ek_rot_j"] = ek_rot_j
    enriched["e_total_j"] = e_total_j
    enriched["ep_j_per_kg"] = ep_j_per_kg
    enriched["ek_trans_j_per_kg"] = ek_trans_j_per_kg
    enriched["ek_rot_j_per_kg"] = ek_rot_j_per_kg
    enriched["e_total_j_per_kg"] = e_total_j_per_kg

    summary = SegmentEnergySummary(
        omega_source=omega_source,
        y_clamped_ratio=y_clamped_ratio,
        y_min_mm=_finite_min(y_physical_calc),
        y_max_mm=_finite_max(y_physical_calc),
        potential_j_peak=_finite_max(ep_j),
        trans_j_peak=_finite_max(ek_trans_j),
        rot_j_peak=_finite_max(ek_rot_j),
        total_j_peak=_finite_max(e_total_j),
        potential_j_mean=_finite_mean(ep_j),
        trans_j_mean=_finite_mean(ek_trans_j),
        rot_j_mean=_finite_mean(ek_rot_j),
        total_j_mean=_finite_mean(e_total_j),
        total_j_final=_last_finite(e_total_j),
        potential_j_kg_peak=_finite_max(ep_j_per_kg),
        trans_j_kg_peak=_finite_max(ek_trans_j_per_kg),
        rot_j_kg_peak=_finite_max(ek_rot_j_per_kg),
        total_j_kg_peak=_finite_max(e_total_j_per_kg),
        potential_j_kg_mean=_finite_mean(ep_j_per_kg),
        trans_j_kg_mean=_finite_mean(ek_trans_j_per_kg),
        rot_j_kg_mean=_finite_mean(ek_rot_j_per_kg),
        total_j_kg_mean=_finite_mean(e_total_j_per_kg),
        total_j_kg_final=_last_finite(e_total_j_per_kg),
    )

    return enriched, summary


def compute_delta_theta_deg(segment_df: pd.DataFrame) -> Tuple[float, str]:
    if "theta_unwrapped" in segment_df.columns:
        theta_unwrapped = pd.to_numeric(segment_df["theta_unwrapped"], errors="coerce").to_numpy(
            dtype=float
        )
        finite = theta_unwrapped[np.isfinite(theta_unwrapped)]
        if finite.size >= 2:
            delta = float(np.nanmax(finite) - np.nanmin(finite))
            return max(0.0, delta), "theta_unwrapped"

    tracks = build_angle_tracks(segment_df)
    fallback_tracks: List[Tuple[str, np.ndarray]] = []

    if tracks.from_theta_unwrapped_available:
        fallback_tracks.append(("derived_theta_unwrapped", tracks.from_theta_unwrapped))
    if tracks.from_theta_available:
        fallback_tracks.append(("derived_theta", tracks.from_theta))

    for label, track in fallback_tracks:
        finite = track[np.isfinite(track)]
        if finite.size >= 2:
            delta = float(np.nanmax(finite) - np.nanmin(finite))
            return max(0.0, delta), label

    return 0.0, "unavailable"


def compute_duration_seconds(segment_df: pd.DataFrame) -> float:
    timestamps = pd.to_numeric(segment_df["timestamp"], errors="coerce").to_numpy(dtype=float)
    finite = timestamps[np.isfinite(timestamps)]
    if finite.size < 2:
        return float("nan")
    return max(0.0, float(finite[-1] - finite[0]))


def compute_x_displacement(segment_df: pd.DataFrame) -> float:
    for x_col in ("x_mm", "x"):
        if x_col not in segment_df.columns:
            continue

        x_values = pd.to_numeric(segment_df[x_col], errors="coerce").to_numpy(dtype=float)
        finite = x_values[np.isfinite(x_values)]
        if finite.size >= 2:
            return max(0.0, float(finite[-1] - finite[0]))

    return 0.0


def rank_segments(
    segments: Sequence[pd.DataFrame],
    ranking_mode: str,
    theta_source_label: Optional[str] = None,
) -> List[RankedSegment]:
    items = []
    for source_index, segment_df in enumerate(segments, start=1):
        delta_theta_deg, delta_source = compute_delta_theta_deg(segment_df)
        turns = delta_theta_deg / 360.0
        x_displacement = compute_x_displacement(segment_df)

        if ranking_mode == "x_displacement":
            ranking_value = x_displacement
            ranking_source = "x_displacement"
        else:
            ranking_value = delta_theta_deg
            ranking_source = "delta_theta_deg"

        frame_start = int(segment_df["frame_index"].iloc[0])
        frame_end = int(segment_df["frame_index"].iloc[-1])
        items.append(
            {
                "source_index": source_index,
                "ranking_value": ranking_value,
                "ranking_source": ranking_source,
                "delta_theta_deg": delta_theta_deg,
                "turns": turns,
                "delta_source": delta_source,
                "theta_source": theta_source_label or delta_source,
                "frame_start": frame_start,
                "frame_end": frame_end,
                "duration_s": compute_duration_seconds(segment_df),
                "rows": int(len(segment_df)),
                "segment_df": segment_df,
            }
        )

    items.sort(key=lambda item: (-item["ranking_value"], item["source_index"]))

    ranked: List[RankedSegment] = []
    for rank, item in enumerate(items, start=1):
        ranked.append(
            RankedSegment(
                rank=rank,
                source_index=int(item["source_index"]),
                delta_theta_deg=float(item["delta_theta_deg"]),
                turns=float(item["turns"]),
                delta_source=str(item["delta_source"]),
                frame_start=int(item["frame_start"]),
                frame_end=int(item["frame_end"]),
                duration_s=float(item["duration_s"]),
                rows=int(item["rows"]),
                ranking_value=float(item["ranking_value"]),
                ranking_source=str(item["ranking_source"]),
                theta_source=str(item["theta_source"]),
                segment_df=item["segment_df"],
            )
        )

    return ranked


def save_ranked_cleaned_segments(
    ranked_segments: Sequence[RankedSegment],
    output_dir: Path,
    source_slug: str,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_paths: List[Path] = []
    for ranked_segment in ranked_segments:
        output_path = output_dir / f"{source_slug}_cleaned_segment_{ranked_segment.rank:02d}.csv"
        ranked_segment.segment_df.to_csv(output_path, index=False, encoding="utf-8")
        saved_paths.append(output_path.resolve())

    return saved_paths


def extract_theta_track_for_plot(segment_df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    theta_unwrapped = pd.to_numeric(segment_df["theta_unwrapped"], errors="coerce").to_numpy(dtype=float)
    if np.isfinite(theta_unwrapped).sum() >= 2:
        return theta_unwrapped, "theta_unwrapped"

    tracks = build_angle_tracks(segment_df)
    if tracks.from_theta_unwrapped_available:
        return tracks.from_theta_unwrapped, "derived_theta_unwrapped"
    if tracks.from_theta_available:
        return tracks.from_theta, "derived_theta"

    return np.full(len(segment_df), np.nan, dtype=float), "unavailable"


def parse_cleaned_segment_filename(csv_path: Path) -> Tuple[str, int]:
    match = CLEANED_SEGMENT_PATTERN.match(csv_path.name)
    if not match:
        raise ValueError(
            "Invalid cleaned-segment filename. Expected "
            "<source>_cleaned_segment_XX.csv: "
            f"{csv_path.name}"
        )

    source_slug = match.group(1)
    rank = int(match.group(2))
    return source_slug, rank


def discover_cleaned_segment_csvs(processed_dir: Path) -> List[Tuple[str, int, Path]]:
    matches: List[Tuple[str, int, Path]] = []
    for csv_path in processed_dir.glob("*_cleaned_segment_*.csv"):
        if not csv_path.is_file():
            continue
        source_slug, rank = parse_cleaned_segment_filename(csv_path)
        matches.append((source_slug, rank, csv_path.resolve()))

    matches.sort(key=lambda item: (item[0], item[1]))
    if not matches:
        raise FileNotFoundError(
            "No cleaned segment CSV files found in processed directory: "
            f"{processed_dir}"
        )

    return matches


def build_ranked_segment_from_cleaned_csv(csv_path: Path, rank: int) -> RankedSegment:
    segment_df = load_and_prepare_csv(csv_path)
    frame_values = pd.to_numeric(segment_df["frame_index"], errors="coerce").to_numpy(dtype=float)
    finite_frames = frame_values[np.isfinite(frame_values)]

    if finite_frames.size >= 1:
        frame_start = int(finite_frames[0])
        frame_end = int(finite_frames[-1])
    else:
        frame_start = 0
        frame_end = max(0, len(segment_df) - 1)

    delta_theta_deg, delta_source = compute_delta_theta_deg(segment_df)
    turns = delta_theta_deg / 360.0
    x_displacement = compute_x_displacement(segment_df)

    return RankedSegment(
        rank=rank,
        source_index=rank,
        delta_theta_deg=float(delta_theta_deg),
        turns=float(turns),
        delta_source=delta_source,
        frame_start=frame_start,
        frame_end=frame_end,
        duration_s=compute_duration_seconds(segment_df),
        rows=int(len(segment_df)),
        ranking_value=float(delta_theta_deg),
        ranking_source="delta_theta_deg",
        theta_source=delta_source,
        segment_df=segment_df,
    )


def plot_segment_trajectory(
    ranked_segment: RankedSegment,
    source_slug: str,
    plots_output_dir: Path,
) -> Path:
    plots_output_dir.mkdir(parents=True, exist_ok=True)
    output_path = plots_output_dir / f"{source_slug}_segment_{ranked_segment.rank:02d}_trajectory.png"

    try:
        x_col, y_col = choose_position_columns(ranked_segment.segment_df)
        timestamps, x_raw, y_raw, speed = compute_segment_speed(
            ranked_segment.segment_df,
            x_col=x_col,
            y_col=y_col,
        )
        y_display = to_up_positive_display_y(y_raw)
        theta_track, theta_source = extract_theta_track_for_plot(ranked_segment.segment_df)

        fig, (ax_traj, ax_theta) = plt.subplots(2, 1, figsize=(11, 8), constrained_layout=True)

        valid_xy = np.isfinite(x_raw) & np.isfinite(y_display)
        if int(valid_xy.sum()) >= 2:
            ax_traj.plot(
                x_raw[valid_xy],
                y_display[valid_xy],
                color="#9CB7FF",
                linewidth=1.2,
                alpha=0.5,
            )

        valid_speed = valid_xy & np.isfinite(speed)
        if int(valid_speed.sum()) >= 2:
            scatter = ax_traj.scatter(
                x_raw[valid_speed],
                y_display[valid_speed],
                c=speed[valid_speed],
                cmap="viridis",
                s=16,
            )
            colorbar = fig.colorbar(scatter, ax=ax_traj, pad=0.015)
            colorbar.set_label("Speed (units/s)")
        elif int(valid_xy.sum()) > 0:
            ax_traj.scatter(x_raw[valid_xy], y_display[valid_xy], color="#4C72B0", s=14)

        ax_traj.set_title(
            f"{source_slug} | Segment {ranked_segment.rank:02d} | "
            + (
                f"Delta theta={ranked_segment.delta_theta_deg:.2f} deg"
                if ranked_segment.ranking_source == "delta_theta_deg"
                else f"X displacement={ranked_segment.ranking_value:.2f}"
            )
        )
        ax_traj.set_xlabel(x_col)
        ax_traj.set_ylabel(f"{y_col} (up-positive display)")
        ax_traj.grid(alpha=0.3, linestyle="--", linewidth=0.6)
        ax_traj.set_aspect("equal", adjustable="box")

        valid_theta = np.isfinite(theta_track)
        if int(valid_theta.sum()) >= 2:
            ax_theta.plot(timestamps, theta_track, color="#DD8452", linewidth=1.8)
        ax_theta.set_xlabel("Timestamp (s)")
        ax_theta.set_ylabel("Theta (deg)")
        ax_theta.set_title(f"Theta source: {theta_source}")
        ax_theta.grid(alpha=0.3, linestyle="--", linewidth=0.6)

        fig.savefig(output_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return output_path.resolve()
    except Exception as exc:
        fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            (
                f"Plot generation failed for {source_slug} segment {ranked_segment.rank:02d}.\n"
                f"Reason: {exc}"
            ),
            ha="center",
            va="center",
            fontsize=10,
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path.resolve()


def run_replot_cleaned_mode(
    project_root: Path,
    plots_output_dir: Path,
    args: argparse.Namespace,
) -> None:
    processed_dir = resolve_optional_dir(
        project_root=project_root,
        path_arg=args.processed_dir or args.output_dir,
        default_dir=default_processed_output_dir(project_root),
    )
    cleaned_segments = discover_cleaned_segment_csvs(processed_dir)

    print(f"Replot mode enabled. Processed directory: {processed_dir}")
    print(f"Discovered {len(cleaned_segments)} cleaned segment CSV files.")

    success = 0
    failed = 0

    for index, (source_slug, rank, csv_path) in enumerate(cleaned_segments, start=1):
        print("-" * 88)
        print(f"[{index}/{len(cleaned_segments)}] Replotting {csv_path.name}")
        try:
            ranked_segment = build_ranked_segment_from_cleaned_csv(csv_path, rank)
            plot_path = plot_segment_trajectory(
                ranked_segment=ranked_segment,
                source_slug=source_slug,
                plots_output_dir=plots_output_dir,
            )
            success += 1
            print(f"Saved trajectory plot: {plot_path}")
        except Exception as exc:
            failed += 1
            print(f"Failed to replot {csv_path.name}: {exc}")

    print("=" * 88)
    print("Replot summary")
    print(f" - cleaned segment CSV files: {len(cleaned_segments)}")
    print(f" - plots updated: {success}")
    print(f" - failures: {failed}")


def plot_source_summary(
    source_slug: str,
    ranked_segments: Sequence[RankedSegment],
    output_path: Path,
    diagnostic_lines: Optional[Sequence[str]] = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not ranked_segments:
        removed_by_y = 0.0
        rejected = 0.0
        candidates = 0.0

        if diagnostic_lines:
            for line in diagnostic_lines:
                text = str(line).strip()
                if text.startswith("Rows removed by Y bounds:"):
                    try:
                        removed_by_y = float(text.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif text.startswith("Rejected candidate segments:"):
                    try:
                        rejected = float(text.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif text.startswith("Candidate segments:"):
                    try:
                        candidates = float(text.split(":", 1)[1].strip())
                    except (ValueError, IndexError):
                        pass

        fig, (ax_info, ax_bar) = plt.subplots(
            1,
            2,
            figsize=(12, 5),
            constrained_layout=True,
            gridspec_kw={"width_ratios": [1.2, 1.8]},
        )

        ax_info.axis("off")
        info_lines = [
            f"{source_slug}",
            "No valid cleaned segments",
            "",
            f"Candidates: {int(candidates)}",
            f"Rejected: {int(rejected)}",
            f"Removed by Y bounds: {int(removed_by_y)}",
        ]
        if diagnostic_lines:
            for line in diagnostic_lines:
                text = str(line).strip()
                if text.startswith("Red gates:"):
                    info_lines.append(text)

        ax_info.text(
            0.04,
            0.95,
            "\n".join(info_lines),
            ha="left",
            va="top",
            fontsize=12,
            bbox={
                "facecolor": "#F6F2E8",
                "edgecolor": "#C9B27C",
                "boxstyle": "round,pad=0.6",
                "linewidth": 1.0,
            },
        )

        labels = ["Accepted", "Rejected", "Removed(Y)", "Candidates"]
        values = [0.0, rejected, removed_by_y, candidates]
        colors = ["#55A868", "#C44E52", "#8172B2", "#4C72B0"]

        bars = ax_bar.barh(labels, values, color=colors, alpha=0.9)
        max_value = max(values) if values else 0.0
        x_pad = max(1.0, 0.03 * max_value)
        ax_bar.set_xlim(0.0, max(1.0, max_value + 2.0 * x_pad))
        ax_bar.set_title("Filtering Diagnostics", fontsize=13)
        ax_bar.set_xlabel("Count")
        ax_bar.grid(axis="x", alpha=0.25, linestyle="--", linewidth=0.6)

        for bar, value in zip(bars, values):
            ax_bar.text(
                bar.get_width() + x_pad,
                bar.get_y() + bar.get_height() / 2.0,
                f"{int(value)}",
                va="center",
                ha="left",
                fontsize=11,
            )

        fig.savefig(output_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return output_path.resolve()

    ranks = np.array([item.rank for item in ranked_segments], dtype=float)
    ranking_source = ranked_segments[0].ranking_source
    ranking_values = np.array([item.ranking_value for item in ranked_segments], dtype=float)
    turns = np.array([item.turns for item in ranked_segments], dtype=float)
    durations = np.array([item.duration_s for item in ranked_segments], dtype=float)

    fig, (ax_delta, ax_aux) = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)

    ax_delta.bar(ranks, ranking_values, color="#4C72B0", alpha=0.9)
    if ranking_source == "x_displacement":
        ax_delta.set_title(f"{source_slug} | Ranked by X Displacement (descending)")
        ax_delta.set_xlabel("Rank (1 = largest X displacement)")
        ax_delta.set_ylabel("X displacement")
    else:
        ax_delta.set_title(f"{source_slug} | Ranked by Delta Theta (descending)")
        ax_delta.set_xlabel("Rank (1 = largest Delta Theta)")
        ax_delta.set_ylabel("Delta Theta (deg)")
    ax_delta.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)

    ax_aux.plot(ranks, turns, marker="o", color="#DD8452", label="Turns")
    ax_aux.plot(ranks, durations, marker="s", color="#55A868", label="Duration (s)")
    ax_aux.set_xlabel("Rank")
    ax_aux.set_ylabel("Value")
    ax_aux.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax_aux.legend(loc="best")

    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return output_path.resolve()


def build_analysis_dataframe(
    source_csv: Path,
    ranked_segments: Sequence[RankedSegment],
    cleaned_csv_paths: Sequence[Path],
    segment_plot_paths: Sequence[Path],
    summary_plot_path: Path,
    min_roll_deg_used: float,
    red_mode: bool,
    accepted_segments: int,
    rejected_segments: int,
    removed_rows_by_y: int,
    energy_spec: Optional[EnergySourceSpec] = None,
    y_mapping_profile: Optional[YMappingProfile] = None,
    position_columns: Optional[Tuple[str, str]] = None,
    energy_summaries: Optional[Sequence[SegmentEnergySummary]] = None,
) -> pd.DataFrame:
    columns = [
        "source_csv",
        "rank",
        "source_segment_index",
        "ranking_source",
        "ranking_value",
        "delta_theta_deg",
        "turns",
        "delta_source",
        "theta_source",
        "frame_start",
        "frame_end",
        "duration_s",
        "rows",
        "min_roll_deg_used",
        "red_mode",
        "accepted_segments",
        "rejected_segments",
        "removed_rows_by_y",
        "result_note",
        "energy_thickness_mm",
        "energy_mass_g",
        "energy_height_mm",
        "energy_width_mm",
        "energy_y_low_mm",
        "energy_y_top_mm",
        "energy_gravity_m_s2",
        "energy_inertia_kg_m2",
        "energy_param_source",
        "energy_position_x_col",
        "energy_position_y_col",
        "energy_y_map_bottom_ref",
        "energy_y_map_q02_up_mm",
        "energy_y_map_q98_up_mm",
        "energy_omega_source",
        "energy_y_clamped_ratio",
        "energy_y_min_mm",
        "energy_y_max_mm",
        "energy_potential_j_peak",
        "energy_trans_j_peak",
        "energy_rot_j_peak",
        "energy_total_j_peak",
        "energy_potential_j_mean",
        "energy_trans_j_mean",
        "energy_rot_j_mean",
        "energy_total_j_mean",
        "energy_total_j_final",
        "energy_potential_j_kg_peak",
        "energy_trans_j_kg_peak",
        "energy_rot_j_kg_peak",
        "energy_total_j_kg_peak",
        "energy_potential_j_kg_mean",
        "energy_trans_j_kg_mean",
        "energy_rot_j_kg_mean",
        "energy_total_j_kg_mean",
        "energy_total_j_kg_final",
        "cleaned_csv_path",
        "segment_plot_path",
        "summary_plot_path",
    ]

    x_col = position_columns[0] if position_columns is not None else ""
    y_col = position_columns[1] if position_columns is not None else ""

    records = []
    for index, ranked_segment in enumerate(ranked_segments):
        cleaned_path = cleaned_csv_paths[index] if index < len(cleaned_csv_paths) else None
        segment_plot_path = segment_plot_paths[index] if index < len(segment_plot_paths) else None
        energy_summary = energy_summaries[index] if energy_summaries and index < len(energy_summaries) else None

        if energy_spec is None:
            thickness_mm = float("nan")
            mass_g = float("nan")
            height_mm = float("nan")
            width_mm = float("nan")
            y_low_mm = float("nan")
            y_top_mm = float("nan")
            gravity_m_s2 = float("nan")
            inertia_kg_m2 = float("nan")
            param_source = ""
        else:
            thickness_mm = energy_spec.thickness_mm
            mass_g = energy_spec.mass_g
            height_mm = energy_spec.height_mm
            width_mm = energy_spec.width_mm
            y_low_mm = energy_spec.y_low_mm
            y_top_mm = energy_spec.y_top_mm
            gravity_m_s2 = energy_spec.gravity_m_s2
            inertia_kg_m2 = energy_spec.inertia_kg_m2
            param_source = energy_spec.config_source

        if y_mapping_profile is None:
            y_bottom_ref = float("nan")
            y_q02 = float("nan")
            y_q98 = float("nan")
        else:
            y_bottom_ref = y_mapping_profile.y_bottom_ref
            y_q02 = y_mapping_profile.q02_up_mm
            y_q98 = y_mapping_profile.q98_up_mm

        if energy_summary is None:
            omega_source = ""
            y_clamped_ratio = float("nan")
            y_min_mm = float("nan")
            y_max_mm = float("nan")
            potential_j_peak = float("nan")
            trans_j_peak = float("nan")
            rot_j_peak = float("nan")
            total_j_peak = float("nan")
            potential_j_mean = float("nan")
            trans_j_mean = float("nan")
            rot_j_mean = float("nan")
            total_j_mean = float("nan")
            total_j_final = float("nan")
            potential_j_kg_peak = float("nan")
            trans_j_kg_peak = float("nan")
            rot_j_kg_peak = float("nan")
            total_j_kg_peak = float("nan")
            potential_j_kg_mean = float("nan")
            trans_j_kg_mean = float("nan")
            rot_j_kg_mean = float("nan")
            total_j_kg_mean = float("nan")
            total_j_kg_final = float("nan")
        else:
            omega_source = energy_summary.omega_source
            y_clamped_ratio = energy_summary.y_clamped_ratio
            y_min_mm = energy_summary.y_min_mm
            y_max_mm = energy_summary.y_max_mm
            potential_j_peak = energy_summary.potential_j_peak
            trans_j_peak = energy_summary.trans_j_peak
            rot_j_peak = energy_summary.rot_j_peak
            total_j_peak = energy_summary.total_j_peak
            potential_j_mean = energy_summary.potential_j_mean
            trans_j_mean = energy_summary.trans_j_mean
            rot_j_mean = energy_summary.rot_j_mean
            total_j_mean = energy_summary.total_j_mean
            total_j_final = energy_summary.total_j_final
            potential_j_kg_peak = energy_summary.potential_j_kg_peak
            trans_j_kg_peak = energy_summary.trans_j_kg_peak
            rot_j_kg_peak = energy_summary.rot_j_kg_peak
            total_j_kg_peak = energy_summary.total_j_kg_peak
            potential_j_kg_mean = energy_summary.potential_j_kg_mean
            trans_j_kg_mean = energy_summary.trans_j_kg_mean
            rot_j_kg_mean = energy_summary.rot_j_kg_mean
            total_j_kg_mean = energy_summary.total_j_kg_mean
            total_j_kg_final = energy_summary.total_j_kg_final

        records.append(
            {
                "source_csv": source_csv.name,
                "rank": ranked_segment.rank,
                "source_segment_index": ranked_segment.source_index,
                "ranking_source": ranked_segment.ranking_source,
                "ranking_value": ranked_segment.ranking_value,
                "delta_theta_deg": ranked_segment.delta_theta_deg,
                "turns": ranked_segment.turns,
                "delta_source": ranked_segment.delta_source,
                "theta_source": ranked_segment.theta_source,
                "frame_start": ranked_segment.frame_start,
                "frame_end": ranked_segment.frame_end,
                "duration_s": ranked_segment.duration_s,
                "rows": ranked_segment.rows,
                "min_roll_deg_used": min_roll_deg_used,
                "red_mode": bool(red_mode),
                "accepted_segments": int(accepted_segments),
                "rejected_segments": int(rejected_segments),
                "removed_rows_by_y": int(removed_rows_by_y),
                "result_note": "accepted_segment",
                "energy_thickness_mm": thickness_mm,
                "energy_mass_g": mass_g,
                "energy_height_mm": height_mm,
                "energy_width_mm": width_mm,
                "energy_y_low_mm": y_low_mm,
                "energy_y_top_mm": y_top_mm,
                "energy_gravity_m_s2": gravity_m_s2,
                "energy_inertia_kg_m2": inertia_kg_m2,
                "energy_param_source": param_source,
                "energy_position_x_col": x_col,
                "energy_position_y_col": y_col,
                "energy_y_map_bottom_ref": y_bottom_ref,
                "energy_y_map_q02_up_mm": y_q02,
                "energy_y_map_q98_up_mm": y_q98,
                "energy_omega_source": omega_source,
                "energy_y_clamped_ratio": y_clamped_ratio,
                "energy_y_min_mm": y_min_mm,
                "energy_y_max_mm": y_max_mm,
                "energy_potential_j_peak": potential_j_peak,
                "energy_trans_j_peak": trans_j_peak,
                "energy_rot_j_peak": rot_j_peak,
                "energy_total_j_peak": total_j_peak,
                "energy_potential_j_mean": potential_j_mean,
                "energy_trans_j_mean": trans_j_mean,
                "energy_rot_j_mean": rot_j_mean,
                "energy_total_j_mean": total_j_mean,
                "energy_total_j_final": total_j_final,
                "energy_potential_j_kg_peak": potential_j_kg_peak,
                "energy_trans_j_kg_peak": trans_j_kg_peak,
                "energy_rot_j_kg_peak": rot_j_kg_peak,
                "energy_total_j_kg_peak": total_j_kg_peak,
                "energy_potential_j_kg_mean": potential_j_kg_mean,
                "energy_trans_j_kg_mean": trans_j_kg_mean,
                "energy_rot_j_kg_mean": rot_j_kg_mean,
                "energy_total_j_kg_mean": total_j_kg_mean,
                "energy_total_j_kg_final": total_j_kg_final,
                "cleaned_csv_path": str(cleaned_path) if cleaned_path is not None else "",
                "segment_plot_path": str(segment_plot_path) if segment_plot_path is not None else "",
                "summary_plot_path": str(summary_plot_path),
            }
        )

    if not records:
        if energy_spec is None:
            thickness_mm = float("nan")
            mass_g = float("nan")
            height_mm = float("nan")
            width_mm = float("nan")
            y_low_mm = float("nan")
            y_top_mm = float("nan")
            gravity_m_s2 = float("nan")
            inertia_kg_m2 = float("nan")
            param_source = ""
        else:
            thickness_mm = energy_spec.thickness_mm
            mass_g = energy_spec.mass_g
            height_mm = energy_spec.height_mm
            width_mm = energy_spec.width_mm
            y_low_mm = energy_spec.y_low_mm
            y_top_mm = energy_spec.y_top_mm
            gravity_m_s2 = energy_spec.gravity_m_s2
            inertia_kg_m2 = energy_spec.inertia_kg_m2
            param_source = energy_spec.config_source

        if y_mapping_profile is None:
            y_bottom_ref = float("nan")
            y_q02 = float("nan")
            y_q98 = float("nan")
        else:
            y_bottom_ref = y_mapping_profile.y_bottom_ref
            y_q02 = y_mapping_profile.q02_up_mm
            y_q98 = y_mapping_profile.q98_up_mm

        records.append(
            {
                "source_csv": source_csv.name,
                "rank": 0,
                "source_segment_index": 0,
                "ranking_source": "none",
                "ranking_value": float("nan"),
                "delta_theta_deg": float("nan"),
                "turns": float("nan"),
                "delta_source": "",
                "theta_source": "",
                "frame_start": -1,
                "frame_end": -1,
                "duration_s": float("nan"),
                "rows": 0,
                "min_roll_deg_used": min_roll_deg_used,
                "red_mode": bool(red_mode),
                "accepted_segments": int(accepted_segments),
                "rejected_segments": int(rejected_segments),
                "removed_rows_by_y": int(removed_rows_by_y),
                "result_note": "no_segment_after_filtering",
                "energy_thickness_mm": thickness_mm,
                "energy_mass_g": mass_g,
                "energy_height_mm": height_mm,
                "energy_width_mm": width_mm,
                "energy_y_low_mm": y_low_mm,
                "energy_y_top_mm": y_top_mm,
                "energy_gravity_m_s2": gravity_m_s2,
                "energy_inertia_kg_m2": inertia_kg_m2,
                "energy_param_source": param_source,
                "energy_position_x_col": x_col,
                "energy_position_y_col": y_col,
                "energy_y_map_bottom_ref": y_bottom_ref,
                "energy_y_map_q02_up_mm": y_q02,
                "energy_y_map_q98_up_mm": y_q98,
                "energy_omega_source": "",
                "energy_y_clamped_ratio": float("nan"),
                "energy_y_min_mm": float("nan"),
                "energy_y_max_mm": float("nan"),
                "energy_potential_j_peak": float("nan"),
                "energy_trans_j_peak": float("nan"),
                "energy_rot_j_peak": float("nan"),
                "energy_total_j_peak": float("nan"),
                "energy_potential_j_mean": float("nan"),
                "energy_trans_j_mean": float("nan"),
                "energy_rot_j_mean": float("nan"),
                "energy_total_j_mean": float("nan"),
                "energy_total_j_final": float("nan"),
                "energy_potential_j_kg_peak": float("nan"),
                "energy_trans_j_kg_peak": float("nan"),
                "energy_rot_j_kg_peak": float("nan"),
                "energy_total_j_kg_peak": float("nan"),
                "energy_potential_j_kg_mean": float("nan"),
                "energy_trans_j_kg_mean": float("nan"),
                "energy_rot_j_kg_mean": float("nan"),
                "energy_total_j_kg_mean": float("nan"),
                "energy_total_j_kg_final": float("nan"),
                "cleaned_csv_path": "",
                "segment_plot_path": "",
                "summary_plot_path": str(summary_plot_path),
            }
        )

    return pd.DataFrame.from_records(records, columns=columns)


def process_single_csv_file(
    csv_path: Path,
    output_dir: Path,
    plots_output_dir: Path,
    args: argparse.Namespace,
    project_root: Path,
    energy_spec_cache: Optional[Dict[str, EnergySourceSpec]] = None,
) -> FileProcessingResult:
    source_slug = sanitize_stem(csv_path.stem)
    analysis_csv_path = output_dir / f"{source_slug}_analysis_metrics.csv"
    summary_plot_path = plots_output_dir / f"{source_slug}_segments_summary.png"

    df = load_and_prepare_csv(csv_path)
    selection = run_cleaning_with_mode(df=df, args=args, csv_path=csv_path)

    segments = selection.segments
    logs = selection.logs
    removed_by_y = selection.removed_by_y
    rejected_segments = selection.rejected_segments
    theta_source_label: Optional[str] = None

    if selection.red_mode and segments:
        reconstructed_segments: List[pd.DataFrame] = []
        reconstruct_sources: List[str] = []
        for index, segment in enumerate(segments, start=1):
            reconstructed, source_label = reconstruct_theta_from_yx(
                segment_df=segment,
                wobble_alpha=args.red_wobble_smoothing_alpha,
                forward_dx_threshold=args.red_forward_dx_threshold,
                cycle_reset_threshold_deg=args.red_cycle_reset_threshold_deg,
            )
            reconstructed_segments.append(reconstructed)
            reconstruct_sources.append(source_label)
            logs.append(
                f"Candidate {index}: Red theta reconstruction source={source_label}, rows={len(reconstructed)}"
            )

        segments = reconstructed_segments
        unique_sources = sorted(set(reconstruct_sources))
        if len(unique_sources) == 1:
            theta_source_label = unique_sources[0]
        else:
            theta_source_label = "mixed_red_reconstruction"

    print(f"Input CSV: {csv_path}")
    print(f"Total rows: {len(df)}")
    print("Angle quality summary:")
    for line in summarize_angle_quality(df):
        print(line)
    if selection.red_mode:
        print(
            "Red no-theta mode enabled. "
            + f"Y gates: start<= {args.red_start_y_max:.3f}, end>= {args.red_end_y_min:.3f}."
        )
        print(
            "Red theta reconstruction: "
            + f"wobble_alpha={args.red_wobble_smoothing_alpha:.3f}, "
            + f"forward_dx_threshold={args.red_forward_dx_threshold:.3f}, "
            + f"cycle_reset_threshold_deg={args.red_cycle_reset_threshold_deg:.3f}"
        )
    else:
        print(
            "Min roll threshold used: "
            + f"{selection.min_roll_deg_used:.3f} deg (requested {args.min_roll_deg:.3f} deg)"
        )
        if selection.fallback_attempts:
            print("Auto-relax attempts (min-roll-deg -> accepted segments):")
            for threshold, accepted_count in selection.fallback_attempts:
                print(f" - {threshold:.3f} -> {accepted_count}")

    print(f"Rows removed by Y hard bounds [{args.y_min}, {args.y_max}]: {removed_by_y}")
    print(f"Rejected candidate segments: {rejected_segments}")
    for line in logs:
        print(line)

    ranking_mode = "x_displacement" if selection.red_mode else "delta_theta"
    ranked_segments = rank_segments(
        segments,
        ranking_mode=ranking_mode,
        theta_source_label=theta_source_label,
    )

    energy_spec: Optional[EnergySourceSpec] = None
    y_mapping_profile: Optional[YMappingProfile] = None
    energy_position_columns: Optional[Tuple[str, str]] = None
    energy_summaries: List[SegmentEnergySummary] = []

    if ranked_segments:
        if energy_spec_cache is None:
            energy_spec_cache = {}

        energy_spec = energy_spec_cache.get(source_slug)
        if energy_spec is None:
            energy_spec = resolve_energy_source_spec(
                project_root=project_root,
                source_slug=source_slug,
                source_csv_name=csv_path.name,
            )
            energy_spec_cache[source_slug] = energy_spec

        energy_position_columns = choose_position_columns(df)
        energy_x_col, energy_y_col = energy_position_columns
        y_mapping_profile = build_source_y_mapping_profile(
            source_df=df,
            y_col=energy_y_col,
            energy_spec=energy_spec,
        )

        print(
            "Energy params: "
            + f"thickness_mm={energy_spec.thickness_mm:.3f}, "
            + f"mass_g={energy_spec.mass_g:.3f}, "
            + f"y_low_mm={energy_spec.y_low_mm:.3f}, "
            + f"y_top_mm={energy_spec.y_top_mm:.3f}, "
            + f"config_source={energy_spec.config_source}"
        )
        print(
            "Energy Y mapping profile: "
            + f"column={energy_y_col}, "
            + f"bottom_ref={y_mapping_profile.y_bottom_ref:.3f}, "
            + f"q02_up={y_mapping_profile.q02_up_mm:.3f}, "
            + f"q98_up={y_mapping_profile.q98_up_mm:.3f}"
        )

        for ranked_segment in ranked_segments:
            enriched_df, energy_summary = compute_segment_energy(
                segment_df=ranked_segment.segment_df,
                x_col=energy_x_col,
                y_col=energy_y_col,
                y_mapping_profile=y_mapping_profile,
                energy_spec=energy_spec,
            )
            ranked_segment.segment_df = enriched_df
            energy_summaries.append(energy_summary)

    cleaned_csv_paths = save_ranked_cleaned_segments(
        ranked_segments=ranked_segments,
        output_dir=output_dir,
        source_slug=source_slug,
    )

    candidate_count: Optional[int] = None
    candidate_matcher = re.compile(r"Step2 segmentation:\s*(\d+)\s+candidate segments\.")
    for log_line in logs:
        matched = candidate_matcher.search(log_line)
        if matched:
            candidate_count = int(matched.group(1))
            break

    summary_diagnostics: List[str] = []
    if not ranked_segments:
        summary_diagnostics.append(f"Rows removed by Y bounds: {removed_by_y}")
        summary_diagnostics.append(f"Rejected candidate segments: {rejected_segments}")
        if candidate_count is not None:
            summary_diagnostics.append(f"Candidate segments: {candidate_count}")
        if selection.red_mode:
            summary_diagnostics.append(
                "Red gates: "
                + f"start<= {args.red_start_y_max:.3f}, end>= {args.red_end_y_min:.3f}"
            )

    generated_summary_plot = plot_source_summary(
        source_slug=source_slug,
        ranked_segments=ranked_segments,
        output_path=summary_plot_path,
        diagnostic_lines=summary_diagnostics,
    )

    segment_plot_paths: List[Path] = []
    for ranked_segment in ranked_segments:
        segment_plot_paths.append(
            plot_segment_trajectory(
                ranked_segment=ranked_segment,
                source_slug=source_slug,
                plots_output_dir=plots_output_dir,
            )
        )

    analysis_df = build_analysis_dataframe(
        source_csv=csv_path,
        ranked_segments=ranked_segments,
        cleaned_csv_paths=cleaned_csv_paths,
        segment_plot_paths=segment_plot_paths,
        summary_plot_path=generated_summary_plot,
        min_roll_deg_used=selection.min_roll_deg_used,
        red_mode=selection.red_mode,
        accepted_segments=len(ranked_segments),
        rejected_segments=rejected_segments,
        removed_rows_by_y=removed_by_y,
        energy_spec=energy_spec,
        y_mapping_profile=y_mapping_profile,
        position_columns=energy_position_columns,
        energy_summaries=energy_summaries,
    )
    analysis_df.to_csv(analysis_csv_path, index=False, encoding="utf-8")

    print(f"Accepted segments after ranking: {len(ranked_segments)}")
    if ranked_segments:
        ranking_source = ranked_segments[0].ranking_source
        ranking_header = (
            "Ranked by x displacement (descending):"
            if ranking_source == "x_displacement"
            else "Ranked by delta theta (descending):"
        )
        print(ranking_header)
        for ranked_segment in ranked_segments:
            print(
                " - rank="
                + f"{ranked_segment.rank:02d}, "
                + f"ranking_source={ranked_segment.ranking_source}, "
                + f"ranking_value={ranked_segment.ranking_value:.3f}, "
                + f"delta_theta={ranked_segment.delta_theta_deg:.3f} deg, "
                + f"turns={ranked_segment.turns:.4f}, "
                + f"theta_source={ranked_segment.theta_source}, "
                + f"rows={ranked_segment.rows}"
            )

    print(f"Saved cleaned CSV files: {len(cleaned_csv_paths)}")
    for path in cleaned_csv_paths:
        print(f" - {path}")
    print(f"Saved analysis CSV: {analysis_csv_path}")
    print(f"Saved summary plot: {generated_summary_plot}")
    print(f"Saved segment plots: {len(segment_plot_paths)}")
    if not ranked_segments:
        if selection.red_mode:
            print(
                "No segment passed final filters in Red no-theta mode. "
                "Consider relaxing --red-start-y-max and/or lowering --red-end-y-min."
            )
        else:
            print(
                "No segment passed final filters. "
                "Check angle span above and consider lowering --min-roll-deg or enabling "
                "--auto-relax-roll-threshold for exploratory runs."
            )

    return FileProcessingResult(
        source_csv=csv_path,
        accepted_segments=len(ranked_segments),
        rejected_segments=rejected_segments,
        removed_rows_by_y=removed_by_y,
        cleaned_csv_paths=cleaned_csv_paths,
        analysis_csv_path=analysis_csv_path.resolve(),
        summary_plot_path=generated_summary_plot,
        segment_plot_paths=segment_plot_paths,
    )


def print_batch_summary(results: Sequence[FileProcessingResult]) -> None:
    total_files = len(results)
    failed_files = sum(1 for result in results if result.failed)
    total_accepted = sum(result.accepted_segments for result in results)
    total_rejected = sum(result.rejected_segments for result in results)
    total_cleaned = sum(len(result.cleaned_csv_paths) for result in results)

    print("=" * 88)
    print("Batch summary")
    print(f" - source CSV files: {total_files}")
    print(f" - failed files: {failed_files}")
    print(f" - accepted segments: {total_accepted}")
    print(f" - rejected segments: {total_rejected}")
    print(f" - cleaned CSV outputs: {total_cleaned}")

    for result in results:
        status = "FAILED" if result.failed else "OK"
        detail = (
            f"accepted={result.accepted_segments}, "
            f"rejected={result.rejected_segments}, "
            f"cleaned_csv={len(result.cleaned_csv_paths)}"
        )
        if result.failed:
            detail += f", reason={result.failure_reason}"
        print(f" - {result.source_csv.name}: {status} ({detail})")


def run_single_mode(
    project_root: Path,
    output_dir: Path,
    plots_output_dir: Path,
    args: argparse.Namespace,
) -> None:
    csv_input = args.csv_path
    if csv_input is None:
        csv_input = input(
            "Enter CSV path (or press Enter to auto-detect from video/config): "
        ).strip().strip('"')
        if not csv_input:
            csv_input = None

    csv_path, video_path, used_legacy_csv, used_explicit_csv = resolve_csv_path(
        project_root=project_root,
        explicit_csv_path=csv_input,
        explicit_video_path=args.video_path,
    )

    energy_spec_cache: Dict[str, EnergySourceSpec] = {}

    process_single_csv_file(
        csv_path=csv_path,
        output_dir=output_dir,
        plots_output_dir=plots_output_dir,
        args=args,
        project_root=project_root,
        energy_spec_cache=energy_spec_cache,
    )

    if video_path is not None:
        print(f"Video source: {video_path}")
    if used_explicit_csv:
        print("CSV source mode: explicit --csv-path")
    elif used_legacy_csv:
        print("CSV source mode: legacy project-root *_raw_data.csv fallback")


def run_batch_mode(
    project_root: Path,
    output_dir: Path,
    plots_output_dir: Path,
    args: argparse.Namespace,
) -> None:
    raw_dir = resolve_raw_input_dir(project_root, args.raw_dir)
    csv_paths = discover_raw_csv_files(raw_dir)

    print(f"Batch mode enabled. Raw directory: {raw_dir}")
    print(f"Discovered {len(csv_paths)} CSV files.")

    results: List[FileProcessingResult] = []
    energy_spec_cache: Dict[str, EnergySourceSpec] = {}
    for index, csv_path in enumerate(csv_paths, start=1):
        print("-" * 88)
        print(f"[{index}/{len(csv_paths)}] Processing {csv_path.name}")
        try:
            result = process_single_csv_file(
                csv_path=csv_path,
                output_dir=output_dir,
                plots_output_dir=plots_output_dir,
                args=args,
                project_root=project_root,
                energy_spec_cache=energy_spec_cache,
            )
        except Exception as exc:
            source_slug = sanitize_stem(csv_path.stem)
            result = FileProcessingResult(
                source_csv=csv_path,
                accepted_segments=0,
                rejected_segments=0,
                removed_rows_by_y=0,
                cleaned_csv_paths=[],
                analysis_csv_path=(output_dir / f"{source_slug}_analysis_metrics.csv"),
                summary_plot_path=(plots_output_dir / f"{source_slug}_segments_summary.png"),
                segment_plot_paths=[],
                failed=True,
                failure_reason=str(exc),
            )
            print(f"Failed to process {csv_path}: {exc}")

        results.append(result)

    print_batch_summary(results)


def main() -> None:
    args = parse_args()

    if args.min_roll_deg <= 0.0:
        raise ValueError(f"--min-roll-deg must be positive. Got: {args.min_roll_deg}")
    if args.auto_relax_roll_threshold:
        # Validate sequence once at startup so bad values fail fast.
        parse_relax_roll_seq(args.relax_roll_seq)

    project_root = find_project_root(Path(__file__))
    output_dir = resolve_optional_dir(
        project_root=project_root,
        path_arg=args.output_dir,
        default_dir=default_processed_output_dir(project_root),
    )
    plots_output_dir = resolve_optional_dir(
        project_root=project_root,
        path_arg=args.plots_dir,
        default_dir=plots_dir(project_root),
    )

    if args.replot_cleaned_only:
        run_replot_cleaned_mode(
            project_root=project_root,
            plots_output_dir=plots_output_dir,
            args=args,
        )
        return

    if args.batch_all_raw:
        run_batch_mode(
            project_root=project_root,
            output_dir=output_dir,
            plots_output_dir=plots_output_dir,
            args=args,
        )
        return

    run_single_mode(
        project_root=project_root,
        output_dir=output_dir,
        plots_output_dir=plots_output_dir,
        args=args,
    )


if __name__ == "__main__":
    main()