from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

CLEANED_SEGMENT_PATTERN = re.compile(
    r"^([A-Za-z]+)_D(\d+)_T([\d.]+)_L([\d.]+)_W([\d.]+)_raw_data_cleaned_segment_(\d+)\.csv$",
    re.IGNORECASE,
)

LEGACY_PATTERN = re.compile(
    r"^([A-Za-z]+)_D(\d+)_T([\d.]+)_L([\d.]+)_W([\d.]+)\.csv$",
    re.IGNORECASE,
)

REQUIRED_COLUMNS = ["x_mm", "y_mm", "theta_unwrapped"]
OPTIONAL_TIME_COLUMNS = ["timestamp", "frame_index"]

SIDE_LENGTH_MM = 50.0
GRAVITY_M_S2 = 9.81
PREFERRED_SMOOTH_WINDOW = 16
SMOOTH_POLY = 3


@dataclass
class TrialParams:
    color: str
    infill: int
    thickness_mm: float
    edge_width_mm: float
    mass_g: float
    segment_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch process CSV files in a directory and generate energy dissipation plots."
        )
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory containing CSV files. Defaults to current directory.",
    )
    return parser.parse_args()


def parse_trial_params(file_name: str) -> TrialParams | None:
    cleaned_match = CLEANED_SEGMENT_PATTERN.match(file_name)
    if cleaned_match:
        color, infill, thickness, edge_width, weight, segment_index = cleaned_match.groups()
        return TrialParams(
            color=color,
            infill=int(infill),
            thickness_mm=float(thickness),
            edge_width_mm=float(edge_width),
            mass_g=float(weight),
            segment_index=int(segment_index),
        )

    legacy_match = LEGACY_PATTERN.match(file_name)
    if not legacy_match:
        return None

    color, infill, thickness, edge_width, weight = legacy_match.groups()
    return TrialParams(
        color=color,
        infill=int(infill),
        thickness_mm=float(thickness),
        edge_width_mm=float(edge_width),
        mass_g=float(weight),
    )


def interpolate_finite(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values.astype(float)).interpolate(
        method="linear",
        limit_direction="both",
    )
    series = series.ffill().bfill()
    output = series.to_numpy(dtype=float)
    if not np.isfinite(output).all():
        raise ValueError("Interpolation failed: series still has non-finite values.")
    return output


def make_strictly_increasing(time_values: np.ndarray) -> np.ndarray:
    times = time_values.copy().astype(float)
    for idx in range(1, len(times)):
        if times[idx] <= times[idx - 1]:
            times[idx] = np.nextafter(times[idx - 1], np.inf)
    return times


def resolve_time_axis_seconds(df: pd.DataFrame) -> Tuple[np.ndarray, str]:
    if "timestamp" in df.columns:
        timestamps = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(timestamps).sum() >= 2:
            ts = interpolate_finite(timestamps)
            if np.ptp(ts) > 0.0:
                return make_strictly_increasing(ts), "timestamp"

    if "frame_index" in df.columns:
        frame_index = pd.to_numeric(df["frame_index"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(frame_index).sum() >= 2:
            fi = interpolate_finite(frame_index)
            if np.ptp(fi) > 0.0:
                return make_strictly_increasing(fi), "frame_index"

    raise ValueError("Cannot build a valid time axis from timestamp or frame_index.")


def validate_and_prepare(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if not any(col in df.columns for col in OPTIONAL_TIME_COLUMNS):
        raise ValueError("Missing both timestamp and frame_index columns.")

    numeric_cols = REQUIRED_COLUMNS + [col for col in OPTIONAL_TIME_COLUMNS if col in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with enough signal to support smoothing and derivatives.
    usable_mask = (
        np.isfinite(df["x_mm"].to_numpy(dtype=float))
        | np.isfinite(df["y_mm"].to_numpy(dtype=float))
        | np.isfinite(df["theta_unwrapped"].to_numpy(dtype=float))
    )
    df = df.loc[usable_mask].copy().reset_index(drop=True)

    min_window = SMOOTH_POLY + 2
    if min_window % 2 == 0:
        min_window += 1

    if len(df) < min_window:
        raise ValueError(
            f"Need at least {min_window} rows for Savitzky-Golay smoothing; got {len(df)}."
        )

    return df


def choose_smoothing_window(sample_count: int) -> int:
    if sample_count <= 0:
        raise ValueError("Smoothing window selection failed: empty series.")

    window = min(PREFERRED_SMOOTH_WINDOW, sample_count)
    if window % 2 == 0:
        window -= 1

    min_window = SMOOTH_POLY + 2
    if min_window % 2 == 0:
        min_window += 1

    if window < min_window:
        raise ValueError(
            f"Need at least {min_window} samples for smoothing with poly={SMOOTH_POLY}; got {sample_count}."
        )

    return window


def smooth_series(values: np.ndarray) -> np.ndarray:
    filled = interpolate_finite(values)
    smooth_window = choose_smoothing_window(len(filled))
    return savgol_filter(
        filled,
        window_length=smooth_window,
        polyorder=SMOOTH_POLY,
        mode="interp",
    )


def map_y_to_physical_height_mm(y_smoothed_mm: np.ndarray, y_raw_mm: np.ndarray, thickness_mm: float) -> np.ndarray:
    finite_raw = y_raw_mm[np.isfinite(y_raw_mm)]
    if finite_raw.size < 2:
        raise ValueError("Not enough finite y_mm samples to build physical mapping.")

    raw_min = float(np.nanmin(finite_raw))
    raw_max = float(np.nanmax(finite_raw))
    if np.isclose(raw_min, raw_max):
        raise ValueError("Cannot map y_mm to physical height: raw y_mm has zero range.")

    y_top_mm = SIDE_LENGTH_MM / 2.0
    y_bottom_mm = thickness_mm / 2.0

    y_phys_mm = np.interp(
        y_smoothed_mm,
        [raw_min, raw_max],
        [y_top_mm, y_bottom_mm],
    )

    lower = min(y_top_mm, y_bottom_mm)
    upper = max(y_top_mm, y_bottom_mm)
    return np.clip(y_phys_mm, lower, upper)


def compute_energy_terms(df: pd.DataFrame, params: TrialParams) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    time_s, time_source = resolve_time_axis_seconds(df)

    x_raw_mm = df["x_mm"].to_numpy(dtype=float)
    y_raw_mm = df["y_mm"].to_numpy(dtype=float)
    theta_raw_deg = df["theta_unwrapped"].to_numpy(dtype=float)

    x_smooth_mm = smooth_series(x_raw_mm)
    y_smooth_mm = smooth_series(y_raw_mm)
    theta_smooth_deg = smooth_series(theta_raw_deg)

    y_phys_mm = map_y_to_physical_height_mm(
        y_smoothed_mm=y_smooth_mm,
        y_raw_mm=y_raw_mm,
        thickness_mm=params.thickness_mm,
    )

    x_m = x_smooth_mm / 1000.0
    y_m = y_phys_mm / 1000.0
    theta_rad = np.deg2rad(theta_smooth_deg)

    vx = np.gradient(x_m, time_s)
    vy = np.gradient(y_m, time_s)
    omega = np.gradient(theta_rad, time_s)

    mass_kg = params.mass_g / 1000.0
    side_length_m = SIDE_LENGTH_MM / 1000.0
    thickness_m = params.thickness_mm / 1000.0

    inertia_kg_m2 = (1.0 / 12.0) * mass_kg * (side_length_m**2 + thickness_m**2)

    e_trans = 0.5 * mass_kg * (vx**2 + vy**2)
    e_rot = 0.5 * inertia_kg_m2 * (omega**2)
    e_pot = mass_kg * GRAVITY_M_S2 * y_m
    e_total = e_trans + e_rot + e_pot

    return time_s, e_trans, e_rot, e_pot, e_total, time_source


def create_plot(
    csv_path: Path,
    params: TrialParams,
    time_s: np.ndarray,
    e_trans: np.ndarray,
    e_rot: np.ndarray,
    e_pot: np.ndarray,
    e_total: np.ndarray,
    time_source: str,
) -> Path:
    time_rel_s = time_s - time_s[0]
    e_trans_mj = e_trans * 1000.0
    e_rot_mj = e_rot * 1000.0
    e_pot_mj = e_pot * 1000.0
    e_total_mj = e_total * 1000.0

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_rel_s, e_total_mj, color="black", linestyle="-", linewidth=2.0, label="Total Energy")
    ax.plot(
        time_rel_s,
        e_trans_mj,
        color="blue",
        linestyle="--",
        alpha=0.5,
        label="Translational KE",
    )
    ax.plot(
        time_rel_s,
        e_rot_mj,
        color="orange",
        linestyle="--",
        alpha=0.5,
        label="Rotational KE",
    )
    ax.plot(
        time_rel_s,
        e_pot_mj,
        color="green",
        linestyle=":",
        alpha=0.7,
        label="Potential Energy",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Energy (mJ)")
    ax.set_title(f"Energy Dissipation: {csv_path.stem}")
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="best")

    info_text = f"M: {params.mass_g:g}g, T: {params.thickness_mm:g}mm, L: {params.edge_width_mm:g}mm"
    ax.text(
        0.02,
        0.98,
        info_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.85, "boxstyle": "round,pad=0.3"},
    )

    ax.text(
        0.98,
        0.02,
        f"time source: {time_source}",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        color="dimgray",
        fontsize=9,
    )

    output_path = csv_path.with_suffix(".jpg")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, format="jpg")
    plt.close(fig)
    return output_path


def process_csv(csv_path: Path) -> Tuple[bool, str]:
    params = parse_trial_params(csv_path.name)
    if params is None:
        return False, "filename does not match supported naming spec"

    df = pd.read_csv(csv_path)
    df = validate_and_prepare(df)

    time_s, e_trans, e_rot, e_pot, e_total, time_source = compute_energy_terms(df, params)
    output_path = create_plot(
        csv_path=csv_path,
        params=params,
        time_s=time_s,
        e_trans=e_trans,
        e_rot=e_rot,
        e_pot=e_pot,
        e_total=e_total,
        time_source=time_source,
    )
    return True, f"saved {output_path.name}"


def main() -> None:
    args = parse_args()
    target_dir = Path(args.dir).expanduser().resolve()

    if not target_dir.exists() or not target_dir.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {target_dir}")

    csv_files = sorted(path for path in target_dir.glob("*.csv") if path.is_file())
    print(f"Scanning directory: {target_dir}")
    print(f"Found {len(csv_files)} CSV file(s).")

    if not csv_files:
        return

    processed = 0
    skipped = 0
    failed = 0

    for idx, csv_path in enumerate(csv_files, start=1):
        print(f"[{idx}/{len(csv_files)}] Processing {csv_path.name}")
        try:
            ok, message = process_csv(csv_path)
            if ok:
                processed += 1
                print(f"  OK: {message}")
            else:
                skipped += 1
                print(f"  SKIP: {message}")
        except Exception as exc:
            failed += 1
            print(f"  FAIL: {exc}")

    print("\nBatch summary")
    print(f"  processed: {processed}")
    print(f"  skipped:   {skipped}")
    print(f"  failed:    {failed}")


if __name__ == "__main__":
    main()