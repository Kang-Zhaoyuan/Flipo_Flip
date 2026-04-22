from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize

if __package__ is None or __package__ == "":
    _scripts_root = Path(__file__).resolve().parent.parent
    if str(_scripts_root) not in sys.path:
        sys.path.insert(0, str(_scripts_root))

from Video_Analysi_Code.csv_paths import (
    default_plot_output_path,
    load_last_calibration_video_path as shared_load_last_calibration_video_path,
    resolve_csv_path as shared_resolve_csv_path,
    resolve_raw_csv_from_video as shared_resolve_raw_csv_from_video,
    resolve_video_path as shared_resolve_video_path,
)
from Video_Analysi_Code.path_registry import find_project_root

REQUIRED_COLUMNS = ["frame_index", "timestamp", "x_mm", "y_mm"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate raw-coordinate trajectory visualization directly from CSV "
            "without any coordinate-axis redefinition."
        )
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help=(
            "Source video path used to resolve <video_stem>_raw_data.csv. "
            "If omitted, fallback uses config.json:last_calibration.video_path."
        ),
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help=(
            "Optional explicit CSV path. If provided, this path is used directly "
            "and video-based CSV discovery is skipped."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flip_analysis_raw_coords.png",
        help="Output image path. Relative paths are resolved from project root.",
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


def load_and_validate_csv(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if len(df) < 2:
        raise ValueError("At least two rows are required for gradient-based speed.")

    if df["timestamp"].notna().sum() < 2:
        raise ValueError("Timestamp column has insufficient finite values.")

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def make_strictly_increasing(time_values: np.ndarray) -> np.ndarray:
    timestamps = pd.Series(time_values.astype(float)).interpolate(
        method="linear", limit_direction="both"
    ).to_numpy(dtype=float)

    if not np.isfinite(timestamps).all():
        raise ValueError("Timestamp interpolation failed; values are still non-finite.")

    for index in range(1, len(timestamps)):
        if timestamps[index] <= timestamps[index - 1]:
            timestamps[index] = np.nextafter(timestamps[index - 1], np.inf)

    return timestamps


def contiguous_slices(mask: np.ndarray) -> List[slice]:
    valid_indices = np.flatnonzero(mask)
    if valid_indices.size == 0:
        return []

    breaks = np.where(np.diff(valid_indices) > 1)[0]
    starts = np.r_[valid_indices[0], valid_indices[breaks + 1]]
    ends = np.r_[valid_indices[breaks], valid_indices[-1]]

    return [slice(int(start), int(end) + 1) for start, end in zip(starts, ends)]


def compute_instantaneous_speed(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    timestamps = make_strictly_increasing(df["timestamp"].to_numpy(dtype=float))

    x_raw = df["x_mm"].to_numpy(dtype=float)
    y_raw = df["y_mm"].to_numpy(dtype=float)

    x_filled = pd.Series(x_raw).interpolate(method="linear", limit_direction="both")
    y_filled = pd.Series(y_raw).interpolate(method="linear", limit_direction="both")

    x_calc = x_filled.to_numpy(dtype=float)
    y_calc = y_filled.to_numpy(dtype=float)

    if not np.isfinite(x_calc).all() or not np.isfinite(y_calc).all():
        raise ValueError("Position interpolation failed; speed cannot be computed reliably.")

    vx = np.gradient(x_calc, timestamps)
    vy = np.gradient(y_calc, timestamps)
    speed = np.hypot(vx, vy)

    return {
        "timestamp": timestamps,
        "x_raw": x_raw,
        "y_raw": y_raw,
        "speed": speed,
    }


def add_speed_colored_trajectory(
    ax: plt.Axes,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    speed: np.ndarray,
    cmap_name: str = "RdYlBu_r",
) -> ScalarMappable:
    valid_mask = np.isfinite(x_mm) & np.isfinite(y_mm) & np.isfinite(speed)
    slices = contiguous_slices(valid_mask)

    segment_data = []
    speed_values = []

    for section in slices:
        if section.stop - section.start < 2:
            continue

        points = np.column_stack((x_mm[section], y_mm[section]))
        segments = np.stack((points[:-1], points[1:]), axis=1)
        seg_speed = 0.5 * (speed[section][:-1] + speed[section][1:])

        finite_segment_mask = np.isfinite(seg_speed)
        if not np.any(finite_segment_mask):
            continue

        segment_data.append((segments[finite_segment_mask], seg_speed[finite_segment_mask]))
        speed_values.append(seg_speed[finite_segment_mask])

    if not speed_values:
        raise ValueError("No valid trajectory segments available for plotting.")

    all_segment_speeds = np.concatenate(speed_values)
    vmin = float(np.nanmin(all_segment_speeds))
    vmax = float(np.nanmax(all_segment_speeds))

    if np.isclose(vmin, vmax):
        margin = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        vmin -= margin
        vmax += margin

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    for segments, segment_speed in segment_data:
        line_collection = LineCollection(
            segments,
            cmap=cmap,
            norm=norm,
            linewidth=2.6,
            capstyle="round",
        )
        line_collection.set_array(segment_speed)
        ax.add_collection(line_collection)

    ax.autoscale()

    color_map_handle = ScalarMappable(norm=norm, cmap=cmap)
    color_map_handle.set_array(all_segment_speeds)
    return color_map_handle


def compute_basic_stats(df: pd.DataFrame, speed: np.ndarray) -> Dict[str, float]:
    total_frames = int(len(df))
    dropped_frames = int(df[["x_mm", "y_mm"]].isna().any(axis=1).sum())

    position_valid = np.isfinite(df["x_mm"].to_numpy(dtype=float)) & np.isfinite(
        df["y_mm"].to_numpy(dtype=float)
    )
    speed_valid = np.isfinite(speed) & position_valid

    average_speed = float(np.nan)
    if np.any(speed_valid):
        average_speed = float(np.mean(speed[speed_valid]))

    return {
        "total_frames": total_frames,
        "dropped_frames": dropped_frames,
        "average_speed": average_speed,
    }


def create_raw_trajectory_figure(speed_data: Dict[str, np.ndarray]) -> plt.Figure:
    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)

    color_mappable = add_speed_colored_trajectory(
        ax=ax,
        x_mm=speed_data["x_raw"],
        y_mm=speed_data["y_raw"],
        speed=speed_data["speed"],
        cmap_name="RdYlBu_r",
    )

    colorbar = fig.colorbar(color_mappable, ax=ax, pad=0.015)
    colorbar.set_label("Speed (mm/s)")

    ax.set_title("Spatial Trajectory Colored by Instantaneous Speed")
    ax.set_xlabel("x_mm from CSV")
    ax.set_ylabel("y_mm from CSV (raw, no transform)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    fig.suptitle("Flipo Flip Raw Coordinate Trajectory", fontsize=16)
    return fig


def print_stats(stats: Dict[str, float], speed_data: Dict[str, np.ndarray]) -> None:
    print(f"Total Frames: {stats['total_frames']}")
    print(f"Number of Dropped Frames (NaNs): {stats['dropped_frames']}")

    avg_speed = stats["average_speed"]
    if np.isfinite(avg_speed):
        print(f"Average Speed: {avg_speed:.3f} mm/s")
    else:
        print("Average Speed: NaN (insufficient valid data)")

    x_raw = speed_data["x_raw"]
    y_raw = speed_data["y_raw"]

    finite_x = x_raw[np.isfinite(x_raw)]
    finite_y = y_raw[np.isfinite(y_raw)]

    if finite_x.size > 0:
        print(f"Raw x_mm range from CSV: [{np.min(finite_x):.3f}, {np.max(finite_x):.3f}]")
    else:
        print("Raw x_mm range from CSV: NaN (no finite values)")

    if finite_y.size > 0:
        print(f"Raw y_mm range from CSV: [{np.min(finite_y):.3f}, {np.max(finite_y):.3f}]")
    else:
        print("Raw y_mm range from CSV: NaN (no finite values)")

    print("Coordinate transform: disabled (using raw CSV x_mm and y_mm directly).")


def main() -> None:
    args = parse_args()

    project_root = find_project_root(Path(__file__))

    output_arg = Path(args.output).expanduser()
    if output_arg == Path("flip_analysis_raw_coords.png"):
        output_path = default_plot_output_path(project_root, output_arg.name)
    elif not output_arg.is_absolute():
        output_path = (project_root / output_arg).resolve()
    else:
        output_path = output_arg

    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_input_path = args.csv_path
    if csv_input_path is None:
        csv_input_path = input(
            "Which .csv file do you want to plot? Enter CSV path (or press Enter to use auto-detection): "
        ).strip().strip('"')
        if not csv_input_path:
            csv_input_path = None

    try:
        csv_path, video_path, used_legacy_csv, used_explicit_csv = resolve_csv_path(
            project_root=project_root,
            explicit_csv_path=csv_input_path,
            explicit_video_path=args.video_path,
        )
        df = load_and_validate_csv(csv_path)
        speed_data = compute_instantaneous_speed(df)
        stats = compute_basic_stats(df, speed_data["speed"])

        figure = create_raw_trajectory_figure(speed_data)
        figure.savefig(output_path, dpi=320, bbox_inches="tight")
        plt.close(figure)

        print_stats(stats, speed_data)
        if video_path is not None:
            print(f"Using video: {video_path}")
        print(f"Using CSV: {csv_path}")
        if used_explicit_csv:
            print("CSV source mode: explicit --csv-path")
        elif used_legacy_csv:
            print(
                "CSV source mode: legacy project-root *_raw_data.csv fallback "
                "(01_Data/Raw preferred)"
            )
        print(f"Saved report: {output_path}")
    except Exception as exc:
        print(f"Failed to generate raw-coordinate visualization report: {exc}")


if __name__ == "__main__":
    main()