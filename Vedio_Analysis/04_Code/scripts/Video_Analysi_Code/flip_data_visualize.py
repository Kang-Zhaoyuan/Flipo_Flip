from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from scipy.signal import savgol_filter

if __package__ is None or __package__ == "":
    _scripts_root = Path(__file__).resolve().parent.parent
    if str(_scripts_root) not in sys.path:
        sys.path.insert(0, str(_scripts_root))

from Video_Analysi_Code.csv_paths import (
    default_plot_output_path,
    load_last_calibration_video_path as shared_load_last_calibration_video_path,
    resolve_raw_csv_from_video as shared_resolve_raw_csv_from_video,
    resolve_video_path as shared_resolve_video_path,
)
from Video_Analysi_Code.path_registry import find_project_root

REQUIRED_COLUMNS = [
    "frame_index",
    "timestamp",
    "x_mm",
    "y_mm",
    "theta_unwrapped",
    "omega_rad_s",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Flipo Flip analysis report with bottom-origin, up-positive "
            "display coordinates for trajectory Y."
        )
    )
    parser.add_argument(
        "--video-path",
        type=str,
        default=None,
        help=(
            "Source video path used to read frame height. "
            "If omitted, fallback uses config.json:last_calibration.video_path."
        ),
    )
    return parser.parse_args()


def load_last_calibration_video_path(config_path: Path) -> Optional[Path]:
    return shared_load_last_calibration_video_path(config_path)


def resolve_video_path(project_root: Path, explicit_video_path: Optional[str]) -> Path:
    return shared_resolve_video_path(project_root, explicit_video_path)


def resolve_csv_path(project_root: Path, video_path: Path) -> Tuple[Path, bool]:
    return shared_resolve_raw_csv_from_video(project_root, video_path)


def get_video_frame_height_px(video_path: Path) -> float:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video for frame-height lookup: {video_path}")

    try:
        frame_height_px = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_height_px > 1e-9:
            return frame_height_px

        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(
                f"Unable to read first frame for frame-height lookup: {video_path}"
            )

        return float(frame.shape[0])
    finally:
        capture.release()


def estimate_mm_per_pixel(df: pd.DataFrame) -> float:
    ratio_samples: List[np.ndarray] = []

    if {"y", "y_mm"}.issubset(df.columns):
        y_px = pd.to_numeric(df["y"], errors="coerce").to_numpy(dtype=float)
        y_mm = df["y_mm"].to_numpy(dtype=float)
        y_mask = np.isfinite(y_px) & np.isfinite(y_mm) & (np.abs(y_px) > 1e-9)
        if np.any(y_mask):
            ratio_samples.append(y_mm[y_mask] / y_px[y_mask])

    if {"x", "x_mm"}.issubset(df.columns):
        x_px = pd.to_numeric(df["x"], errors="coerce").to_numpy(dtype=float)
        x_mm = df["x_mm"].to_numpy(dtype=float)
        x_mask = np.isfinite(x_px) & np.isfinite(x_mm) & (np.abs(x_px) > 1e-9)
        if np.any(x_mask):
            ratio_samples.append(x_mm[x_mask] / x_px[x_mask])

    if not ratio_samples:
        raise ValueError(
            "Cannot estimate mm/pixel from CSV. "
            "Expected finite x/x_mm or y/y_mm pairs."
        )

    ratios = np.concatenate(ratio_samples)
    ratios = ratios[np.isfinite(ratios)]

    if ratios.size == 0:
        raise ValueError("mm/pixel estimation failed: no finite ratio samples.")

    k_mm_per_pixel = float(np.median(ratios))
    if not np.isfinite(k_mm_per_pixel) or k_mm_per_pixel <= 0.0:
        raise ValueError(f"mm/pixel estimation failed: invalid scale {k_mm_per_pixel}.")

    return k_mm_per_pixel


def build_bottom_origin_y_mm(
    df: pd.DataFrame,
    y_mm: np.ndarray,
    frame_height_px: float,
) -> Tuple[np.ndarray, float, float]:
    k_mm_per_pixel = estimate_mm_per_pixel(df)
    y_bottom_mm = frame_height_px * k_mm_per_pixel
    y_display_mm = y_bottom_mm - y_mm
    return y_display_mm, k_mm_per_pixel, y_bottom_mm


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


def smooth_signal_with_gaps(
    timestamps: np.ndarray,
    signal: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
) -> np.ndarray:
    smoothed = np.full(signal.shape, np.nan, dtype=float)
    valid_mask = np.isfinite(timestamps) & np.isfinite(signal)

    min_window = polyorder + 2
    if min_window % 2 == 0:
        min_window += 1

    for section in contiguous_slices(valid_mask):
        segment = signal[section]
        segment_len = segment.size

        if segment_len < 3:
            smoothed[section] = segment
            continue

        adaptive_window = min(window_length, segment_len)
        if adaptive_window % 2 == 0:
            adaptive_window -= 1

        if adaptive_window < min_window:
            smoothed[section] = segment
            continue

        adaptive_polyorder = min(polyorder, adaptive_window - 1)
        smoothed[section] = savgol_filter(
            segment,
            window_length=adaptive_window,
            polyorder=adaptive_polyorder,
            mode="interp",
        )

    return smoothed


def compute_basic_stats(df: pd.DataFrame, speed: np.ndarray) -> Dict[str, float]:
    total_frames = int(len(df))

    dropped_frames = int(
        df[["x_mm", "y_mm", "theta_unwrapped"]].isna().any(axis=1).sum()
    )

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


def create_report_figure(
    df: pd.DataFrame,
    speed_data: Dict[str, np.ndarray],
    trajectory_y_mm: np.ndarray,
) -> plt.Figure:
    plt.style.use("dark_background")

    fig = plt.figure(figsize=(13, 10), constrained_layout=True)
    grid = fig.add_gridspec(3, 1, height_ratios=[2.4, 1.0, 1.0])

    ax_traj = fig.add_subplot(grid[0, 0])
    ax_theta = fig.add_subplot(grid[1, 0])
    ax_omega = fig.add_subplot(grid[2, 0], sharex=ax_theta)

    color_mappable = add_speed_colored_trajectory(
        ax=ax_traj,
        x_mm=speed_data["x_raw"],
        y_mm=trajectory_y_mm,
        speed=speed_data["speed"],
        cmap_name="RdYlBu_r",
    )

    colorbar = fig.colorbar(color_mappable, ax=ax_traj, pad=0.015)
    colorbar.set_label("Speed (mm/s)")

    ax_traj.set_title("Spatial Trajectory Colored by Instantaneous Speed")
    ax_traj.set_xlabel(r"$x$ (mm)")
    ax_traj.set_ylabel("y (mm, bottom-origin, up-positive)")
    ax_traj.set_aspect("equal", adjustable="box")
    ax_traj.grid(alpha=0.25, linestyle="--", linewidth=0.6)

    timestamps = speed_data["timestamp"]
    theta = df["theta_unwrapped"].to_numpy(dtype=float)
    omega = df["omega_rad_s"].to_numpy(dtype=float)

    ax_theta.plot(
        timestamps,
        theta,
        color="#3FB7FF",
        linewidth=1.4,
        label="Raw theta_unwrapped",
    )
    ax_theta.set_ylabel(r"$\theta_{unwrapped}$ (deg)")
    ax_theta.grid(alpha=0.28, linestyle="--", linewidth=0.6)
    ax_theta.legend(loc="upper left", framealpha=0.35)

    omega_smooth = smooth_signal_with_gaps(
        timestamps=timestamps,
        signal=omega,
        window_length=11,
        polyorder=3,
    )

    ax_omega.plot(
        timestamps,
        omega,
        color="#BFD7EA",
        linewidth=1.1,
        alpha=0.35,
        label="Raw omega",
    )
    ax_omega.plot(
        timestamps,
        omega_smooth,
        color="#FF4D4D",
        linewidth=2.0,
        label="Savitzky-Golay smooth",
    )
    ax_omega.set_ylabel(r"$\omega$ (rad/s)")
    ax_omega.set_xlabel("Timestamp (s)")
    ax_omega.grid(alpha=0.28, linestyle="--", linewidth=0.6)
    ax_omega.legend(loc="upper left", framealpha=0.35)

    fig.suptitle("Flipo Flip Kinematic Analysis", fontsize=16)
    return fig


def print_stats(stats: Dict[str, float]) -> None:
    print(f"Total Frames: {stats['total_frames']}")
    print(f"Number of Dropped Frames (NaNs): {stats['dropped_frames']}")

    avg_speed = stats["average_speed"]
    if np.isfinite(avg_speed):
        print(f"Average Speed: {avg_speed:.3f} mm/s")
    else:
        print("Average Speed: NaN (insufficient valid data)")


def main() -> None:
    args = parse_args()

    project_root = find_project_root(Path(__file__))
    output_path = default_plot_output_path(project_root, "flip_analysis_report.png")

    try:
        video_path = resolve_video_path(project_root, args.video_path)
        csv_path, used_legacy_csv = resolve_csv_path(project_root, video_path)
        df = load_and_validate_csv(csv_path)
        speed_data = compute_instantaneous_speed(df)
        frame_height_px = get_video_frame_height_px(video_path)
        trajectory_y_mm, k_mm_per_pixel, y_bottom_mm = build_bottom_origin_y_mm(
            df=df,
            y_mm=speed_data["y_raw"],
            frame_height_px=frame_height_px,
        )
        stats = compute_basic_stats(df, speed_data["speed"])

        figure = create_report_figure(
            df=df,
            speed_data=speed_data,
            trajectory_y_mm=trajectory_y_mm,
        )
        figure.savefig(output_path, dpi=320, bbox_inches="tight")
        plt.close(figure)

        print_stats(stats)
        print(
            "Y display transform: "
            f"y_plot_mm = ({frame_height_px:.1f} px * {k_mm_per_pixel:.6f} mm/px) - y_mm"
        )
        print(f"Bottom-origin reference y=0 at frame bottom; y_top={y_bottom_mm:.3f} mm")
        print(f"Using video: {video_path}")
        print(f"Using CSV: {csv_path}")
        if used_legacy_csv:
            print(
                "Preferred CSV in 01_Data/Raw not found; "
                "fell back to legacy project-root *_raw_data.csv"
            )
        print(f"Saved report: {output_path}")
    except Exception as exc:
        print(f"Failed to generate visualization report: {exc}")


if __name__ == "__main__":
    main()
