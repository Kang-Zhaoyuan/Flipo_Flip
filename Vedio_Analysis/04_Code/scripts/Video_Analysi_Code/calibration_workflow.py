from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from Video_Analysi_Code.calibration_io import save_config
from Video_Analysi_Code.calibration_ui import (
    prompt_float,
    prompt_reuse_or_calibrate,
    select_calibration_frame,
    select_two_points,
)


def build_calibration_record(
    points: List[Tuple[int, int]],
    distance_mm: float,
    video_path: str,
    calibration_frame_index: int,
) -> Dict[str, object]:
    p1 = np.array(points[0], dtype=np.float32)
    p2 = np.array(points[1], dtype=np.float32)
    pixel_distance = float(np.linalg.norm(p1 - p2))

    if pixel_distance <= 1e-9:
        raise RuntimeError("Selected points are identical. Calibration failed.")

    k_mm_per_pixel = distance_mm / pixel_distance

    return {
        "video_path": video_path,
        "calibration_frame_index": int(calibration_frame_index),
        "point1": [int(points[0][0]), int(points[0][1])],
        "point2": [int(points[1][0]), int(points[1][1])],
        "distance_mm": float(distance_mm),
        "pixel_distance": float(pixel_distance),
        "k_mm_per_pixel": float(k_mm_per_pixel),
    }


def run_calibration_interactive(video_path: str) -> Dict[str, object]:
    frame_selection = select_calibration_frame(video_path)
    if frame_selection is None:
        raise RuntimeError("Calibration canceled by user.")

    calibration_frame, calibration_frame_index = frame_selection

    selected = select_two_points(calibration_frame)
    if selected is None:
        raise RuntimeError("Calibration canceled by user.")

    distance_mm = prompt_float("Enter known distance between points (mm): ", minimum=0.0)
    calibration = build_calibration_record(
        selected,
        distance_mm,
        video_path,
        calibration_frame_index,
    )

    print(
        "Calibration complete. "
        f"K = {calibration['k_mm_per_pixel']:.6f} mm/pixel"
    )
    return calibration


def choose_calibration(
    video_path: str,
    config_path: Path,
    config_data: Dict[str, object],
) -> Dict[str, object]:
    videos = config_data.setdefault("videos", {})
    if not isinstance(videos, dict):
        videos = {}
        config_data["videos"] = videos

    existing = videos.get(video_path)
    last_calibration = config_data.get("last_calibration")

    calibration: Dict[str, object]

    if isinstance(existing, dict):
        choice = prompt_reuse_or_calibrate(
            "Calibration found for this video. Reuse or recalibrate?"
        )
        if choice == "reuse":
            calibration = dict(existing)
        else:
            calibration = run_calibration_interactive(video_path)
    else:
        if isinstance(last_calibration, dict):
            choice = prompt_reuse_or_calibrate(
                "No calibration for this video. Reuse last calibration or calibrate now?"
            )
            if choice == "reuse":
                calibration = dict(last_calibration)
                calibration["video_path"] = video_path
            else:
                calibration = run_calibration_interactive(video_path)
        else:
            print("No previous calibration found. Starting calibration.")
            calibration = run_calibration_interactive(video_path)

    videos[video_path] = calibration
    config_data["last_calibration"] = calibration
    save_config(config_path, config_data)
    return calibration
