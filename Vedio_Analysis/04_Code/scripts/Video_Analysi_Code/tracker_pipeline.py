from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from Video_Analysi_Code.tracker_detection import (
    adjust_center_to_black_long_edge_band,
    detect_black_line,
    detect_pink_object,
    preprocess_hsv,
    validate_black_detection,
)
from Video_Analysi_Code.tracker_geometry import (
    compute_oriented_box_and_center,
    constrain_center_step,
)
from Video_Analysi_Code.tracker_models import BlackDetection, TrackerParams, WINDOW_TRACK
from Video_Analysi_Code.tracker_overlay import draw_overlay, format_angle_text
from Video_Analysi_Code.tracker_postprocess import postprocess_angles
from Video_Analysi_Code.tracker_state import TrackingState, point_is_finite


def _resolve_frame_index(capture: cv2.VideoCapture, record_count: int) -> int:
    frame_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    if frame_index < 0:
        frame_index = record_count
    return frame_index


def _update_anomaly_state(
    state: TrackingState,
    black_status: str,
    frame_index: int,
    timestamp_s: float,
    low_conf_reason: str,
    pink_area: float,
    black: Optional[BlackDetection],
) -> None:
    if black_status in {"LOW_CONF", "LOST"}:
        if black_status == state.last_anomaly_status:
            state.anomaly_streak += 1
        else:
            state.anomaly_streak = 1
            state.last_anomaly_status = black_status

        if state.anomaly_streak == 1 or state.anomaly_streak % 15 == 0:
            black_area = black.area if black is not None else np.nan
            black_aspect = black.aspect_ratio if black is not None else np.nan
            black_long_edge = black.long_edge_px if black is not None else np.nan
            print(
                f"[{black_status}] frame={frame_index} t={timestamp_s:.4f}s "
                f"reason={low_conf_reason} "
                f"pink_area={format_angle_text(pink_area)} "
                f"black_area={format_angle_text(black_area)} "
                f"black_aspect={format_angle_text(black_aspect)} "
                f"black_long_edge={format_angle_text(black_long_edge)}"
            )
        return

    if state.last_anomaly_status in {"LOW_CONF", "LOST"}:
        print(f"[RECOVER] frame={frame_index} t={timestamp_s:.4f}s")
    state.last_anomaly_status = "NONE"
    state.anomaly_streak = 0


def _resolve_theta_display(state: TrackingState, theta_raw: float) -> float:
    if np.isnan(theta_raw):
        return state.last_theta_for_display

    state.last_theta_for_display = theta_raw
    return theta_raw


def _blend_centroid_with_black(
    centroid: Tuple[float, float],
    pink_contour: Optional[np.ndarray],
    pink_bbox: Optional[Tuple[int, int, int, int]],
    green_center: Tuple[float, float],
    theta_raw: float,
    black_box: Optional[np.ndarray],
    frame_index: int,
    timestamp_s: float,
    params: TrackerParams,
) -> Tuple[float, float]:
    if not np.isfinite(theta_raw) or black_box is None or pink_contour is None:
        return centroid

    _pink_box, pink_center = compute_oriented_box_and_center(pink_contour, theta_raw)
    if not point_is_finite(pink_center):
        return centroid

    adjusted_center = adjust_center_to_black_long_edge_band(
        pink_center,
        black_box=black_box,
        theta_deg=theta_raw,
    )

    if pink_bbox is None:
        return centroid

    gx, gy, gw, gh = pink_bbox
    green_diag = float(np.hypot(gw, gh))
    center_offset = float(
        np.hypot(
            adjusted_center[0] - green_center[0],
            adjusted_center[1] - green_center[1],
        )
    )

    if green_diag <= 1e-9 or center_offset > params.max_red_center_offset_ratio * green_diag:
        print(
            f"[CENTER_RECENTER] frame={frame_index} t={timestamp_s:.4f}s "
            f"offset={center_offset:.2f}px"
        )
        return green_center

    alpha = float(params.red_center_blend_ok)
    return (
        float((1.0 - alpha) * green_center[0] + alpha * adjusted_center[0]),
        float((1.0 - alpha) * green_center[1] + alpha * adjusted_center[1]),
    )


def _clip_centroid_to_bbox(
    centroid: Tuple[float, float],
    pink_bbox: Optional[Tuple[int, int, int, int]],
) -> Tuple[float, float]:
    if pink_bbox is None or not point_is_finite(centroid):
        return centroid

    gx, gy, gw, gh = pink_bbox
    return (
        float(np.clip(centroid[0], gx, gx + gw)),
        float(np.clip(centroid[1], gy, gy + gh)),
    )


def _apply_center_step_constraint(
    state: TrackingState,
    centroid: Tuple[float, float],
    black_status: str,
    frame_index: int,
    timestamp_s: float,
    params: TrackerParams,
) -> Tuple[float, float]:
    if point_is_finite(state.last_output_centroid) and point_is_finite(centroid):
        step = float(
            np.hypot(
                float(centroid[0]) - float(state.last_output_centroid[0]),
                float(centroid[1]) - float(state.last_output_centroid[1]),
            )
        )

        if black_status == "OK":
            centroid, clipped = constrain_center_step(
                centroid,
                state.last_output_centroid,
                params.max_center_step_px_ok,
            )
            if clipped:
                print(
                    f"[CENTER_CLIP_OK] frame={frame_index} t={timestamp_s:.4f}s "
                    f"step={step:.2f}px"
                )

    if point_is_finite(centroid):
        state.last_output_centroid = (float(centroid[0]), float(centroid[1]))

    return centroid


def _append_record(
    records: List[Dict[str, float]],
    frame_index: int,
    timestamp_s: float,
    centroid: Tuple[float, float],
    theta_raw: float,
    k_mm_per_pixel: float,
) -> None:
    x_val = float(centroid[0]) if not np.isnan(centroid[0]) else np.nan
    y_val = float(centroid[1]) if not np.isnan(centroid[1]) else np.nan

    x_mm = x_val * k_mm_per_pixel if not np.isnan(x_val) else np.nan
    y_mm = y_val * k_mm_per_pixel if not np.isnan(y_val) else np.nan

    records.append(
        {
            "frame_index": int(frame_index),
            "timestamp": float(timestamp_s),
            "x": x_val,
            "y": y_val,
            "theta": float(theta_raw),
            "x_mm": float(x_mm),
            "y_mm": float(y_mm),
        }
    )


def _handle_pause_if_requested(
    key: int,
    capture: cv2.VideoCapture,
    records: List[Dict[str, float]],
) -> Optional[pd.DataFrame]:
    if key != ord(" "):
        return None

    print("Paused. Press SPACE to resume or Q/ESC to quit.")
    while True:
        pause_key = cv2.waitKey(0) & 0xFF
        if pause_key == ord(" "):
            return None
        if pause_key in (ord("q"), 27):
            capture.release()
            cv2.destroyAllWindows()
            return postprocess_angles(pd.DataFrame(records))


def run_tracker(
    video_path: str,
    calibration: Dict[str, object],
    params: TrackerParams,
    enable_preview: bool = True,
) -> pd.DataFrame:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 1e-9:
        fps = 30.0

    k_mm_per_pixel = float(calibration["k_mm_per_pixel"])

    records: List[Dict[str, float]] = []
    state = TrackingState()

    if enable_preview:
        cv2.namedWindow(WINDOW_TRACK, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        frame_index = _resolve_frame_index(capture, len(records))
        timestamp_s = frame_index / fps

        hsv = preprocess_hsv(frame, params)
        pink = detect_pink_object(hsv, params)

        centroid = pink.centroid
        green_center = pink.centroid
        pink_bbox: Optional[Tuple[int, int, int, int]] = None
        theta_raw = np.nan
        black_box = None
        black_status = "NO_PINK"
        black: Optional[BlackDetection] = None
        low_conf_reason = "none"

        if pink.found and pink.contour is not None:
            pink_bbox = cv2.boundingRect(pink.contour)
            gx, gy, gw, gh = pink_bbox
            green_center = (float(gx + gw / 2.0), float(gy + gh / 2.0))
            centroid = green_center

        if pink.found:
            black = detect_black_line(hsv, pink.contour, params)
            black_ok, theta_candidate, black_reason = validate_black_detection(
                black=black,
                pink_area=pink.area,
                pink_bbox=pink_bbox,
                pink_center=green_center,
                last_valid_theta=state.last_valid_theta,
                params=params,
            )

            if black_ok:
                theta_raw = theta_candidate
                state.last_valid_theta = theta_raw
                black_box = black.box_points
                black_status = "OK"
            elif black.found:
                black_status = "LOW_CONF"
                low_conf_reason = black_reason
            else:
                black_status = "LOST"
                low_conf_reason = black_reason

        _update_anomaly_state(
            state=state,
            black_status=black_status,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            low_conf_reason=low_conf_reason,
            pink_area=pink.area,
            black=black,
        )

        theta_display = _resolve_theta_display(state, theta_raw)

        centroid = _blend_centroid_with_black(
            centroid=centroid,
            pink_contour=pink.contour,
            pink_bbox=pink_bbox,
            green_center=green_center,
            theta_raw=theta_raw,
            black_box=black_box,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            params=params,
        )

        centroid = _clip_centroid_to_bbox(centroid, pink_bbox)

        centroid = _apply_center_step_constraint(
            state=state,
            centroid=centroid,
            black_status=black_status,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            params=params,
        )

        _append_record(
            records=records,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            centroid=centroid,
            theta_raw=theta_raw,
            k_mm_per_pixel=k_mm_per_pixel,
        )

        if enable_preview:
            preview = draw_overlay(
                frame=frame,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                pink_contour=pink.contour,
                centroid=centroid,
                black_box=black_box,
                theta_raw=theta_raw,
                theta_display=theta_display,
                k_mm_per_pixel=k_mm_per_pixel,
            )

            cv2.imshow(WINDOW_TRACK, preview)
            key = cv2.waitKey(params.wait_key_ms) & 0xFF

            if key in (ord("q"), 27):
                break

            paused_result = _handle_pause_if_requested(key, capture, records)
            if paused_result is not None:
                return paused_result

    capture.release()
    if enable_preview:
        cv2.destroyAllWindows()

    df = pd.DataFrame(records)
    return postprocess_angles(df)
