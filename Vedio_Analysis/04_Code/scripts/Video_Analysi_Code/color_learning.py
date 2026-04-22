from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from Video_Analysi_Code.calibration_ui import select_calibration_frame
from Video_Analysi_Code.color_profiles import (
    serialize_hsv_range,
    serialize_hsv_ranges,
)
from Video_Analysi_Code.tracker_detection import detect_black_line, detect_pink_object, preprocess_hsv
from Video_Analysi_Code.tracker_models import TrackerParams, WINDOW_CALIB

HSVTriplet = Tuple[int, int, int]
HSVRange = Tuple[HSVTriplet, HSVTriplet]
ROI = Tuple[int, int, int, int]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _roi_is_valid(roi_xywh: ROI) -> bool:
    return roi_xywh[2] > 0 and roi_xywh[3] > 0


def _clip_roi_to_frame(
    roi_xywh: ROI,
    frame_shape: Tuple[int, int],
) -> Optional[ROI]:
    frame_h, frame_w = frame_shape
    x, y, w, h = roi_xywh
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame_w, int(x + w))
    y1 = min(frame_h, int(y + h))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _extract_hsv_pixels(hsv: np.ndarray, roi_xywh: ROI) -> np.ndarray:
    clipped = _clip_roi_to_frame(roi_xywh, hsv.shape[:2])
    if clipped is None:
        return np.empty((0, 3), dtype=np.int32)

    x, y, w, h = clipped
    roi_pixels = hsv[y : y + h, x : x + w]
    if roi_pixels.size == 0:
        return np.empty((0, 3), dtype=np.int32)
    return roi_pixels.reshape(-1, 3).astype(np.int32)


def _percentile_bounds(
    values: np.ndarray,
    low_q: float,
    high_q: float,
    pad: int,
    clip_low: int,
    clip_high: int,
) -> Tuple[int, int]:
    if values.size == 0:
        return clip_low, clip_high

    low = int(np.percentile(values, low_q)) - pad
    high = int(np.percentile(values, high_q)) + pad
    low = max(clip_low, low)
    high = min(clip_high, high)
    if low > high:
        return clip_low, clip_high
    return low, high


def _remove_distractor_like_pixels(
    object_pixels: np.ndarray,
    distractor_pixels: np.ndarray,
) -> np.ndarray:
    if object_pixels.shape[0] == 0 or distractor_pixels.shape[0] == 0:
        return object_pixels

    object_h = object_pixels[:, 0]
    distractor_h = distractor_pixels[:, 0]

    object_hist = np.bincount(object_h, minlength=181).astype(np.float64)
    distractor_hist = np.bincount(distractor_h, minlength=181).astype(np.float64)

    # Keep hue bins where object support is stronger than distractor support.
    allowed_hue = object_hist > (distractor_hist * 1.15)
    kept = object_pixels[allowed_hue[object_h]]

    minimum_kept = max(200, int(object_pixels.shape[0] * 0.15))
    if kept.shape[0] < minimum_kept:
        return object_pixels
    return kept


def _estimate_object_hsv_ranges_from_roi(
    frame: np.ndarray,
    object_roi: ROI,
    color_name: str,
    distractor_roi: Optional[ROI] = None,
) -> Tuple[HSVRange, ...]:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    object_pixels = _extract_hsv_pixels(hsv, object_roi)
    if object_pixels.shape[0] == 0:
        raise RuntimeError("Object ROI is empty. Cannot estimate HSV ranges.")

    distractor_pixels = (
        _extract_hsv_pixels(hsv, distractor_roi)
        if distractor_roi is not None
        else np.empty((0, 3), dtype=np.int32)
    )
    object_pixels = _remove_distractor_like_pixels(object_pixels, distractor_pixels)

    s_all = object_pixels[:, 1]
    v_all = object_pixels[:, 2]

    if color_name == "white":
        s_low, s_high = _percentile_bounds(s_all, 2.0, 90.0, 5, 0, 255)
        v_low, v_high = _percentile_bounds(v_all, 10.0, 98.0, 8, 0, 255)
        s_high = min(120, s_high)
        v_low = max(80, v_low)
        return (((0, s_low, v_low), (180, s_high, v_high)),)

    s_low, s_high = _percentile_bounds(s_all, 5.0, 95.0, 10, 0, 255)
    v_low, v_high = _percentile_bounds(v_all, 5.0, 95.0, 10, 0, 255)
    s_low = max(35, s_low)
    v_low = max(20, v_low)

    sv_focus = object_pixels[(object_pixels[:, 1] >= s_low) & (object_pixels[:, 2] >= v_low)]
    focus = sv_focus if sv_focus.shape[0] >= 120 else object_pixels
    h_focus = focus[:, 0]

    if color_name in {"dark_red", "pink"}:
        low_group = h_focus[h_focus <= 25]
        high_group = h_focus[h_focus >= 155]
        min_group_size = max(80, int(h_focus.shape[0] * 0.05))

        ranges: List[HSVRange] = []
        if low_group.size >= min_group_size:
            h_low, h_high = _percentile_bounds(low_group, 5.0, 95.0, 3, 0, 30)
            ranges.append(((h_low, s_low, v_low), (h_high, s_high, v_high)))

        if high_group.size >= min_group_size:
            h_low, h_high = _percentile_bounds(high_group, 5.0, 95.0, 3, 150, 180)
            ranges.append(((h_low, s_low, v_low), (h_high, s_high, v_high)))

        if ranges:
            return tuple(ranges)

    h_low, h_high = _percentile_bounds(h_focus, 5.0, 95.0, 4, 0, 180)
    return (((h_low, s_low, v_low), (h_high, s_high, v_high)),)


def select_learning_rois(
    frame: np.ndarray,
) -> Optional[Tuple[ROI, ROI, Optional[ROI]]]:
    print("Draw ROI #1 for Flipo Flip body, then press ENTER or SPACE.")
    cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)
    object_roi_raw = cv2.selectROI(WINDOW_CALIB, frame, fromCenter=False, showCrosshair=True)
    object_roi = (
        int(object_roi_raw[0]),
        int(object_roi_raw[1]),
        int(object_roi_raw[2]),
        int(object_roi_raw[3]),
    )
    if not _roi_is_valid(object_roi):
        cv2.destroyWindow(WINDOW_CALIB)
        return None

    print("Draw ROI #2 for the black short line, then press ENTER or SPACE.")
    black_roi_raw = cv2.selectROI(WINDOW_CALIB, frame, fromCenter=False, showCrosshair=True)
    black_roi = (
        int(black_roi_raw[0]),
        int(black_roi_raw[1]),
        int(black_roi_raw[2]),
        int(black_roi_raw[3]),
    )

    if not _roi_is_valid(black_roi):
        cv2.destroyWindow(WINDOW_CALIB)
        return None

    print(
        "Draw ROI #3 for distractor region (for example arm/background) to exclude it. "
        "Press 'c' to skip this optional step."
    )
    distractor_raw = cv2.selectROI(WINDOW_CALIB, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(WINDOW_CALIB)

    distractor_roi: Optional[ROI] = None
    distractor_candidate = (
        int(distractor_raw[0]),
        int(distractor_raw[1]),
        int(distractor_raw[2]),
        int(distractor_raw[3]),
    )
    if _roi_is_valid(distractor_candidate):
        distractor_roi = distractor_candidate

    return object_roi, black_roi, distractor_roi


def _point_in_roi(point_xy: Tuple[float, float], roi_xywh: ROI) -> bool:
    x, y, w, h = roi_xywh
    px, py = point_xy
    return bool(x <= px <= x + w and y <= py <= y + h)


def _draw_learning_overlay(
    frame: np.ndarray,
    object_roi: ROI,
    black_roi: ROI,
    distractor_roi: Optional[ROI],
    object_contour: Optional[np.ndarray],
    black_box: Optional[np.ndarray],
    object_ranges: Tuple[HSVRange, ...],
    object_found: bool,
    prediction_in_object_roi: bool,
    black_found: bool,
    black_in_hint: bool,
    color_name: str,
) -> np.ndarray:
    overlay = frame.copy()

    ox, oy, ow, oh = object_roi
    bx, by, bw, bh = black_roi

    cv2.rectangle(overlay, (ox, oy), (ox + ow, oy + oh), (60, 220, 80), 2, cv2.LINE_AA)
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (160, 220, 255), 1, cv2.LINE_AA)
    if distractor_roi is not None:
        dx, dy, dw, dh = distractor_roi
        cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), (20, 80, 255), 2, cv2.LINE_AA)

    if object_contour is not None:
        cv2.drawContours(overlay, [object_contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(object_contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)

    if black_box is not None:
        x, y, w, h = cv2.boundingRect(black_box.astype(np.int32))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)

    range_descriptions = [
        (
            f"R{idx + 1}: "
            f"H[{lower[0]},{upper[0]}] "
            f"S[{lower[1]},{upper[1]}] "
            f"V[{lower[2]},{upper[2]}]"
        )
        for idx, (lower, upper) in enumerate(object_ranges)
    ]

    lines = [
        f"Interactive learning: {color_name}",
        (
            "Model judgment (targets ROI #1): "
            f"{'yes' if prediction_in_object_roi else 'no'}"
        ),
        f"Object found: {'yes' if object_found else 'no'}",
        f"Black-line found: {'yes' if black_found else 'no'}",
        f"Black center in ROI #2: {'yes' if black_in_hint else 'no'}",
        "ENTER: correct and accept   R: wrong and relearn   ESC/Q: cancel",
    ] + range_descriptions

    y0 = 26
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        y0 += 24

    return overlay


def _confirm_overlay(overlay: np.ndarray) -> str:
    cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)

    while True:
        cv2.imshow(WINDOW_CALIB, overlay)
        key = cv2.waitKey(20) & 0xFF

        if key in (13, 10):
            cv2.destroyWindow(WINDOW_CALIB)
            return "accept"
        if key in (ord("r"),):
            cv2.destroyWindow(WINDOW_CALIB)
            return "redo"
        if key in (27, ord("q")):
            cv2.destroyWindow(WINDOW_CALIB)
            return "cancel"


def run_first_time_color_learning(
    video_path: str,
    color_name: str,
    base_params: TrackerParams,
    round_count: int = 1,
) -> Dict[str, object]:
    _ = round_count  # Keep signature stable for callers while using feedback loop.
    first_seen_at = _now_iso()

    print(
        f"First time detecting color '{color_name}'. "
        "Starting interactive feedback learning (no manual HSV input)."
    )
    print(
        "Workflow: draw Flipo ROI + optional distractor ROI, then confirm "
        "whether model judgment is correct."
    )

    while True:
        frame_selection = select_calibration_frame(video_path)
        if frame_selection is None:
            raise RuntimeError("Color learning canceled by user during frame selection.")

        frame, frame_index = frame_selection
        roi_selection = select_learning_rois(frame)
        if roi_selection is None:
            raise RuntimeError("Color learning canceled by user during ROI selection.")

        object_roi, black_roi, distractor_roi = roi_selection
        object_ranges = _estimate_object_hsv_ranges_from_roi(
            frame=frame,
            object_roi=object_roi,
            color_name=color_name,
            distractor_roi=distractor_roi,
        )
        black_range = base_params.black_range

        print("Auto-estimated object HSV ranges:")
        for idx, (lower, upper) in enumerate(object_ranges, start=1):
            print(f"  range {idx}: LOWER={lower} UPPER={upper}")

        learning_params = replace(
            base_params,
            color_name=color_name,
            object_ranges=object_ranges,
            black_range=black_range,
        )

        hsv = preprocess_hsv(frame, learning_params)
        # Run global detection for feedback so user can judge if distractors are being selected.
        obj = detect_pink_object(hsv, learning_params, search_roi=None)
        black = detect_black_line(
            hsv,
            obj.contour,
            learning_params,
            black_hint_bbox=black_roi,
        )

        black_center = (np.nan, np.nan)
        black_in_hint = False
        if black.box_points is not None:
            center = np.mean(black.box_points.astype(np.float32), axis=0)
            black_center = (float(center[0]), float(center[1]))
            black_in_hint = _point_in_roi(black_center, black_roi)

        prediction_in_object_roi = False
        if obj.contour is not None:
            x, y, w, h = cv2.boundingRect(obj.contour)
            predicted_center = (float(x + w / 2.0), float(y + h / 2.0))
            prediction_in_object_roi = _point_in_roi(predicted_center, object_roi)

        overlay = _draw_learning_overlay(
            frame=frame,
            object_roi=object_roi,
            black_roi=black_roi,
            distractor_roi=distractor_roi,
            object_contour=obj.contour,
            black_box=black.box_points,
            object_ranges=object_ranges,
            object_found=bool(obj.found),
            prediction_in_object_roi=prediction_in_object_roi,
            black_found=bool(black.found),
            black_in_hint=black_in_hint,
            color_name=color_name,
        )

        decision = _confirm_overlay(overlay)
        if decision == "cancel":
            raise RuntimeError("Color learning canceled by user during feedback confirmation.")
        if decision == "redo":
            print("Model judgment marked wrong. Relearning with new feedback...")
            continue

        sample = {
            "round_index": 1,
            "frame_index": int(frame_index),
            "video_path": str(Path(video_path)),
            "object_roi": [
                int(object_roi[0]),
                int(object_roi[1]),
                int(object_roi[2]),
                int(object_roi[3]),
            ],
            "black_roi": [
                int(black_roi[0]),
                int(black_roi[1]),
                int(black_roi[2]),
                int(black_roi[3]),
            ],
            "distractor_roi": (
                [
                    int(distractor_roi[0]),
                    int(distractor_roi[1]),
                    int(distractor_roi[2]),
                    int(distractor_roi[3]),
                ]
                if distractor_roi is not None
                else None
            ),
            "object_ranges": serialize_hsv_ranges(object_ranges),
            "black_range": serialize_hsv_range(black_range),
            "used_black_override": False,
            "object_found": bool(obj.found),
            "prediction_in_object_roi": bool(prediction_in_object_roi),
            "black_found": bool(black.found),
            "black_center": [float(black_center[0]), float(black_center[1])],
            "black_center_in_roi": bool(black_in_hint),
        }
        print("Accepted interactive learning parameters.")
        break

    profile = {
        "color": color_name,
        "learned": True,
        "first_seen_at": first_seen_at,
        "updated_at": _now_iso(),
        "source_video": str(Path(video_path)),
        "selected_sample_index": 0,
        "object_ranges": sample["object_ranges"],
        "black_range": sample["black_range"],
        "samples": [sample],
        "bootstrap": False,
    }
    return profile
