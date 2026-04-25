from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from Video_Analysi_Code.calibration_ui import select_ground_line
from Video_Analysi_Code.color_ml import train_dark_red_gaussian_model
from Video_Analysi_Code.color_profiles import (
    serialize_hsv_range,
    serialize_hsv_ranges,
)
from Video_Analysi_Code.tracker_detection import (
    detect_black_line,
    detect_pink_object,
    preprocess_hsv,
)
from Video_Analysi_Code.tracker_models import TrackerParams, WINDOW_CALIB

HSVTriplet = Tuple[int, int, int]
HSVRange = Tuple[HSVTriplet, HSVTriplet]
ROI = Tuple[int, int, int, int]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _default_line_range_for_color(
    color_name: str,
    base_params: TrackerParams,
) -> HSVRange:
    if color_name == "blue":
        return (
            (0, 0, int(base_params.white_line_learning_v_floor)),
            (180, int(base_params.white_line_learning_s_cap), 255),
        )
    return base_params.black_range


def _roi_is_valid(roi_xywh: ROI) -> bool:
    return roi_xywh[2] > 0 and roi_xywh[3] > 0


def _paint_mask_on_roi(
    frame: np.ndarray,
    roi_xywh: ROI,
    title: str,
    prompt: str,
    brush_size: int = 11,
) -> Optional[np.ndarray]:
    clipped = _clip_roi_to_frame(roi_xywh, frame.shape[:2])
    if clipped is None:
        return None

    x, y, w, h = clipped
    roi_frame = frame[y : y + h, x : x + w].copy()
    fg_mask = np.zeros((h, w), dtype=np.uint8)
    bg_mask = np.zeros((h, w), dtype=np.uint8)
    state = {"brush": max(3, int(brush_size))}

    def _paint(mask: np.ndarray, px: int, py: int) -> None:
        cv2.circle(mask, (px, py), state["brush"], 255, thickness=-1, lineType=cv2.LINE_AA)

    def _on_mouse(event: int, px: int, py: int, flags: int, _userdata: object) -> None:
        nonlocal fg_mask, bg_mask
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and (flags & cv2.EVENT_FLAG_LBUTTON):
            _paint(fg_mask, px, py)
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MOUSEMOVE) and (flags & cv2.EVENT_FLAG_RBUTTON):
            _paint(bg_mask, px, py)
        elif event == cv2.EVENT_MBUTTONDOWN:
            fg_mask = np.zeros_like(fg_mask)
            bg_mask = np.zeros_like(bg_mask)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, _on_mouse)

    try:
        while True:
            fg_overlay = np.zeros_like(roi_frame)
            bg_overlay = np.zeros_like(roi_frame)
            fg_overlay[fg_mask > 0] = (0, 255, 0)
            bg_overlay[bg_mask > 0] = (0, 0, 255)

            preview = cv2.addWeighted(roi_frame, 1.0, fg_overlay, 0.45, 0.0)
            preview = cv2.addWeighted(preview, 1.0, bg_overlay, 0.35, 0.0)
            cv2.rectangle(preview, (0, 0), (w - 1, h - 1), (255, 255, 255), 1)
            cv2.putText(
                preview,
                prompt,
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                f"Brush {state['brush']} | Left=fg Right=bg Middle=clear +/-=brush Enter=accept",
                (10, max(44, h - 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(title, preview)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10):
                break
            if key in (27, ord("q")):
                cv2.destroyWindow(title)
                return None
            if key == ord("r"):
                fg_mask = np.zeros_like(fg_mask)
                bg_mask = np.zeros_like(bg_mask)
            if key in (ord("+"), ord("=")):
                state["brush"] = min(61, state["brush"] + 2)
            if key in (ord("-"), ord("_")):
                state["brush"] = max(3, state["brush"] - 2)
    finally:
        cv2.destroyWindow(title)

    final_mask = fg_mask.copy()
    final_mask[bg_mask > 0] = 0
    if not np.any(final_mask):
        return None
    return final_mask


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


def _extract_hsv_pixels_from_mask(
    hsv: np.ndarray,
    roi_xywh: ROI,
    roi_mask: Optional[np.ndarray],
) -> np.ndarray:
    clipped = _clip_roi_to_frame(roi_xywh, hsv.shape[:2])
    if clipped is None:
        return np.empty((0, 3), dtype=np.int32)

    x, y, w, h = clipped
    roi_pixels = hsv[y : y + h, x : x + w]
    if roi_pixels.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    if roi_mask is None or roi_mask.shape[:2] != (h, w):
        return roi_pixels.reshape(-1, 3).astype(np.int32)

    painted = roi_mask > 0
    if not np.any(painted):
        return np.empty((0, 3), dtype=np.int32)

    return roi_pixels[painted].reshape(-1, 3).astype(np.int32)


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

    allowed_hue = object_hist > (distractor_hist * 1.15)
    kept = object_pixels[allowed_hue[object_h]]

    minimum_kept = max(200, int(object_pixels.shape[0] * 0.15))
    if kept.shape[0] < minimum_kept:
        return object_pixels
    return kept


def _filter_object_pixels_for_learning(
    object_pixels: np.ndarray,
    color_name: str,
    base_params: TrackerParams,
) -> np.ndarray:
    if object_pixels.shape[0] == 0:
        return object_pixels

    if color_name != "blue":
        return object_pixels

    sat = object_pixels[:, 1]
    val = object_pixels[:, 2]

    strict_mask = (
        sat >= int(base_params.blue_learning_s_floor)
    ) & (
        val >= int(base_params.blue_learning_v_floor)
    )
    strict_pixels = object_pixels[strict_mask]
    if strict_pixels.shape[0] >= 120:
        return strict_pixels

    # Relax slightly if user selects a tight ROI with few pixels.
    relaxed_mask = (
        sat >= max(10, int(base_params.blue_learning_s_floor) - 10)
    ) & (
        val >= max(10, int(base_params.blue_learning_v_floor) - 10)
    )
    return object_pixels[relaxed_mask]


def _filter_line_pixels_for_learning(
    line_pixels: np.ndarray,
    color_name: str,
    base_params: TrackerParams,
) -> np.ndarray:
    if line_pixels.shape[0] == 0:
        return line_pixels

    if color_name != "blue":
        return line_pixels

    sat = line_pixels[:, 1]
    val = line_pixels[:, 2]

    strict_mask = (
        sat <= int(base_params.white_line_learning_s_cap)
    ) & (
        val >= int(base_params.white_line_learning_v_floor)
    )
    strict_pixels = line_pixels[strict_mask]
    if strict_pixels.shape[0] >= 60:
        return strict_pixels

    relaxed_mask = (
        sat <= min(255, int(base_params.white_line_learning_s_cap) + 15)
    ) & (
        val >= max(0, int(base_params.white_line_learning_v_floor) - 20)
    )
    return line_pixels[relaxed_mask]


def _estimate_object_hsv_ranges_from_pixels(
    object_pixels: np.ndarray,
    color_name: str,
) -> Tuple[HSVRange, ...]:
    if object_pixels.shape[0] == 0:
        raise RuntimeError("Object ROI is empty. Cannot estimate HSV ranges.")

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

    if color_name == "blue":
        s_low = max(45, s_low)
        v_low = max(35, v_low)

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


def _estimate_line_hsv_range_from_pixels(
    line_pixels: np.ndarray,
    color_name: str,
    base_params: TrackerParams,
) -> HSVRange:
    if line_pixels.shape[0] == 0:
        return _default_line_range_for_color(color_name, base_params)

    if color_name != "blue":
        return _default_line_range_for_color(color_name, base_params)

    sat = line_pixels[:, 1]
    val = line_pixels[:, 2]

    _s_low, s_high = _percentile_bounds(sat, 2.0, 97.0, 4, 0, 255)
    v_low, v_high = _percentile_bounds(val, 4.0, 99.0, 6, 0, 255)

    s_high = min(int(base_params.white_line_learning_s_cap), s_high)
    v_low = max(int(base_params.white_line_learning_v_floor), v_low)
    if s_high < 10:
        s_high = 10

    return ((0, 0, v_low), (180, s_high, max(v_high, v_low)))


def _estimate_object_hsv_ranges_from_roi(
    frame: np.ndarray,
    object_roi: ROI,
    color_name: str,
    distractor_roi: Optional[ROI] = None,
    base_params: Optional[TrackerParams] = None,
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

    if base_params is not None:
        object_pixels = _filter_object_pixels_for_learning(object_pixels, color_name, base_params)

    return _estimate_object_hsv_ranges_from_pixels(object_pixels, color_name)


def _estimate_line_hsv_range_from_roi(
    frame: np.ndarray,
    line_roi: ROI,
    color_name: str,
    base_params: TrackerParams,
) -> HSVRange:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    line_pixels = _extract_hsv_pixels(hsv, line_roi)
    line_pixels = _filter_line_pixels_for_learning(line_pixels, color_name, base_params)
    return _estimate_line_hsv_range_from_pixels(line_pixels, color_name, base_params)


def _aggregate_object_ranges(
    object_pixel_sets: Sequence[np.ndarray],
    color_name: str,
) -> Tuple[HSVRange, ...]:
    valid_sets = [item for item in object_pixel_sets if item.shape[0] > 0]
    if not valid_sets:
        raise RuntimeError("No valid object pixels were collected for aggregation.")

    merged = np.vstack(valid_sets).astype(np.int32)
    return _estimate_object_hsv_ranges_from_pixels(merged, color_name)


def _aggregate_line_range(
    line_pixel_sets: Sequence[np.ndarray],
    color_name: str,
    base_params: TrackerParams,
) -> HSVRange:
    valid_sets = [item for item in line_pixel_sets if item.shape[0] > 0]
    if not valid_sets:
        return _default_line_range_for_color(color_name, base_params)

    merged = np.vstack(valid_sets).astype(np.int32)
    return _estimate_line_hsv_range_from_pixels(merged, color_name, base_params)


def select_learning_rois(
    frame: np.ndarray,
) -> Optional[Tuple[ROI, ROI, Optional[ROI], Optional[np.ndarray], Optional[np.ndarray]]]:
    print("Draw a coarse ROI #1 for the Flipo Flip body, then press ENTER or SPACE.")
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

    object_mask = _paint_mask_on_roi(
        frame=frame,
        roi_xywh=object_roi,
        title=f"{WINDOW_CALIB} - Paint Object",
        prompt="Paint blue body: left=foreground, right=background",
    )
    if object_mask is None:
        return None

    print("Draw a coarse ROI #2 for the short line (black/white), then press ENTER or SPACE.")
    line_roi_raw = cv2.selectROI(WINDOW_CALIB, frame, fromCenter=False, showCrosshair=True)
    line_roi = (
        int(line_roi_raw[0]),
        int(line_roi_raw[1]),
        int(line_roi_raw[2]),
        int(line_roi_raw[3]),
    )
    if not _roi_is_valid(line_roi):
        cv2.destroyWindow(WINDOW_CALIB)
        return None

    line_mask = _paint_mask_on_roi(
        frame=frame,
        roi_xywh=line_roi,
        title=f"{WINDOW_CALIB} - Paint Short Line",
        prompt="Paint short line: left=foreground, right=background",
    )
    if line_mask is None:
        return None

    print(
        "Draw ROI #3 for distractor region (arm/background) to exclude it. "
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

    return object_roi, line_roi, distractor_roi, object_mask, line_mask


def _point_in_roi(point_xy: Tuple[float, float], roi_xywh: ROI) -> bool:
    x, y, w, h = roi_xywh
    px, py = point_xy
    return bool(x <= px <= x + w and y <= py <= y + h)


def _prompt_manual_reannotation(reason: str) -> bool:
    while True:
        raw = input(f"[UNCERTAIN] {reason}. Re-annotate current frame? [Y/n]: ").strip().lower()
        if raw in {"", "y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please type 'y' or 'n'.")


def _draw_learning_overlay(
    frame: np.ndarray,
    object_roi: ROI,
    line_roi: ROI,
    distractor_roi: Optional[ROI],
    object_contour: Optional[np.ndarray],
    line_box: Optional[np.ndarray],
    object_ranges: Tuple[HSVRange, ...],
    line_range: HSVRange,
    object_found: bool,
    prediction_in_object_roi: bool,
    line_found: bool,
    line_in_hint: bool,
    color_name: str,
    video_label: str,
    round_index: int,
    round_total: int,
) -> np.ndarray:
    overlay = frame.copy()

    ox, oy, ow, oh = object_roi
    bx, by, bw, bh = line_roi

    cv2.rectangle(overlay, (ox, oy), (ox + ow, oy + oh), (60, 220, 80), 2, cv2.LINE_AA)
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (160, 220, 255), 1, cv2.LINE_AA)
    if distractor_roi is not None:
        dx, dy, dw, dh = distractor_roi
        cv2.rectangle(overlay, (dx, dy), (dx + dw, dy + dh), (20, 80, 255), 2, cv2.LINE_AA)

    if object_contour is not None:
        cv2.drawContours(overlay, [object_contour], -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(object_contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)

    if line_box is not None:
        x, y, w, h = cv2.boundingRect(line_box.astype(np.int32))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2, cv2.LINE_AA)

    range_descriptions = [
        (
            f"Obj R{idx + 1}: "
            f"H[{lower[0]},{upper[0]}] "
            f"S[{lower[1]},{upper[1]}] "
            f"V[{lower[2]},{upper[2]}]"
        )
        for idx, (lower, upper) in enumerate(object_ranges)
    ]
    line_desc = (
        f"Line R: H[{line_range[0][0]},{line_range[1][0]}] "
        f"S[{line_range[0][1]},{line_range[1][1]}] "
        f"V[{line_range[0][2]},{line_range[1][2]}]"
    )

    lines = [
        f"Interactive learning: {color_name}",
        f"Video: {video_label}   Round: {round_index}/{round_total}",
        (
            "Model judgment (targets ROI #1): "
            f"{'yes' if prediction_in_object_roi else 'no'}"
        ),
        f"Object found: {'yes' if object_found else 'no'}",
        f"Short-line found: {'yes' if line_found else 'no'}",
        f"Short-line center in ROI #2: {'yes' if line_in_hint else 'no'}",
        "ENTER: accept   R: redo   ESC/Q: cancel",
        line_desc,
    ] + range_descriptions

    y0 = 26
    for line in lines:
        cv2.putText(
            overlay,
            line,
            (12, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
        y0 += 23

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


def _collect_learning_samples_for_video(
    video_path: str,
    color_name: str,
    base_params: TrackerParams,
    round_count: int,
    start_round_index: int = 1,
) -> Tuple[List[Dict[str, object]], List[np.ndarray], List[np.ndarray], Optional[Dict[str, object]]]:
    required_rounds = max(1, int(round_count))
    accepted_samples: List[Dict[str, object]] = []
    object_pixel_sets: List[np.ndarray] = []
    line_pixel_sets: List[np.ndarray] = []
    red_ml_model: Optional[Dict[str, object]] = None

    # Ground calibration for blue objects
    ground_y: Optional[int] = None
    if color_name == "blue":
        print("Calibrating ground line for blue object filtering...")
        frame_for_ground = _get_first_frame(video_path)
        if frame_for_ground is not None:
            ground_y = select_ground_line(frame_for_ground)
            if ground_y is None:
                raise RuntimeError("Ground calibration canceled by user.")
            print(f"Ground line set at Y={ground_y}")
        else:
            print("Warning: Could not load frame for ground calibration, skipping ground filtering.")

    print(
        f"Collecting {required_rounds} accepted samples from video: {Path(video_path).name}"
    )

    while len(accepted_samples) < required_rounds:
        local_round = len(accepted_samples) + 1
        global_round = start_round_index + len(accepted_samples)
        print(
            f"[Learning] video={Path(video_path).name} "
            f"round={local_round}/{required_rounds}"
        )

        frame_selection = select_calibration_frame(video_path)
        if frame_selection is None:
            raise RuntimeError("Color learning canceled by user during frame selection.")

        frame, frame_index = frame_selection

        roi_selection = select_learning_rois(frame)
        if roi_selection is None:
            raise RuntimeError("Color learning canceled by user during ROI selection.")

        object_roi, line_roi, distractor_roi, object_mask, line_mask = roi_selection

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv_for_learning = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        object_pixels = _extract_hsv_pixels_from_mask(hsv_for_learning, object_roi, object_mask)
        if object_pixels.shape[0] == 0:
            if _prompt_manual_reannotation("Object ROI is empty"):
                continue

        distractor_pixels = (
            _extract_hsv_pixels(hsv_for_learning, distractor_roi)
            if distractor_roi is not None
            else np.empty((0, 3), dtype=np.int32)
        )
        object_pixels = _remove_distractor_like_pixels(object_pixels, distractor_pixels)
        filtered_object_pixels = _filter_object_pixels_for_learning(object_pixels, color_name, base_params)

        line_pixels = _extract_hsv_pixels_from_mask(hsv_for_learning, line_roi, line_mask)
        filtered_line_pixels = _filter_line_pixels_for_learning(line_pixels, color_name, base_params)

        if filtered_object_pixels.shape[0] < 120:
            if _prompt_manual_reannotation(
                "Object pixels are too noisy or too dark; please tighten ROI on blue body"
            ):
                continue

        if filtered_line_pixels.shape[0] < 60:
            if _prompt_manual_reannotation(
                "Line pixels are uncertain; please tighten ROI on white short line"
            ):
                continue

        object_ranges_round = _estimate_object_hsv_ranges_from_pixels(filtered_object_pixels, color_name)
        line_range_round = _estimate_line_hsv_range_from_pixels(filtered_line_pixels, color_name, base_params)

        if color_name == "dark_red":
            red_candidate = train_dark_red_gaussian_model(
                frame=frame,
                object_roi=object_roi,
                distractor_roi=distractor_roi,
            )
            if isinstance(red_candidate, dict):
                red_ml_model = red_candidate

        learning_params = replace(
            base_params,
            color_name=color_name,
            object_ranges=object_ranges_round,
            black_range=line_range_round,
            red_ml_model=red_ml_model,
        )

        hsv_for_detect = preprocess_hsv(frame, learning_params)
        obj = detect_pink_object(hsv_for_detect, learning_params, search_roi=None)
        line_det = detect_black_line(
            hsv_for_detect,
            obj.contour,
            learning_params,
            black_hint_bbox=line_roi,
        )

        line_center = (np.nan, np.nan)
        line_in_hint = False
        if line_det.box_points is not None:
            center = np.mean(line_det.box_points.astype(np.float32), axis=0)
            line_center = (float(center[0]), float(center[1]))
            line_in_hint = _point_in_roi(line_center, line_roi)

        prediction_in_object_roi = False
        if obj.contour is not None:
            x, y, w, h = cv2.boundingRect(obj.contour)
            predicted_center = (float(x + w / 2.0), float(y + h / 2.0))
            prediction_in_object_roi = _point_in_roi(predicted_center, object_roi)

        uncertain_reasons: List[str] = []
        if not bool(obj.found):
            uncertain_reasons.append("object_not_found")
        if not bool(prediction_in_object_roi):
            uncertain_reasons.append("object_mismatch")
        if not bool(line_det.found):
            uncertain_reasons.append("line_not_found")
        if bool(line_det.found) and not bool(line_in_hint):
            uncertain_reasons.append("line_outside_hint")

        if uncertain_reasons:
            reason_text = ", ".join(uncertain_reasons)
            if _prompt_manual_reannotation(
                f"Model uncertain ({reason_text})"
            ):
                continue

        overlay = _draw_learning_overlay(
            frame=frame,
            object_roi=object_roi,
            line_roi=line_roi,
            distractor_roi=distractor_roi,
            object_contour=obj.contour,
            line_box=line_det.box_points,
            object_ranges=object_ranges_round,
            line_range=line_range_round,
            object_found=bool(obj.found),
            prediction_in_object_roi=prediction_in_object_roi,
            line_found=bool(line_det.found),
            line_in_hint=line_in_hint,
            color_name=color_name,
            video_label=Path(video_path).name,
            round_index=local_round,
            round_total=required_rounds,
        )

        decision = _confirm_overlay(overlay)
        if decision == "cancel":
            raise RuntimeError("Color learning canceled by user during feedback confirmation.")
        if decision == "redo":
            print("Model judgment marked wrong. Re-annotate this round.")
            continue

        sample = {
            "round_index": int(global_round),
            "frame_index": int(frame_index),
            "video_path": str(Path(video_path)),
            "object_roi": [
                int(object_roi[0]),
                int(object_roi[1]),
                int(object_roi[2]),
                int(object_roi[3]),
            ],
            "black_roi": [
                int(line_roi[0]),
                int(line_roi[1]),
                int(line_roi[2]),
                int(line_roi[3]),
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
            "object_ranges": serialize_hsv_ranges(object_ranges_round),
            "black_range": serialize_hsv_range(line_range_round),
            "red_ml_enabled": bool(isinstance(red_ml_model, dict)),
            "used_black_override": False,
            "object_found": bool(obj.found),
            "prediction_in_object_roi": bool(prediction_in_object_roi),
            "black_found": bool(line_det.found),
            "black_center": [float(line_center[0]), float(line_center[1])],
            "black_center_in_roi": bool(line_in_hint),
        }

        accepted_samples.append(sample)
        object_pixel_sets.append(filtered_object_pixels)
        line_pixel_sets.append(filtered_line_pixels)
        print(
            f"Accepted sample {local_round}/{required_rounds} "
            f"for {Path(video_path).name}."
        )

    return accepted_samples, object_pixel_sets, line_pixel_sets, red_ml_model


def _build_profile_payload(
    color_name: str,
    source_video: str,
    first_seen_at: str,
    object_ranges: Tuple[HSVRange, ...],
    line_range: HSVRange,
    red_ml_model: Optional[Dict[str, object]],
    samples: List[Dict[str, object]],
) -> Dict[str, object]:
    selected_sample_index = len(samples) - 1 if samples else None

    return {
        "color": color_name,
        "learned": True,
        "first_seen_at": first_seen_at,
        "updated_at": _now_iso(),
        "source_video": source_video,
        "selected_sample_index": selected_sample_index,
        "object_ranges": serialize_hsv_ranges(object_ranges),
        "black_range": serialize_hsv_range(line_range),
        "red_ml_model": red_ml_model,
        "samples": samples,
        "bootstrap": False,
    }


def run_first_time_color_learning(
    video_path: str,
    color_name: str,
    base_params: TrackerParams,
    round_count: int = 1,
) -> Dict[str, object]:
    first_seen_at = _now_iso()
    rounds = max(1, int(round_count))

    print(
        f"First time detecting color '{color_name}'. "
        "Starting interactive feedback learning (coarse box + paint mask, no manual HSV input)."
    )
    print(
        "Workflow: draw a coarse body ROI, paint the body pixels, draw a coarse line ROI, paint the short line pixels. "
        "If model is uncertain, you will be asked whether to re-annotate."
    )

    samples, object_pixel_sets, line_pixel_sets, red_ml_model = _collect_learning_samples_for_video(
        video_path=video_path,
        color_name=color_name,
        base_params=base_params,
        round_count=rounds,
        start_round_index=1,
    )

    object_ranges = _aggregate_object_ranges(object_pixel_sets, color_name)
    line_range = _aggregate_line_range(line_pixel_sets, color_name, base_params)

    print("Aggregated object HSV ranges:")
    for idx, (lower, upper) in enumerate(object_ranges, start=1):
        print(f"  range {idx}: LOWER={lower} UPPER={upper}")
    print(f"Aggregated line HSV range: LOWER={line_range[0]} UPPER={line_range[1]}")

    return _build_profile_payload(
        color_name=color_name,
        source_video=str(Path(video_path)),
        first_seen_at=first_seen_at,
        object_ranges=object_ranges,
        line_range=line_range,
        red_ml_model=red_ml_model,
        samples=samples,
    )


def run_multi_video_color_learning(
    video_paths: Sequence[str],
    color_name: str,
    base_params: TrackerParams,
    round_count: int = 8,
) -> Dict[str, object]:
    if not video_paths:
        raise RuntimeError("No learning videos were provided.")

    first_seen_at = _now_iso()
    rounds_per_video = max(1, int(round_count))

    all_samples: List[Dict[str, object]] = []
    all_object_pixels: List[np.ndarray] = []
    all_line_pixels: List[np.ndarray] = []
    red_ml_model: Optional[Dict[str, object]] = None

    print(
        f"Starting multi-video learning for '{color_name}'. "
        f"Videos: {len(video_paths)}, rounds/video: {rounds_per_video}"
    )

    for video_path in video_paths:
        video_obj = Path(video_path)
        if not video_obj.exists() or not video_obj.is_file():
            raise RuntimeError(f"Learning video does not exist: {video_path}")

        start_round_index = len(all_samples) + 1
        samples, object_pixels, line_pixels, red_candidate = _collect_learning_samples_for_video(
            video_path=video_path,
            color_name=color_name,
            base_params=base_params,
            round_count=rounds_per_video,
            start_round_index=start_round_index,
        )

        if isinstance(red_candidate, dict):
            red_ml_model = red_candidate

        all_samples.extend(samples)
        all_object_pixels.extend(object_pixels)
        all_line_pixels.extend(line_pixels)

    object_ranges = _aggregate_object_ranges(all_object_pixels, color_name)
    line_range = _aggregate_line_range(all_line_pixels, color_name, base_params)

    print("Aggregated object HSV ranges across videos:")
    for idx, (lower, upper) in enumerate(object_ranges, start=1):
        print(f"  range {idx}: LOWER={lower} UPPER={upper}")
    print(f"Aggregated line HSV range across videos: LOWER={line_range[0]} UPPER={line_range[1]}")

    return _build_profile_payload(
        color_name=color_name,
        source_video=str(Path(video_paths[0])),
        first_seen_at=first_seen_at,
        object_ranges=object_ranges,
        line_range=line_range,
        red_ml_model=red_ml_model,
        samples=all_samples,
    )
