from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from Video_Analysi_Code.color_ml import build_dark_red_ml_mask
from Video_Analysi_Code.tracker_geometry import (
    long_axis_theta_from_box,
    min_distance_to_bbox_diagonals,
    pick_orientation_closest_to_previous,
)
from Video_Analysi_Code.tracker_models import BlackDetection, PinkDetection, TrackerParams


def preprocess_hsv(frame: np.ndarray, params: TrackerParams) -> np.ndarray:
    blurred = cv2.GaussianBlur(frame, params.blur_kernel, 0)
    return cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)


def clean_binary_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def _apply_blue_clahe(frame_bgr: np.ndarray) -> np.ndarray:
    lab_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    light_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    light_enhanced = clahe.apply(light_channel)
    return cv2.cvtColor(cv2.merge([light_enhanced, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def _detect_ground_line(frame_bgr: np.ndarray) -> int:
    height = frame_bgr.shape[0]
    gray_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_image, 30, 100)
    search_top = int(height * 0.60)
    row_strength = np.sum(edge_image[search_top:, :], axis=1)
    if row_strength.size == 0 or int(np.max(row_strength)) <= 0:
        return int(height * 0.92)
    return search_top + int(np.argmax(row_strength))


def _build_blue_mask(enhanced_bgr: np.ndarray) -> np.ndarray:
    hsv_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv_image, lower_blue, upper_blue)


def _solidify_and_cut(mask: np.ndarray) -> np.ndarray:
    close_kernel = np.ones((7, 7), np.uint8)
    erode_kernel = np.ones((5, 5), np.uint8)
    solid_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    eroded_mask = cv2.erode(solid_mask, erode_kernel, iterations=3)
    return cv2.dilate(eroded_mask, erode_kernel, iterations=3)


def _reject_ground_components(mask: np.ndarray, ground_y: int) -> np.ndarray:
    ground_tolerance = 15
    thin_limit = 30
    aspect_limit = 6.0

    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask)
    kept_mask = np.zeros_like(mask)

    for component_index in range(1, component_count):
        x_pos, y_pos, width, height, area = statistics[component_index, :5]
        if area < 100:
            continue

        bottom_edge = y_pos + height
        aspect_ratio = width / max(height, 1)
        near_ground = bottom_edge >= ground_y - ground_tolerance
        is_thin = height < thin_limit
        is_wide = aspect_ratio > aspect_limit

        if not (near_ground and is_thin and is_wide):
            kept_mask[labels == component_index] = 255

    return kept_mask


def _strip_thin_extensions(blob_mask: np.ndarray, min_thickness: int) -> np.ndarray:
    cleaned_mask = blob_mask.copy()

    column_fill = np.sum(cleaned_mask > 0, axis=0)
    cleaned_mask[:, column_fill < min_thickness] = 0

    row_fill = np.sum(cleaned_mask > 0, axis=1)
    cleaned_mask[row_fill < min_thickness, :] = 0

    return cleaned_mask


def _detect_blue_object(
    frame_bgr: np.ndarray,
    params: TrackerParams,
    search_roi: Optional[Tuple[int, int, int, int]] = None,
) -> PinkDetection:
    enhanced_bgr = _apply_blue_clahe(frame_bgr)
    blue_mask = _build_blue_mask(enhanced_bgr)
    morph_mask = _solidify_and_cut(blue_mask)

    ground_y = (
        int(params.blue_ground_y)
        if params.blue_ground_y is not None
        else _detect_ground_line(frame_bgr)
    )
    kept_mask = _reject_ground_components(morph_mask, ground_y)

    contour_candidates, _ = cv2.findContours(
        kept_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contour_candidates = [
        contour for contour in contour_candidates if cv2.contourArea(contour) > 200
    ]
    if not contour_candidates:
        return PinkDetection(False, (np.nan, np.nan), None, np.nan)

    largest_contour = max(contour_candidates, key=cv2.contourArea)
    single_blob_mask = np.zeros_like(kept_mask)
    cv2.drawContours(single_blob_mask, [largest_contour], -1, 255, -1)

    min_dim = min(frame_bgr.shape[:2])
    thickness_ratio = max(0.01, float(params.blue_min_thickness_ratio))
    min_thickness = max(28, int(min_dim * thickness_ratio))

    stripped_core = _strip_thin_extensions(single_blob_mask, min_thickness)
    stripped_core = cv2.morphologyEx(
        stripped_core,
        cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8),
        iterations=2,
    )

    if search_roi is not None:
        clipped = _clip_roi_to_frame(search_roi, stripped_core.shape[:2])
        if clipped is None:
            return PinkDetection(False, (np.nan, np.nan), None, np.nan)
        x, y, w, h = clipped
        roi_mask = np.zeros(stripped_core.shape, dtype=np.uint8)
        roi_mask[y : y + h, x : x + w] = 255
        stripped_core = cv2.bitwise_and(stripped_core, roi_mask)

    contours, _ = cv2.findContours(stripped_core, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 200]
    if not contours:
        return PinkDetection(False, (np.nan, np.nan), None, np.nan)

    contour = max(contours, key=cv2.contourArea)
    pink_area = float(cv2.contourArea(contour))
    if pink_area < params.pink_min_area:
        return PinkDetection(False, (np.nan, np.nan), None, np.nan)

    moments = cv2.moments(contour)
    if moments["m00"] <= 1e-9:
        x, y, w, h = cv2.boundingRect(contour)
        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
    else:
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

    return PinkDetection(True, (cx, cy), contour, pink_area)


def _clip_roi_to_frame(
    roi_xywh: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
) -> Optional[Tuple[int, int, int, int]]:
    frame_h, frame_w = frame_shape
    x, y, w, h = roi_xywh
    x0 = max(0, int(x))
    y0 = max(0, int(y))
    x1 = min(frame_w, int(x + w))
    y1 = min(frame_h, int(y + h))
    if x0 >= x1 or y0 >= y1:
        return None
    return x0, y0, x1 - x0, y1 - y0


def _contour_inside_ratio(
    contour: np.ndarray,
    parent_contour: np.ndarray,
    max_samples: int = 220,
) -> float:
    points = contour.reshape(-1, 2)
    if points.size == 0:
        return 0.0

    if points.shape[0] > max_samples:
        step = max(1, points.shape[0] // max_samples)
        points = points[::step]

    inside_count = 0
    for px, py in points:
        inside = cv2.pointPolygonTest(parent_contour, (float(px), float(py)), measureDist=False)
        if inside >= 0:
            inside_count += 1

    return float(inside_count / max(1, points.shape[0]))


def _box_inside_ratio(
    box_points: np.ndarray,
    parent_contour: np.ndarray,
) -> float:
    points = box_points.reshape(-1, 2)
    if points.size == 0:
        return 0.0

    inside_count = 0
    for px, py in points:
        inside = cv2.pointPolygonTest(parent_contour, (float(px), float(py)), measureDist=False)
        if inside >= 0:
            inside_count += 1

    return float(inside_count / max(1, points.shape[0]))


def detect_pink_object(
    hsv: np.ndarray,
    params: TrackerParams,
    search_roi: Optional[Tuple[int, int, int, int]] = None,
    frame_bgr: Optional[np.ndarray] = None,
) -> PinkDetection:
    if params.color_name == "blue" and frame_bgr is not None:
        return _detect_blue_object(frame_bgr, params, search_roi)

    if params.color_name == "dark_red" and isinstance(params.red_ml_model, dict):
        ml_mask = build_dark_red_ml_mask(hsv, params.red_ml_model)

        if search_roi is not None:
            clipped = _clip_roi_to_frame(search_roi, hsv.shape[:2])
            if clipped is None:
                return PinkDetection(False, (np.nan, np.nan), None, np.nan)
            x, y, w, h = clipped
            roi_mask = np.zeros(ml_mask.shape, dtype=np.uint8)
            roi_mask[y : y + h, x : x + w] = 255
            ml_mask = cv2.bitwise_and(ml_mask, roi_mask)

        ml_mask = clean_binary_mask(ml_mask, params.morph_kernel_size)
        ml_contours, _ = cv2.findContours(ml_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if ml_contours:
            contour = max(ml_contours, key=cv2.contourArea)
            area = float(cv2.contourArea(contour))
            if area >= params.pink_min_area:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 0 and h > 0:
                    cx = float(x + w / 2.0)
                    cy = float(y + h / 2.0)
                    return PinkDetection(True, (cx, cy), contour, area)

        # Fall back to HSV thresholding if ML did not produce a usable contour.

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

    for lower, upper in params.object_ranges:
        lower_np = np.array(lower, dtype=np.uint8)
        upper_np = np.array(upper, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_np, upper_np))

    if search_roi is not None:
        clipped = _clip_roi_to_frame(search_roi, hsv.shape[:2])
        if clipped is None:
            return PinkDetection(False, (np.nan, np.nan), None, np.nan)

        x, y, w, h = clipped
        roi_mask = np.zeros(mask.shape, dtype=np.uint8)
        roi_mask[y : y + h, x : x + w] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    mask = clean_binary_mask(mask, params.morph_kernel_size)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return PinkDetection(False, (np.nan, np.nan), None, np.nan)

    contour = max(contours, key=cv2.contourArea)
    pink_area = float(cv2.contourArea(contour))
    if pink_area < params.pink_min_area:
        return PinkDetection(False, (np.nan, np.nan), None, np.nan)

    x, y, w, h = cv2.boundingRect(contour)
    if w <= 0 or h <= 0:
        return PinkDetection(False, (np.nan, np.nan), None, np.nan)

    cx = float(x + w / 2.0)
    cy = float(y + h / 2.0)
    return PinkDetection(True, (cx, cy), contour, pink_area)


def detect_black_line(
    hsv: np.ndarray,
    pink_contour: Optional[np.ndarray],
    params: TrackerParams,
    black_hint_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> BlackDetection:
    if pink_contour is None:
        return BlackDetection(False, np.nan, None, np.nan, np.nan, np.nan)

    pink_area = float(cv2.contourArea(pink_contour))
    if pink_area <= 1e-9:
        return BlackDetection(False, np.nan, None, np.nan, np.nan, np.nan)

    frame_h, frame_w = hsv.shape[:2]
    x, y, w, h = cv2.boundingRect(pink_contour)

    x0 = max(0, x - params.roi_padding)
    y0 = max(0, y - params.roi_padding)
    x1 = min(frame_w, x + w + params.roi_padding)
    y1 = min(frame_h, y + h + params.roi_padding)

    if x0 >= x1 or y0 >= y1:
        return BlackDetection(False, np.nan, None, np.nan, np.nan, np.nan)

    roi_hsv = hsv[y0:y1, x0:x1]

    lower_black = np.array(params.black_range[0], dtype=np.uint8)
    upper_black = np.array(params.black_range[1], dtype=np.uint8)
    black_mask = cv2.inRange(roi_hsv, lower_black, upper_black)

    pink_roi_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    shifted = pink_contour.copy()
    shifted[:, 0, 0] = shifted[:, 0, 0] - x0
    shifted[:, 0, 1] = shifted[:, 0, 1] - y0
    cv2.drawContours(pink_roi_mask, [shifted], -1, 255, thickness=-1)

    black_mask = cv2.bitwise_and(black_mask, pink_roi_mask)

    if black_hint_bbox is not None:
        hint = _clip_roi_to_frame(black_hint_bbox, hsv.shape[:2])
        if hint is not None:
            hx, hy, hw, hh = hint
            hx0 = max(0, hx - x0)
            hy0 = max(0, hy - y0)
            hx1 = min(roi_hsv.shape[1], hx + hw - x0)
            hy1 = min(roi_hsv.shape[0], hy + hh - y0)
            if hx0 < hx1 and hy0 < hy1:
                hint_mask = np.zeros_like(black_mask)
                hint_mask[hy0:hy1, hx0:hx1] = 255
                black_mask = cv2.bitwise_and(black_mask, hint_mask)

    black_mask = clean_binary_mask(black_mask, 3)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return BlackDetection(False, np.nan, None, np.nan, np.nan, np.nan)

    best_score = -1.0
    best_area = np.nan
    best_aspect = np.nan
    best_long_edge = np.nan
    best_box = None

    for contour in contours:
        black_area = float(cv2.contourArea(contour))
        if black_area < params.black_min_area:
            continue

        if black_area > (params.black_max_area_ratio * pink_area):
            continue

        rect = cv2.minAreaRect(contour)
        (_cx, _cy), (w_rect, h_rect), _angle = rect
        long_edge = float(max(w_rect, h_rect))
        short_edge = float(min(w_rect, h_rect))
        if short_edge <= 1e-6:
            continue

        aspect_ratio = float(long_edge / short_edge)
        if aspect_ratio < params.black_min_aspect_ratio:
            continue
        if long_edge < params.black_min_long_edge_px:
            continue

        rect_area = float(w_rect * h_rect)
        if rect_area <= 1e-6:
            continue

        rect_fill_ratio = float(black_area / rect_area)
        if rect_fill_ratio < params.black_min_rect_fill_ratio:
            continue

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        if hull_area <= 1e-6:
            continue

        solidity = float(black_area / hull_area)
        if solidity < params.black_min_solidity:
            continue

        contour_frame = contour.copy()
        contour_frame[:, 0, 0] = contour_frame[:, 0, 0] + x0
        contour_frame[:, 0, 1] = contour_frame[:, 0, 1] + y0

        inside_ratio = _contour_inside_ratio(contour_frame, pink_contour)
        if inside_ratio < params.black_min_inside_ratio:
            continue

        box = cv2.boxPoints(rect)
        box[:, 0] += x0
        box[:, 1] += y0

        box_ratio = _box_inside_ratio(box, pink_contour)
        if box_ratio < params.black_min_inside_ratio:
            continue

        # Prefer elongated, compact candidates that stay inside the blue contour.
        score = float(
            black_area
            * min(aspect_ratio, params.black_preferred_aspect_ratio + 2.5)
            * (0.25 + 0.75 * inside_ratio)
            * (0.25 + 0.75 * solidity)
        )

        if score > best_score:
            best_score = score
            best_area = black_area
            best_aspect = aspect_ratio
            best_long_edge = long_edge
            best_box = box

    if best_box is None:
        return BlackDetection(False, np.nan, None, np.nan, np.nan, np.nan)

    theta_deg = long_axis_theta_from_box(best_box)
    return BlackDetection(True, theta_deg, best_box, best_area, best_aspect, best_long_edge)


def adjust_center_to_black_long_edge_band(
    center_xy: Tuple[float, float],
    black_box: Optional[np.ndarray],
    theta_deg: float,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    cx, cy = center_xy
    if np.isnan(cx) or np.isnan(cy):
        return center_xy
    if black_box is None or np.isnan(theta_deg):
        return center_xy

    black_poly = black_box.astype(np.float32)
    in_black = (
        cv2.pointPolygonTest(black_poly, (float(cx), float(cy)), measureDist=False) >= 0
    )
    if in_black:
        return center_xy

    theta_rad = float(np.deg2rad(theta_deg))
    short_axis = np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float32)

    black_short_proj = black_poly @ short_axis
    s_min = float(np.min(black_short_proj))
    s_max = float(np.max(black_short_proj))
    s_center = float(np.dot(np.array([cx, cy], dtype=np.float32), short_axis))

    s_target = float(np.clip(s_center, s_min, s_max))
    delta = s_target - s_center
    if abs(delta) <= eps:
        return center_xy

    adjusted = np.array([cx, cy], dtype=np.float32) + delta * short_axis
    return float(adjusted[0]), float(adjusted[1])


def adjust_center_to_blue_long_edge_band(
    center_xy: Tuple[float, float],
    blue_box: Optional[np.ndarray],
    theta_deg: float,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    # Compatibility wrapper for existing call sites.
    return adjust_center_to_black_long_edge_band(
        center_xy=center_xy,
        black_box=blue_box,
        theta_deg=theta_deg,
        eps=eps,
    )


def validate_black_detection(
    black: BlackDetection,
    pink_area: float,
    pink_bbox: Optional[Tuple[int, int, int, int]],
    pink_center: Tuple[float, float],
    last_valid_theta: float,
    params: TrackerParams,
) -> Tuple[bool, float, str]:
    if not black.found:
        return False, np.nan, "no_contour"

    if (
        not np.isfinite(black.area)
        or not np.isfinite(black.aspect_ratio)
        or not np.isfinite(black.long_edge_px)
        or not np.isfinite(pink_area)
    ):
        return False, np.nan, "non_finite"

    if black.box_points is None or pink_bbox is None:
        return False, np.nan, "missing_geometry"

    dynamic_area_threshold = max(params.black_min_area, params.black_min_area_ratio * pink_area)
    if black.area < dynamic_area_threshold:
        return False, np.nan, "area"

    if black.area > (params.black_max_area_ratio * pink_area):
        return False, np.nan, "area_large"

    if black.aspect_ratio < params.black_min_aspect_ratio:
        return False, np.nan, "aspect"

    if black.long_edge_px < params.black_min_long_edge_px:
        return False, np.nan, "long_edge"

    theta_candidate = pick_orientation_closest_to_previous(black.theta_deg, last_valid_theta)
    if np.isfinite(last_valid_theta):
        theta_step = abs(theta_candidate - last_valid_theta)
        if theta_step > params.max_theta_step_deg:
            return False, np.nan, "theta_step"

    center_xy = np.mean(black.box_points.astype(np.float32), axis=0)
    center_x = float(center_xy[0])
    center_y = float(center_xy[1])

    x, y, w, h = pink_bbox
    if not (x <= center_x <= x + w and y <= center_y <= y + h):
        return False, np.nan, "outside_green_bbox"

    bbox_diag = float(np.hypot(w, h))
    if bbox_diag > 1e-9:
        center_dist = float(np.hypot(center_x - pink_center[0], center_y - pink_center[1]))
        if center_dist > params.max_blue_center_offset_ratio * bbox_diag:
            return False, np.nan, "center_far"

    diag_dist = min_distance_to_bbox_diagonals((center_x, center_y), pink_bbox)
    diag_band = params.max_blue_diag_distance_ratio * float(min(w, h))
    if diag_dist > diag_band:
        return False, np.nan, "diag_far"

    return True, theta_candidate, "ok"
