from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np

from Video_Analysi_Code.tracker_models import BlackDetection, TrackerParams


def normalize_angle_deg(angle_deg: float) -> float:
    return (angle_deg + 180.0) % 360.0 - 180.0


def lift_axis_angle_near_reference(angle_deg: float, reference_deg: float) -> float:
    if not np.isfinite(reference_deg):
        return normalize_angle_deg(angle_deg)

    half_turns = round((reference_deg - angle_deg) / 180.0)
    return float(angle_deg + 180.0 * half_turns)


def constrain_center_step(
    candidate_xy: Tuple[float, float],
    previous_xy: Tuple[float, float],
    max_step_px: float,
) -> Tuple[Tuple[float, float], bool]:
    coords = np.array(
        [candidate_xy[0], candidate_xy[1], previous_xy[0], previous_xy[1]],
        dtype=np.float64,
    )
    if max_step_px <= 0.0 or not np.isfinite(coords).all():
        return candidate_xy, False

    delta_x = float(candidate_xy[0] - previous_xy[0])
    delta_y = float(candidate_xy[1] - previous_xy[1])
    step = float(np.hypot(delta_x, delta_y))
    if step <= max_step_px:
        return candidate_xy, False

    scale = max_step_px / step
    clipped_xy = (
        float(previous_xy[0] + delta_x * scale),
        float(previous_xy[1] + delta_y * scale),
    )
    return clipped_xy, True


def pick_orientation_closest_to_previous(theta_deg: float, previous_theta: float) -> float:
    principal_theta = normalize_angle_deg(theta_deg)
    if np.isnan(previous_theta):
        return principal_theta

    return lift_axis_angle_near_reference(principal_theta, previous_theta)


def long_axis_theta_from_box(box_points: np.ndarray) -> float:
    points = box_points.astype(np.float32)
    edge0 = points[1] - points[0]
    edge1 = points[2] - points[1]

    if np.linalg.norm(edge0) >= np.linalg.norm(edge1):
        axis_vector = edge0
    else:
        axis_vector = edge1

    # Angle convention: 0 deg along +Y (up), clockwise positive.
    theta_deg = float(np.degrees(np.arctan2(axis_vector[0], -axis_vector[1])))
    return normalize_angle_deg(theta_deg)


def point_line_distance(
    point_xy: Tuple[float, float],
    line_a: Tuple[float, float],
    line_b: Tuple[float, float],
) -> float:
    p = np.array(point_xy, dtype=np.float64)
    a = np.array(line_a, dtype=np.float64)
    b = np.array(line_b, dtype=np.float64)
    ab = b - a
    denom = float(np.linalg.norm(ab))
    if denom <= 1e-9:
        return float("inf")
    return float(abs(np.cross(ab, p - a)) / denom)


def min_distance_to_bbox_diagonals(
    point_xy: Tuple[float, float],
    bbox: Tuple[int, int, int, int],
) -> float:
    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return float("inf")

    p_tl = (float(x), float(y))
    p_br = (float(x + w), float(y + h))
    p_tr = (float(x + w), float(y))
    p_bl = (float(x), float(y + h))

    d1 = point_line_distance(point_xy, p_tl, p_br)
    d2 = point_line_distance(point_xy, p_tr, p_bl)
    return min(d1, d2)


def compute_oriented_box_and_center(
    contour: Optional[np.ndarray],
    theta_deg: float,
) -> Tuple[Optional[np.ndarray], Tuple[float, float]]:
    if contour is None or np.isnan(theta_deg):
        return None, (np.nan, np.nan)

    points = contour.reshape(-1, 2).astype(np.float32)
    if points.size == 0:
        return None, (np.nan, np.nan)

    theta_rad = float(np.deg2rad(theta_deg))
    sin_t = float(np.sin(theta_rad))
    cos_t = float(np.cos(theta_rad))

    u = np.array([sin_t, -cos_t], dtype=np.float32)
    v = np.array([cos_t, sin_t], dtype=np.float32)

    proj_u = points @ u
    proj_v = points @ v

    u_min = float(np.min(proj_u))
    u_max = float(np.max(proj_u))
    v_min = float(np.min(proj_v))
    v_max = float(np.max(proj_v))

    if not np.isfinite([u_min, u_max, v_min, v_max]).all():
        return None, (np.nan, np.nan)

    box_uv = np.array(
        [
            [u_min, v_min],
            [u_max, v_min],
            [u_max, v_max],
            [u_min, v_max],
        ],
        dtype=np.float32,
    )
    box_xy = box_uv[:, :1] * u + box_uv[:, 1:] * v

    center_u = 0.5 * (u_min + u_max)
    center_v = 0.5 * (v_min + v_max)
    center_xy = center_u * u + center_v * v

    return box_xy.astype(np.float32), (float(center_xy[0]), float(center_xy[1]))


def adjust_center_to_blue_long_edge_band(
    center_xy: Tuple[float, float],
    blue_box: Optional[np.ndarray],
    theta_deg: float,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    cx, cy = center_xy
    if np.isnan(cx) or np.isnan(cy):
        return center_xy
    if blue_box is None or np.isnan(theta_deg):
        return center_xy

    blue_poly = blue_box.astype(np.float32)
    in_blue = cv2.pointPolygonTest(blue_poly, (float(cx), float(cy)), measureDist=False) >= 0
    if in_blue:
        return center_xy

    theta_rad = float(np.deg2rad(theta_deg))
    short_axis = np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=np.float32)

    blue_short_proj = blue_poly @ short_axis
    s_min = float(np.min(blue_short_proj))
    s_max = float(np.max(blue_short_proj))
    s_center = float(np.dot(np.array([cx, cy], dtype=np.float32), short_axis))

    s_target = float(np.clip(s_center, s_min, s_max))
    delta = s_target - s_center
    if abs(delta) <= eps:
        return center_xy

    adjusted = np.array([cx, cy], dtype=np.float32) + delta * short_axis
    return float(adjusted[0]), float(adjusted[1])


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
