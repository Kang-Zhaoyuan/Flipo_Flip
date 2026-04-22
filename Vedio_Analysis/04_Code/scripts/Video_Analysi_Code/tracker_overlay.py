from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def format_angle_text(value: float) -> str:
    if np.isnan(value):
        return "NaN"
    return f"{value:.3f}"


def draw_overlay(
    frame: np.ndarray,
    frame_index: int,
    timestamp_s: float,
    pink_contour: Optional[np.ndarray],
    centroid: Tuple[float, float],
    black_box: Optional[np.ndarray],
    theta_raw: float,
    theta_display: float,
    k_mm_per_pixel: float,
) -> np.ndarray:
    overlay = frame.copy()

    if pink_contour is not None:
        cv2.drawContours(overlay, [pink_contour], -1, (60, 220, 60), 2)
        x, y, w, h = cv2.boundingRect(pink_contour)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)

    if not np.isnan(centroid[0]) and not np.isnan(centroid[1]):
        cx = int(round(centroid[0]))
        cy = int(round(centroid[1]))
        cv2.circle(overlay, (cx, cy), 6, (0, 0, 255), -1)

    if black_box is not None:
        cv2.drawContours(overlay, [black_box.astype(np.int32)], -1, (255, 120, 0), 2)

    text_lines = [
        f"frame={frame_index}",
        f"t={timestamp_s:.4f}s",
        f"x={format_angle_text(centroid[0])} px  y={format_angle_text(centroid[1])} px",
        f"theta(raw)={format_angle_text(theta_raw)} deg",
        f"theta(display)={format_angle_text(theta_display)} deg",
        f"K={k_mm_per_pixel:.6f} mm/px",
        "SPACE: pause/resume   Q/ESC: quit",
    ]

    y0 = 24
    for line in text_lines:
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
