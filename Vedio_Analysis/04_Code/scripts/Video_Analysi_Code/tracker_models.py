from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class TrackerParams:
    """Runtime parameters for color segmentation and visualization."""

    blur_kernel: Tuple[int, int] = (7, 7)
    color_name: str = "pink"
    object_ranges: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...] = (
        ((140, 50, 50), (179, 255, 255)),
        ((125, 40, 40), (139, 255, 255)),
    )
    black_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
        (0, 0, 0),
        (180, 255, 75),
    )
    pink_min_area: float = 300.0
    black_min_area: float = 20.0
    black_min_area_ratio: float = 0.0025
    black_min_aspect_ratio: float = 1.65
    black_preferred_aspect_ratio: float = 2.1
    black_min_long_edge_px: float = 5.0
    black_min_long_edge_ratio_of_pink_min_side: float = 0.25
    black_min_edge_ratio_for_orientation: float = 1.22
    max_theta_step_deg: float = 25.0
    max_theta_step_deg_low_aspect: float = 12.0
    max_blue_center_offset_ratio: float = 0.40
    max_blue_diag_distance_ratio: float = 0.18
    max_red_center_offset_ratio: float = 0.22
    red_center_blend_ok: float = 0.30
    max_center_step_px_ok: float = 22.0
    max_center_step_px_unreliable: float = 10.0
    morph_kernel_size: int = 5
    roi_padding: int = 12
    wait_key_ms: int = 1

    @property
    def pink_ranges(
        self,
    ) -> Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...]:
        """Backward-compatible alias for historical naming."""
        return self.object_ranges

    @pink_ranges.setter
    def pink_ranges(
        self,
        ranges: Tuple[Tuple[Tuple[int, int, int], Tuple[int, int, int]], ...],
    ) -> None:
        self.object_ranges = ranges


@dataclass
class PinkDetection:
    found: bool
    centroid: Tuple[float, float]
    contour: Optional[np.ndarray]
    area: float


@dataclass
class BlackDetection:
    found: bool
    theta_deg: float
    box_points: Optional[np.ndarray]
    area: float
    aspect_ratio: float
    long_edge_px: float


WINDOW_TRACK = "Flipo Flip Tracker"
WINDOW_CALIB = "Calibration"
