from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class TrackingState:
    last_theta_for_display: float = np.nan
    last_valid_theta: float = np.nan
    last_output_centroid: Tuple[float, float] = (np.nan, np.nan)
    last_anomaly_status: str = "NONE"
    anomaly_streak: int = 0


def point_is_finite(point_xy: Tuple[float, float]) -> bool:
    return bool(np.isfinite(point_xy[0]) and np.isfinite(point_xy[1]))
