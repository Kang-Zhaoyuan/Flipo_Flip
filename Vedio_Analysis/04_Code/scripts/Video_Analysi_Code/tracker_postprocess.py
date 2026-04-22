from __future__ import annotations

import numpy as np
import pandas as pd


def postprocess_angles(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["theta_unwrapped"] = []
        df["omega_rad_s"] = []
        return df

    theta_series = df["theta"].astype(float)
    theta_filled = theta_series.interpolate(method="linear", limit_direction="both")

    if theta_filled.notna().any():
        theta_rad = np.deg2rad(theta_filled.to_numpy(dtype=float))
        theta_unwrapped_rad = np.unwrap(theta_rad)
        df["theta_unwrapped"] = np.rad2deg(theta_unwrapped_rad)

        if len(df) >= 2:
            timestamps = df["timestamp"].to_numpy(dtype=float)
            df["omega_rad_s"] = np.gradient(theta_unwrapped_rad, timestamps)
        else:
            df["omega_rad_s"] = np.nan
    else:
        df["theta_unwrapped"] = np.nan
        df["omega_rad_s"] = np.nan

    df["theta"] = theta_series.round(3)
    df["theta_unwrapped"] = pd.to_numeric(df["theta_unwrapped"], errors="coerce").round(3)

    return df
