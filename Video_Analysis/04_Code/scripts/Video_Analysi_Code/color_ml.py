from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


ROI = Tuple[int, int, int, int]


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
        return np.empty((0, 3), dtype=np.float64)

    x, y, w, h = clipped
    roi = hsv[y : y + h, x : x + w]
    if roi.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    return roi.reshape(-1, 3).astype(np.float64)


def _hsv_to_features(hsv_pixels: np.ndarray) -> np.ndarray:
    if hsv_pixels.shape[0] == 0:
        return np.empty((0, 4), dtype=np.float64)

    hue_rad = (hsv_pixels[:, 0] / 180.0) * (2.0 * np.pi)
    cos_h = np.cos(hue_rad)
    sin_h = np.sin(hue_rad)
    sat = hsv_pixels[:, 1] / 255.0
    val = hsv_pixels[:, 2] / 255.0
    return np.column_stack((cos_h, sin_h, sat, val)).astype(np.float64)


def _sample_rows(values: np.ndarray, max_rows: int, seed: int = 7) -> np.ndarray:
    if values.shape[0] <= max_rows:
        return values
    rng = np.random.default_rng(seed)
    idx = rng.choice(values.shape[0], size=max_rows, replace=False)
    return values[idx]


def _fit_gaussian(features: np.ndarray, regularization: float) -> Tuple[np.ndarray, np.ndarray, float]:
    mean = np.mean(features, axis=0)
    centered = features - mean
    cov = (centered.T @ centered) / max(1, features.shape[0] - 1)
    cov = cov + np.eye(cov.shape[0], dtype=np.float64) * regularization
    inv_cov = np.linalg.inv(cov)
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        # Numerical safeguard for degenerate covariance.
        cov = cov + np.eye(cov.shape[0], dtype=np.float64) * (regularization * 10.0)
        inv_cov = np.linalg.inv(cov)
        _, logdet = np.linalg.slogdet(cov)
    return mean, inv_cov, float(logdet)


def _gaussian_log_likelihood(
    features: np.ndarray,
    mean: np.ndarray,
    inv_cov: np.ndarray,
    logdet_cov: float,
) -> np.ndarray:
    centered = features - mean
    quad = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    # Constant term is omitted because only score differences are used.
    return -0.5 * (quad + logdet_cov)


def _build_negative_mask(
    frame_shape: Tuple[int, int],
    object_roi: ROI,
    distractor_roi: Optional[ROI],
) -> np.ndarray:
    h, w = frame_shape
    neg_mask = np.zeros((h, w), dtype=np.uint8)

    if distractor_roi is not None:
        clipped = _clip_roi_to_frame(distractor_roi, frame_shape)
        if clipped is not None:
            x, y, ww, hh = clipped
            neg_mask[y : y + hh, x : x + ww] = 255

    # Fallback: use a local ring around the object as negative examples.
    if not np.any(neg_mask):
        obj = _clip_roi_to_frame(object_roi, frame_shape)
        if obj is not None:
            x, y, ww, hh = obj
            pad_x = max(10, int(ww * 0.8))
            pad_y = max(10, int(hh * 0.8))
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(w, x + ww + pad_x)
            y1 = min(h, y + hh + pad_y)
            neg_mask[y0:y1, x0:x1] = 255
            neg_mask[y : y + hh, x : x + ww] = 0

    return neg_mask


def _select_threshold(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if pos_scores.size == 0 or neg_scores.size == 0:
        return 0.0

    candidates = np.quantile(
        np.concatenate((pos_scores, neg_scores)),
        np.linspace(0.05, 0.95, 25),
    )

    best_threshold = 0.0
    best_j = -1.0
    for threshold in candidates:
        tpr = float(np.mean(pos_scores >= threshold))
        fpr = float(np.mean(neg_scores >= threshold))
        j_stat = tpr - fpr
        if j_stat > best_j:
            best_j = j_stat
            best_threshold = float(threshold)
    return best_threshold


def train_dark_red_gaussian_model(
    frame: np.ndarray,
    object_roi: ROI,
    distractor_roi: Optional[ROI],
) -> Optional[Dict[str, object]]:
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    pos_pixels = _extract_hsv_pixels(hsv, object_roi)
    if pos_pixels.shape[0] < 120:
        return None

    neg_mask = _build_negative_mask(hsv.shape[:2], object_roi, distractor_roi)
    neg_pixels = hsv[neg_mask > 0].reshape(-1, 3).astype(np.float64)
    if neg_pixels.shape[0] < 120:
        return None

    # Keep the model lightweight and robust against noisy pixels.
    pos_pixels = pos_pixels[(pos_pixels[:, 1] >= 20) & (pos_pixels[:, 2] >= 15)]
    neg_pixels = neg_pixels[(neg_pixels[:, 2] >= 10)]
    if pos_pixels.shape[0] < 120 or neg_pixels.shape[0] < 120:
        return None

    pos_pixels = _sample_rows(pos_pixels, 6000)
    neg_pixels = _sample_rows(neg_pixels, 8000)

    pos_features = _hsv_to_features(pos_pixels)
    neg_features = _hsv_to_features(neg_pixels)

    regularization = 1e-2
    pos_mean, pos_inv_cov, pos_logdet = _fit_gaussian(pos_features, regularization)
    neg_mean, neg_inv_cov, neg_logdet = _fit_gaussian(neg_features, regularization)

    pos_scores = _gaussian_log_likelihood(pos_features, pos_mean, pos_inv_cov, pos_logdet) - _gaussian_log_likelihood(
        pos_features,
        neg_mean,
        neg_inv_cov,
        neg_logdet,
    )
    neg_scores = _gaussian_log_likelihood(neg_features, pos_mean, pos_inv_cov, pos_logdet) - _gaussian_log_likelihood(
        neg_features,
        neg_mean,
        neg_inv_cov,
        neg_logdet,
    )
    threshold = _select_threshold(pos_scores, neg_scores)

    s_low = max(15, int(np.percentile(pos_pixels[:, 1], 3.0)) - 12)
    v_low = max(10, int(np.percentile(pos_pixels[:, 2], 3.0)) - 15)

    return {
        "enabled": True,
        "model_type": "gaussian_hsv_binary",
        "feature_mode": "cos_sin_h_sv",
        "threshold": float(threshold),
        "s_low": int(s_low),
        "v_low": int(v_low),
        "regularization": float(regularization),
        "positive": {
            "mean": pos_mean.astype(np.float64).tolist(),
            "inv_cov": pos_inv_cov.astype(np.float64).tolist(),
            "logdet": float(pos_logdet),
            "count": int(pos_features.shape[0]),
        },
        "negative": {
            "mean": neg_mean.astype(np.float64).tolist(),
            "inv_cov": neg_inv_cov.astype(np.float64).tolist(),
            "logdet": float(neg_logdet),
            "count": int(neg_features.shape[0]),
        },
    }


def build_dark_red_ml_mask(hsv: np.ndarray, model_payload: Dict[str, object]) -> np.ndarray:
    if not bool(model_payload.get("enabled", False)):
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    positive = model_payload.get("positive")
    negative = model_payload.get("negative")
    if not isinstance(positive, dict) or not isinstance(negative, dict):
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    try:
        pos_mean = np.array(positive["mean"], dtype=np.float64)
        pos_inv_cov = np.array(positive["inv_cov"], dtype=np.float64)
        pos_logdet = float(positive["logdet"])

        neg_mean = np.array(negative["mean"], dtype=np.float64)
        neg_inv_cov = np.array(negative["inv_cov"], dtype=np.float64)
        neg_logdet = float(negative["logdet"])
        threshold = float(model_payload.get("threshold", 0.0))
        s_low = int(model_payload.get("s_low", 15))
        v_low = int(model_payload.get("v_low", 10))
    except (KeyError, TypeError, ValueError):
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    flat_hsv = hsv.reshape(-1, 3).astype(np.float64)
    gate = (flat_hsv[:, 1] >= s_low) & (flat_hsv[:, 2] >= v_low)
    if not np.any(gate):
        return np.zeros(hsv.shape[:2], dtype=np.uint8)

    features = _hsv_to_features(flat_hsv)
    scores = _gaussian_log_likelihood(features, pos_mean, pos_inv_cov, pos_logdet) - _gaussian_log_likelihood(
        features,
        neg_mean,
        neg_inv_cov,
        neg_logdet,
    )

    pred = (scores >= threshold) & gate
    mask = pred.astype(np.uint8).reshape(hsv.shape[:2]) * 255
    return mask