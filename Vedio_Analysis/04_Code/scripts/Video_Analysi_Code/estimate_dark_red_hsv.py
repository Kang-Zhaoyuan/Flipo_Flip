from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np


HSVTriplet = Tuple[int, int, int]
HSVRange = Tuple[HSVTriplet, HSVTriplet]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_image_paths(inputs: Sequence[str]) -> List[Path]:
    resolved: List[Path] = []
    seen = set()

    for raw in inputs:
        candidate = Path(raw).expanduser().resolve()
        if not candidate.exists():
            print(f"[WARN] Path not found and skipped: {candidate}")
            continue

        if candidate.is_dir():
            files = sorted(path for path in candidate.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
            for file_path in files:
                key = str(file_path)
                if key not in seen:
                    seen.add(key)
                    resolved.append(file_path)
            continue

        if candidate.suffix.lower() not in IMAGE_SUFFIXES:
            print(f"[WARN] Non-image file skipped: {candidate}")
            continue

        key = str(candidate)
        if key not in seen:
            seen.add(key)
            resolved.append(candidate)

    return resolved


def _largest_component(mask_u8: np.ndarray, min_area: int) -> np.ndarray:
    component_mask = np.zeros(mask_u8.shape, dtype=bool)
    if mask_u8.ndim != 2:
        return component_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return component_mask

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_offset = int(np.argmax(areas))
    largest_area = int(areas[largest_offset])
    if largest_area < min_area:
        return component_mask

    largest_label = 1 + largest_offset
    component_mask = labels == largest_label
    return component_mask


def _extract_dark_red_pixels(
    image_bgr: np.ndarray,
    min_s: int,
    min_v: int,
    hue_low_max: int,
    hue_high_min: int,
    min_component_area: int,
) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    # Seed mask for red hues with basic saturation/value filtering.
    seed = (
        (s_channel >= min_s)
        & (v_channel >= min_v)
        & ((h_channel <= hue_low_max) | (h_channel >= hue_high_min))
    )

    seed_u8 = (seed.astype(np.uint8)) * 255
    kernel = np.ones((5, 5), dtype=np.uint8)
    seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    seed_u8 = cv2.morphologyEx(seed_u8, cv2.MORPH_CLOSE, kernel, iterations=2)

    component_mask = _largest_component(seed_u8, min_component_area)
    if not np.any(component_mask):
        return np.empty((0, 3), dtype=np.int32)

    pixels = hsv[component_mask]
    if pixels.size == 0:
        return np.empty((0, 3), dtype=np.int32)

    return pixels.astype(np.int32)


def _percentile_bounds(
    values: np.ndarray,
    low_q: float,
    high_q: float,
    pad: int,
    clamp_low: int,
    clamp_high: int,
) -> Tuple[int, int]:
    low = int(np.percentile(values, low_q)) - pad
    high = int(np.percentile(values, high_q)) + pad

    low = max(clamp_low, low)
    high = min(clamp_high, high)
    if low > high:
        low, high = clamp_low, clamp_high

    return low, high


def _estimate_dark_red_ranges(
    hsv_pixels: np.ndarray,
    low_q: float,
    high_q: float,
    hue_pad: int,
    sv_pad: int,
    min_s_floor: int,
    min_v_floor: int,
    hue_low_max: int,
    hue_high_min: int,
    min_hue_group_size: int,
) -> Tuple[List[HSVRange], dict]:
    if hsv_pixels.size == 0:
        return [], {"pixel_count": 0}

    h = hsv_pixels[:, 0]
    s = hsv_pixels[:, 1]
    v = hsv_pixels[:, 2]

    s_low, s_high = _percentile_bounds(s, low_q, high_q, sv_pad, 0, 255)
    v_low, v_high = _percentile_bounds(v, low_q, high_q, sv_pad, 0, 255)
    s_low = max(min_s_floor, s_low)
    v_low = max(min_v_floor, v_low)

    ranges: List[HSVRange] = []

    low_h_group = h[h <= hue_low_max]
    if low_h_group.size >= min_hue_group_size:
        h_low, h_high = _percentile_bounds(low_h_group, low_q, high_q, hue_pad, 0, hue_low_max)
        ranges.append(((h_low, s_low, v_low), (h_high, s_high, v_high)))

    high_h_group = h[h >= hue_high_min]
    if high_h_group.size >= min_hue_group_size:
        h_low, h_high = _percentile_bounds(high_h_group, low_q, high_q, hue_pad, hue_high_min, 180)
        ranges.append(((h_low, s_low, v_low), (h_high, s_high, v_high)))

    if not ranges:
        # Fallback to a single contiguous hue range when split groups are too small.
        h_low, h_high = _percentile_bounds(h, low_q, high_q, hue_pad, 0, 180)
        ranges.append(((h_low, s_low, v_low), (h_high, s_high, v_high)))

    ranges.sort(key=lambda item: item[0][0])
    summary = {
        "pixel_count": int(hsv_pixels.shape[0]),
        "h_percentile_low": int(np.percentile(h, low_q)),
        "h_percentile_high": int(np.percentile(h, high_q)),
        "s_percentile_low": int(np.percentile(s, low_q)),
        "s_percentile_high": int(np.percentile(s, high_q)),
        "v_percentile_low": int(np.percentile(v, low_q)),
        "v_percentile_high": int(np.percentile(v, high_q)),
        "low_h_group_size": int(low_h_group.size),
        "high_h_group_size": int(high_h_group.size),
    }
    return ranges, summary


def _serialize_ranges(ranges: Sequence[HSVRange]) -> List[List[List[int]]]:
    payload: List[List[List[int]]] = []
    for lower, upper in ranges:
        payload.append([list(lower), list(upper)])
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate dark-red HSV ranges from one or more images. "
            "Input can be image files or folders."
        )
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="Image file paths or directories containing dark-red object images.",
    )
    parser.add_argument("--low-q", type=float, default=5.0, help="Low percentile for robust bounds.")
    parser.add_argument("--high-q", type=float, default=95.0, help="High percentile for robust bounds.")
    parser.add_argument("--hue-pad", type=int, default=3, help="Hue padding added around percentile bounds.")
    parser.add_argument("--sv-pad", type=int, default=10, help="S/V padding added around percentile bounds.")
    parser.add_argument("--min-s", type=int, default=60, help="Minimum saturation floor for final range.")
    parser.add_argument("--min-v", type=int, default=40, help="Minimum value floor for final range.")
    parser.add_argument(
        "--hue-low-max",
        type=int,
        default=25,
        help="Upper hue limit for low-red group (0 side).",
    )
    parser.add_argument(
        "--hue-high-min",
        type=int,
        default=155,
        help="Lower hue limit for high-red group (180 side).",
    )
    parser.add_argument(
        "--min-component-area",
        type=int,
        default=400,
        help="Minimum area (pixels) for the red component extraction.",
    )
    parser.add_argument(
        "--min-hue-group-size",
        type=int,
        default=100,
        help="Minimum pixels needed in a hue group before creating its HSV range.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default="",
        help="Optional output path to save the estimated JSON payload.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.low_q < 0 or args.high_q > 100 or args.low_q >= args.high_q:
        print("[ERROR] Percentiles must satisfy 0 <= low-q < high-q <= 100.")
        return 2

    image_paths = _resolve_image_paths(args.images)
    if not image_paths:
        print("[ERROR] No usable image files found.")
        return 2

    all_pixels: List[np.ndarray] = []
    used_images: List[str] = []

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Failed to read image and skipped: {path}")
            continue

        pixels = _extract_dark_red_pixels(
            image_bgr=image,
            min_s=args.min_s,
            min_v=args.min_v,
            hue_low_max=args.hue_low_max,
            hue_high_min=args.hue_high_min,
            min_component_area=args.min_component_area,
        )
        if pixels.shape[0] == 0:
            print(f"[WARN] No dark-red component detected in: {path}")
            continue

        all_pixels.append(pixels)
        used_images.append(str(path))
        print(f"[INFO] {path.name}: extracted {pixels.shape[0]} dark-red pixels")

    if not all_pixels:
        print("[ERROR] Unable to extract dark-red pixels from all inputs.")
        return 2

    merged = np.vstack(all_pixels)
    ranges, summary = _estimate_dark_red_ranges(
        hsv_pixels=merged,
        low_q=float(args.low_q),
        high_q=float(args.high_q),
        hue_pad=int(args.hue_pad),
        sv_pad=int(args.sv_pad),
        min_s_floor=int(args.min_s),
        min_v_floor=int(args.min_v),
        hue_low_max=int(args.hue_low_max),
        hue_high_min=int(args.hue_high_min),
        min_hue_group_size=int(args.min_hue_group_size),
    )

    if not ranges:
        print("[ERROR] No HSV ranges generated.")
        return 2

    serialized_ranges = _serialize_ranges(ranges)
    payload = {
        "color": "dark_red",
        "object_ranges": serialized_ranges,
        "summary": summary,
        "images_used": used_images,
    }

    print("\nSuggested dark_red object_ranges:")
    for idx, (lower, upper) in enumerate(ranges, start=1):
        print(f"  range {idx}: LOWER={lower} UPPER={upper}")

    print("\nJSON payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=True))

    if args.save_json:
        out_path = Path(args.save_json).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nSaved JSON to: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())