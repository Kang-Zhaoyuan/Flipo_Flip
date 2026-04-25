from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from Video_Analysi_Code.tracker_models import TrackerParams

HSVTriplet = Tuple[int, int, int]
HSVRange = Tuple[HSVTriplet, HSVTriplet]
HSVRangeList = Tuple[HSVRange, ...]

SUPPORTED_COLORS = ("pink", "dark_red", "dark_green", "white", "blue")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def serialize_hsv_range(hsv_range: HSVRange) -> List[List[int]]:
    lower, upper = hsv_range
    return [list(lower), list(upper)]


def serialize_hsv_ranges(hsv_ranges: HSVRangeList) -> List[List[List[int]]]:
    return [serialize_hsv_range(item) for item in hsv_ranges]


def _coerce_triplet(raw: object) -> Optional[HSVTriplet]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        return None

    values: List[int] = []
    for item in raw:
        if not isinstance(item, int):
            return None
        values.append(int(item))

    h, s, v = values
    if not (0 <= h <= 180 and 0 <= s <= 255 and 0 <= v <= 255):
        return None

    return h, s, v


def deserialize_hsv_range(raw: object, fallback: HSVRange) -> HSVRange:
    parsed = _try_deserialize_hsv_range(raw)
    if parsed is None:
        return fallback
    return parsed


def _try_deserialize_hsv_range(raw: object) -> Optional[HSVRange]:
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        return None

    lower = _coerce_triplet(raw[0])
    upper = _coerce_triplet(raw[1])
    if lower is None or upper is None:
        return None

    for idx in range(3):
        if lower[idx] > upper[idx]:
            return None

    return (lower, upper)


def deserialize_hsv_ranges(raw: object, fallback: HSVRangeList) -> HSVRangeList:
    if not isinstance(raw, (list, tuple)):
        return fallback

    parsed: List[HSVRange] = []
    for item in raw:
        candidate = _try_deserialize_hsv_range(item)
        if candidate is None:
            continue
        parsed.append(candidate)

    if not parsed:
        return fallback

    return tuple(parsed)


def _default_profile(color_name: str, base_params: TrackerParams) -> Dict[str, object]:
    timestamp = _now_iso()
    if color_name == "pink":
        return {
            "color": "pink",
            "learned": True,
            "first_seen_at": timestamp,
            "updated_at": timestamp,
            "source_video": "bootstrap_default",
            "selected_sample_index": None,
            "object_ranges": serialize_hsv_ranges(base_params.object_ranges),
            "black_range": serialize_hsv_range(base_params.black_range),
            "red_ml_model": None,
            "samples": [],
            "bootstrap": True,
        }

    return {
        "color": color_name,
        "learned": False,
        "first_seen_at": None,
        "updated_at": timestamp,
        "source_video": None,
        "selected_sample_index": None,
        "object_ranges": [],
        "black_range": None,
        "red_ml_model": None,
        "samples": [],
        "bootstrap": False,
    }


def _default_document(base_params: TrackerParams) -> Dict[str, object]:
    now = _now_iso()
    profiles = {
        color: _default_profile(color, base_params) for color in SUPPORTED_COLORS
    }
    return {
        "schema_version": 1,
        "updated_at": now,
        "profiles": profiles,
    }


def _normalize_profile(
    raw_profile: object,
    color_name: str,
    base_params: TrackerParams,
) -> Dict[str, object]:
    fallback = _default_profile(color_name, base_params)
    if not isinstance(raw_profile, dict):
        return fallback

    learned = bool(raw_profile.get("learned", fallback["learned"]))
    first_seen_at = raw_profile.get("first_seen_at")
    updated_at = raw_profile.get("updated_at")
    source_video = raw_profile.get("source_video")
    selected_sample_index = raw_profile.get("selected_sample_index")
    bootstrap = bool(raw_profile.get("bootstrap", fallback["bootstrap"]))

    if not isinstance(first_seen_at, str):
        first_seen_at = fallback["first_seen_at"]
    if not isinstance(updated_at, str):
        updated_at = _now_iso()
    if not isinstance(source_video, str):
        source_video = fallback["source_video"]
    if not isinstance(selected_sample_index, int):
        selected_sample_index = None

    object_ranges_fallback: HSVRangeList = (
        base_params.object_ranges if color_name == "pink" else tuple()
    )
    object_ranges = deserialize_hsv_ranges(
        raw_profile.get("object_ranges"),
        object_ranges_fallback,
    )

    black_range = deserialize_hsv_range(
        raw_profile.get("black_range"),
        base_params.black_range,
    )

    red_ml_model = raw_profile.get("red_ml_model")
    if not isinstance(red_ml_model, dict):
        red_ml_model = None

    samples = raw_profile.get("samples")
    if not isinstance(samples, list):
        samples = []

    return {
        "color": color_name,
        "learned": learned,
        "first_seen_at": first_seen_at,
        "updated_at": updated_at,
        "source_video": source_video,
        "selected_sample_index": selected_sample_index,
        "object_ranges": serialize_hsv_ranges(object_ranges),
        "black_range": serialize_hsv_range(black_range),
        "red_ml_model": red_ml_model,
        "samples": samples,
        "bootstrap": bootstrap,
    }


def load_color_profiles(
    profile_path: Path,
    base_params: TrackerParams,
) -> Dict[str, object]:
    default_doc = _default_document(base_params)
    if not profile_path.exists():
        return default_doc

    try:
        payload = json.loads(profile_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_doc

    if not isinstance(payload, dict):
        return default_doc

    profiles_raw = payload.get("profiles")
    if not isinstance(profiles_raw, dict):
        profiles_raw = {}

    normalized_profiles = {
        color: _normalize_profile(profiles_raw.get(color), color, base_params)
        for color in SUPPORTED_COLORS
    }

    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str):
        updated_at = _now_iso()

    return {
        "schema_version": 1,
        "updated_at": updated_at,
        "profiles": normalized_profiles,
    }


def save_color_profiles(profile_path: Path, payload: Dict[str, object]) -> None:
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def is_color_learned(payload: Dict[str, object], color_name: str) -> bool:
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return False

    profile = profiles.get(color_name)
    if not isinstance(profile, dict):
        return False

    if not bool(profile.get("learned", False)):
        return False

    object_ranges = profile.get("object_ranges")
    return bool(isinstance(object_ranges, list) and len(object_ranges) > 0)


def get_color_profile(
    payload: Dict[str, object],
    color_name: str,
) -> Optional[Dict[str, object]]:
    profiles = payload.get("profiles")
    if not isinstance(profiles, dict):
        return None

    profile = profiles.get(color_name)
    if not isinstance(profile, dict):
        return None

    return profile


def upsert_color_profile(
    payload: Dict[str, object],
    color_name: str,
    profile_data: Dict[str, object],
    base_params: TrackerParams,
) -> None:
    profiles = payload.setdefault("profiles", {})
    if not isinstance(profiles, dict):
        profiles = {}
        payload["profiles"] = profiles

    merged_profile = _normalize_profile(profile_data, color_name, base_params)
    merged_profile["learned"] = bool(profile_data.get("learned", True))

    first_seen_at = profile_data.get("first_seen_at")
    if isinstance(first_seen_at, str):
        merged_profile["first_seen_at"] = first_seen_at
    elif not isinstance(merged_profile.get("first_seen_at"), str):
        merged_profile["first_seen_at"] = _now_iso()

    merged_profile["updated_at"] = _now_iso()
    profiles[color_name] = merged_profile
    payload["updated_at"] = _now_iso()


def build_tracker_params_for_color(
    base_params: TrackerParams,
    payload: Dict[str, object],
    color_name: str,
) -> TrackerParams:
    profile = get_color_profile(payload, color_name)
    if not isinstance(profile, dict):
        return replace(base_params, color_name=color_name)

    object_ranges = deserialize_hsv_ranges(
        profile.get("object_ranges"),
        base_params.object_ranges,
    )
    black_range = deserialize_hsv_range(
        profile.get("black_range"),
        base_params.black_range,
    )
    red_ml_model = profile.get("red_ml_model")
    if not isinstance(red_ml_model, dict):
        red_ml_model = None

    return replace(
        base_params,
        color_name=color_name,
        object_ranges=object_ranges,
        black_range=black_range,
        red_ml_model=red_ml_model,
    )
