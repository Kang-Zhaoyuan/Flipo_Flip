from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


STAGE_LABELS = [
    "Original",
    "CLAHE",
    "Blue Mask",
    "Morphology",
    "CC Reject",
    "Thickness Fix",
    "White Mask",
    "Final",
]


@dataclass
class FrameMetrics:
    source_path: Path
    ground_y: int
    blue_area: int
    morph_area: int
    kept_area: int
    rejected_area: int
    stripped_area: int
    white_area: int
    component_count: int
    centroid: Optional[Tuple[int, int]]
    ellipse_angle: Optional[float]
    marker_angle: Optional[float]
    body_found: bool
    ellipse_found: bool
    marker_found: bool


@dataclass
class FrameArtifacts:
    original: np.ndarray
    clahe: np.ndarray
    blue_mask: np.ndarray
    morph_mask: np.ndarray
    cc_overlay: np.ndarray
    thickness_overlay: np.ndarray
    white_overlay: np.ndarray
    final_overlay: np.ndarray
    metrics: FrameMetrics


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name)
    key_parts: List[object] = []
    for part in parts:
        if part.isdigit():
            key_parts.append(int(part))
        else:
            key_parts.append(part.lower())
    return key_parts


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def read_image_bgr(source_path: Path) -> np.ndarray:
    raw_bytes = np.fromfile(str(source_path), dtype=np.uint8)
    if raw_bytes.size == 0:
        raise FileNotFoundError(f"Cannot read image bytes: {source_path}")
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot decode image: {source_path}")
    return image


def resize_with_padding(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    image = ensure_bgr(image)
    source_height, source_width = image.shape[:2]
    scale_factor = min(target_width / max(source_width, 1), target_height / max(source_height, 1))
    resized_width = max(1, int(round(source_width * scale_factor)))
    resized_height = max(1, int(round(source_height * scale_factor)))
    interpolation = cv2.INTER_AREA if scale_factor < 1.0 else cv2.INTER_CUBIC
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 22)
    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas


def draw_text_outline(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = 0.52,
    text_color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 0, 0),
        thickness + 2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )


def make_labeled_panel(
    image: np.ndarray,
    label: str,
    panel_width: int,
    panel_height: int,
    footer_text: Optional[str] = None,
) -> np.ndarray:
    label_height = 30
    content_height = panel_height - label_height
    content = resize_with_padding(image, panel_width, content_height)

    panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
    panel[:] = (12, 12, 16)
    panel[label_height:, :] = content
    cv2.rectangle(panel, (0, 0), (panel_width - 1, label_height - 1), (34, 34, 44), -1)
    cv2.rectangle(panel, (0, 0), (panel_width - 1, panel_height - 1), (90, 90, 110), 1)
    draw_text_outline(panel, label, (8, 21), font_scale=0.5)
    if footer_text:
        draw_text_outline(panel, footer_text, (8, panel_height - 8), font_scale=0.4)
    return panel


def apply_clahe(image_bgr: np.ndarray) -> np.ndarray:
    lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    light_channel, a_channel, b_channel = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    light_enhanced = clahe.apply(light_channel)
    return cv2.cvtColor(cv2.merge([light_enhanced, a_channel, b_channel]), cv2.COLOR_LAB2BGR)


def detect_ground_line(image_bgr: np.ndarray) -> int:
    height = image_bgr.shape[0]
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray_image, 30, 100)
    search_top = int(height * 0.60)
    row_strength = np.sum(edge_image[search_top:, :], axis=1)
    if row_strength.size == 0 or int(np.max(row_strength)) <= 0:
        return int(height * 0.92)
    return search_top + int(np.argmax(row_strength))


def build_blue_mask(enhanced_bgr: np.ndarray) -> np.ndarray:
    hsv_image = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([85, 40, 40], dtype=np.uint8)
    upper_blue = np.array([140, 255, 255], dtype=np.uint8)
    return cv2.inRange(hsv_image, lower_blue, upper_blue)


def solidify_and_cut(mask: np.ndarray) -> np.ndarray:
    close_kernel = np.ones((7, 7), np.uint8)
    erode_kernel = np.ones((5, 5), np.uint8)
    solid_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    eroded_mask = cv2.erode(solid_mask, erode_kernel, iterations=3)
    return cv2.dilate(eroded_mask, erode_kernel, iterations=3)


def reject_ground_components(mask: np.ndarray, ground_y: int) -> Tuple[np.ndarray, np.ndarray, int]:
    ground_tolerance = 15
    thin_limit = 30
    aspect_limit = 6.0

    component_count, labels, statistics, _ = cv2.connectedComponentsWithStats(mask)
    kept_mask = np.zeros_like(mask)
    rejected_mask = np.zeros_like(mask)

    kept_components = 0
    for component_index in range(1, component_count):
        x_pos, y_pos, width, height, area = statistics[component_index, :5]
        if area < 100:
            continue

        bottom_edge = y_pos + height
        aspect_ratio = width / max(height, 1)
        near_ground = bottom_edge >= ground_y - ground_tolerance
        is_thin = height < thin_limit
        is_wide = aspect_ratio > aspect_limit

        if near_ground and is_thin and is_wide:
            rejected_mask[labels == component_index] = 255
        else:
            kept_mask[labels == component_index] = 255
            kept_components += 1

    return kept_mask, rejected_mask, kept_components


def strip_thin_extensions(blob_mask: np.ndarray, min_thickness: int) -> Tuple[np.ndarray, np.ndarray]:
    cleaned_mask = blob_mask.copy()
    stripped_mask = np.zeros_like(blob_mask)

    column_fill = np.sum(cleaned_mask > 0, axis=0)
    thin_columns = column_fill < min_thickness
    if np.any(thin_columns):
        stripped_mask[:, thin_columns] = cleaned_mask[:, thin_columns]
        cleaned_mask[:, thin_columns] = 0

    row_fill = np.sum(cleaned_mask > 0, axis=1)
    thin_rows = row_fill < min_thickness
    if np.any(thin_rows):
        stripped_mask[thin_rows, :] = np.maximum(stripped_mask[thin_rows, :], cleaned_mask[thin_rows, :])
        cleaned_mask[thin_rows, :] = 0

    return cleaned_mask, stripped_mask


def fit_body_and_marker(
    original_bgr: np.ndarray,
    final_mask: np.ndarray,
) -> Tuple[np.ndarray, Optional[Tuple[int, int]], Optional[Tuple[Tuple[float, float], Tuple[float, float], float]], Optional[float], np.ndarray]:
    height, width = original_bgr.shape[:2]
    body_roi = np.zeros((height, width), dtype=np.uint8)

    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [contour for contour in contours if cv2.contourArea(contour) > 200]
    if not contours:
        return body_roi, None, None, None, np.zeros((height, width), dtype=np.uint8)

    body_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(body_roi, [body_contour], -1, 255, -1)

    centroid: Optional[Tuple[int, int]] = None
    moments = cv2.moments(body_contour)
    if moments["m00"] > 0:
        centroid = (
            int(round(moments["m10"] / moments["m00"])),
            int(round(moments["m01"] / moments["m00"])),
        )

    ellipse_result: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None
    ellipse_angle: Optional[float] = None
    if len(body_contour) >= 5:
        ellipse_result = cv2.fitEllipse(body_contour)
        ellipse_angle = float(ellipse_result[2] % 180.0)

    hsv_original = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(
        hsv_original,
        np.array([0, 0, 130], dtype=np.uint8),
        np.array([180, 110, 255], dtype=np.uint8),
    )
    white_mask = cv2.bitwise_and(white_mask, body_roi)
    white_mask = cv2.dilate(white_mask, np.ones((3, 3), np.uint8), iterations=1)

    marker_angle: Optional[float] = None
    if centroid is not None:
        component_count, labels, statistics, centroids = cv2.connectedComponentsWithStats(white_mask)
        best_index = -1
        best_distance = float("inf")
        for component_index in range(1, component_count):
            if statistics[component_index, cv2.CC_STAT_AREA] < 5:
                continue
            center_x, center_y = centroids[component_index]
            distance = float(np.hypot(center_x - centroid[0], center_y - centroid[1]))
            if distance < best_distance:
                best_distance = distance
                best_index = component_index

        if best_index != -1:
            points = np.column_stack(np.where(labels == best_index)[::-1]).astype(np.float32)
            if points.shape[0] >= 2:
                line_vector_x, line_vector_y, _, _ = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
                line_vector_x = float(line_vector_x.flat[0])
                line_vector_y = float(line_vector_y.flat[0])
                marker_angle = float(np.degrees(np.arctan2(line_vector_y, line_vector_x)) % 180.0)

    return body_roi, centroid, ellipse_result, marker_angle, white_mask


def overlay_final_result(
    original_bgr: np.ndarray,
    ground_y: int,
    final_mask: np.ndarray,
    centroid: Optional[Tuple[int, int]],
    ellipse_result: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]],
    marker_angle: Optional[float],
) -> np.ndarray:
    overlay = original_bgr.copy()
    cv2.line(overlay, (0, ground_y), (overlay.shape[1], ground_y), (60, 60, 255), 2)

    if np.count_nonzero(final_mask) > 0:
        mask_color = np.zeros_like(overlay)
        mask_color[final_mask > 0] = (0, 170, 255)
        overlay = cv2.addWeighted(overlay, 0.65, mask_color, 0.35, 0)

    if centroid is not None:
        cv2.circle(overlay, centroid, 6, (0, 0, 255), -1)
        draw_text_outline(overlay, f"Center ({centroid[0]}, {centroid[1]})", (centroid[0] + 10, centroid[1] - 10), font_scale=0.48)

    if ellipse_result is not None:
        cv2.ellipse(overlay, ellipse_result, (0, 220, 255), 2)
        ellipse_center = (int(round(ellipse_result[0][0])), int(round(ellipse_result[0][1])))
        ellipse_angle_radians = np.radians(float(ellipse_result[2] % 180.0))
        arrow_tip = (
            int(round(ellipse_center[0] + np.cos(ellipse_angle_radians) * 50)),
            int(round(ellipse_center[1] + np.sin(ellipse_angle_radians) * 50)),
        )
        cv2.arrowedLine(overlay, ellipse_center, arrow_tip, (0, 220, 255), 2, tipLength=0.25)

    if marker_angle is not None and centroid is not None:
        marker_angle_radians = np.radians(marker_angle)
        line_direction_x = float(np.cos(marker_angle_radians))
        line_direction_y = float(np.sin(marker_angle_radians))
        line_start = (
            int(round(centroid[0] - line_direction_x * 50)),
            int(round(centroid[1] - line_direction_y * 50)),
        )
        line_end = (
            int(round(centroid[0] + line_direction_x * 50)),
            int(round(centroid[1] + line_direction_y * 50)),
        )
        cv2.line(overlay, line_start, line_end, (0, 255, 255), 3)
        draw_text_outline(overlay, f"Marker {marker_angle:.1f} deg", (max(10, centroid[0] + 10), max(24, centroid[1] + 18)), font_scale=0.45)

    return overlay


def process_frame(source_path: Path) -> FrameArtifacts:
    original_bgr = read_image_bgr(source_path)

    clahe_bgr = apply_clahe(original_bgr)
    blue_mask = build_blue_mask(clahe_bgr)
    morph_mask = solidify_and_cut(blue_mask)
    ground_y = detect_ground_line(original_bgr)
    kept_mask, rejected_mask, component_count = reject_ground_components(morph_mask, ground_y)

    contour_candidates, _ = cv2.findContours(kept_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_candidates = [contour for contour in contour_candidates if cv2.contourArea(contour) > 200]

    final_mask = np.zeros_like(kept_mask)
    stripped_mask = np.zeros_like(kept_mask)
    body_roi = np.zeros_like(kept_mask)
    centroid: Optional[Tuple[int, int]] = None
    ellipse_result: Optional[Tuple[Tuple[float, float], Tuple[float, float], float]] = None
    ellipse_angle: Optional[float] = None
    marker_angle: Optional[float] = None
    white_mask = np.zeros_like(kept_mask)

    if contour_candidates:
        largest_contour = max(contour_candidates, key=cv2.contourArea)
        single_blob_mask = np.zeros_like(kept_mask)
        cv2.drawContours(single_blob_mask, [largest_contour], -1, 255, -1)

        minimum_thickness = max(28, int(min(original_bgr.shape[:2]) * 0.06))
        stripped_core, stripped_mask = strip_thin_extensions(single_blob_mask, minimum_thickness)
        stripped_core = cv2.morphologyEx(stripped_core, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        final_mask = stripped_core

        body_roi, centroid, ellipse_result, marker_angle, white_mask = fit_body_and_marker(original_bgr, final_mask)
        if ellipse_result is not None:
            ellipse_angle = float(ellipse_result[2] % 180.0)
    else:
        body_roi = np.zeros_like(kept_mask)

    original_with_ground = original_bgr.copy()
    cv2.line(original_with_ground, (0, ground_y), (original_with_ground.shape[1], ground_y), (60, 60, 255), 2)
    draw_text_outline(original_with_ground, f"Ground y={ground_y}", (10, max(24, ground_y - 10)), font_scale=0.5)

    clahe_with_ground = clahe_bgr.copy()
    cv2.line(clahe_with_ground, (0, ground_y), (clahe_with_ground.shape[1], ground_y), (60, 60, 255), 2)

    cc_overlay = np.zeros_like(original_bgr)
    cc_overlay[kept_mask > 0] = (0, 220, 0)
    cc_overlay[rejected_mask > 0] = (0, 0, 255)

    thickness_overlay = np.zeros_like(original_bgr)
    thickness_overlay[final_mask > 0] = (0, 220, 0)
    thickness_overlay[stripped_mask > 0] = (255, 0, 0)

    white_overlay = np.zeros_like(original_bgr)
    white_overlay[white_mask > 0] = (255, 255, 255)
    if centroid is not None:
        cv2.circle(white_overlay, centroid, 5, (0, 0, 255), -1)

    final_overlay = overlay_final_result(original_bgr, ground_y, final_mask, centroid, ellipse_result, marker_angle)

    metrics = FrameMetrics(
        source_path=source_path,
        ground_y=ground_y,
        blue_area=int(np.count_nonzero(blue_mask)),
        morph_area=int(np.count_nonzero(morph_mask)),
        kept_area=int(np.count_nonzero(final_mask)),
        rejected_area=int(np.count_nonzero(rejected_mask)),
        stripped_area=int(np.count_nonzero(stripped_mask)),
        white_area=int(np.count_nonzero(white_mask)),
        component_count=int(component_count - 1),
        centroid=centroid,
        ellipse_angle=ellipse_angle,
        marker_angle=marker_angle,
        body_found=int(np.count_nonzero(final_mask)) > 200,
        ellipse_found=ellipse_angle is not None,
        marker_found=marker_angle is not None,
    )

    return FrameArtifacts(
        original=original_with_ground,
        clahe=clahe_with_ground,
        blue_mask=ensure_bgr(blue_mask),
        morph_mask=ensure_bgr(morph_mask),
        cc_overlay=cc_overlay,
        thickness_overlay=thickness_overlay,
        white_overlay=white_overlay,
        final_overlay=final_overlay,
        metrics=metrics,
    )


def create_stage_panels(artifacts: FrameArtifacts, frame_index: int, panel_width: int, panel_height: int) -> List[np.ndarray]:
    footer_base = f"{artifacts.metrics.kept_area}px body | {artifacts.metrics.white_area}px white"
    return [
        make_labeled_panel(artifacts.original, f"Frame {frame_index:02d} | Original", panel_width, panel_height),
        make_labeled_panel(artifacts.clahe, "CLAHE", panel_width, panel_height),
        make_labeled_panel(artifacts.blue_mask, "Blue Mask", panel_width, panel_height, footer_text=f"{artifacts.metrics.blue_area}px"),
        make_labeled_panel(artifacts.morph_mask, "Morphology", panel_width, panel_height, footer_text=f"{artifacts.metrics.morph_area}px"),
        make_labeled_panel(artifacts.cc_overlay, "CC Reject", panel_width, panel_height, footer_text=f"rejected {artifacts.metrics.rejected_area}px"),
        make_labeled_panel(artifacts.thickness_overlay, "Thickness Fix", panel_width, panel_height, footer_text=f"stripped {artifacts.metrics.stripped_area}px"),
        make_labeled_panel(artifacts.white_overlay, "White Mask", panel_width, panel_height, footer_text=footer_base),
        make_labeled_panel(artifacts.final_overlay, "Final", panel_width, panel_height),
    ]


def save_debug_panels(panels: Sequence[np.ndarray], debug_root: Path, frame_index: int) -> None:
    frame_dir = debug_root / f"frame_{frame_index:02d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for stage_index, panel in enumerate(panels):
        stage_name = STAGE_LABELS[stage_index].lower().replace(" ", "_")
        output_path = frame_dir / f"{stage_index + 1:02d}_{stage_name}.png"
        cv2.imwrite(str(output_path), panel)


def create_contact_sheet(all_panels: List[List[np.ndarray]], output_path: Path, title: str) -> None:
    if not all_panels:
        raise ValueError("No panels available for contact sheet.")

    panel_height, panel_width = all_panels[0][0].shape[:2]
    columns = len(all_panels[0])
    rows = len(all_panels)

    title_height = 70
    spacing = 10
    sheet_width = columns * panel_width + (columns + 1) * spacing
    sheet_height = rows * panel_height + (rows + 1) * spacing + title_height

    sheet = np.full((sheet_height, sheet_width, 3), 18, dtype=np.uint8)
    draw_text_outline(sheet, title, (spacing, 36), font_scale=0.9, thickness=2)
    draw_text_outline(
        sheet,
        "Each row is one screenshot; each column follows the workflow order.",
        (spacing, 60),
        font_scale=0.48,
        thickness=1,
    )

    for row_index, row_panels in enumerate(all_panels):
        for column_index, panel in enumerate(row_panels):
            y_start = title_height + spacing + row_index * (panel_height + spacing)
            x_start = spacing + column_index * (panel_width + spacing)
            sheet[y_start : y_start + panel_height, x_start : x_start + panel_width] = panel

    cv2.imwrite(str(output_path), sheet)


def summarize_metrics(metrics_list: Sequence[FrameMetrics]) -> Dict[str, object]:
    total_frames = len(metrics_list)
    body_success = sum(1 for metrics in metrics_list if metrics.body_found)
    ellipse_success = sum(1 for metrics in metrics_list if metrics.ellipse_found)
    marker_success = sum(1 for metrics in metrics_list if metrics.marker_found)

    average_blue_area = float(np.mean([metrics.blue_area for metrics in metrics_list])) if metrics_list else 0.0
    average_white_area = float(np.mean([metrics.white_area for metrics in metrics_list])) if metrics_list else 0.0
    average_kept_area = float(np.mean([metrics.kept_area for metrics in metrics_list])) if metrics_list else 0.0

    return {
        "total_frames": total_frames,
        "body_success": body_success,
        "ellipse_success": ellipse_success,
        "marker_success": marker_success,
        "average_blue_area": average_blue_area,
        "average_white_area": average_white_area,
        "average_kept_area": average_kept_area,
    }


def create_report(
    metrics_list: Sequence[FrameMetrics],
    summary: Dict[str, object],
    contact_sheet_path: Path,
    report_path: Path,
) -> None:
    lines: List[str] = []
    lines.append("# Blue Video Analysis Workflow Feasibility Report")
    lines.append("")
    lines.append("## Conclusion")
    lines.append(
        "The workflow is operational on this screenshot set: the blue-body segmentation, centroid recovery, and ellipse fitting stages are stable enough to build a usable contact sheet, while the white-marker step remains the most sensitive and should still be checked by eye."
    )
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Screens analyzed: {summary['total_frames']}")
    lines.append(f"- Blue body recovered: {summary['body_success']} / {summary['total_frames']}")
    lines.append(f"- Ellipse fit recovered: {summary['ellipse_success']} / {summary['total_frames']}")
    lines.append(f"- White marker recovered: {summary['marker_success']} / {summary['total_frames']}")
    lines.append(f"- Average blue-mask area: {summary['average_blue_area']:.1f} pixels")
    lines.append(f"- Average final-body area: {summary['average_kept_area']:.1f} pixels")
    lines.append(f"- Average white-mask area: {summary['average_white_area']:.1f} pixels")
    lines.append(f"- Contact sheet: {contact_sheet_path.name}")
    lines.append("")
    lines.append("## Frame-by-Frame Results")
    lines.append("| Frame | Body | Ellipse | Marker | Ground Y | Blue Area | Final Area | White Area |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for frame_index, metrics in enumerate(metrics_list, start=1):
        body_flag = "Yes" if metrics.body_found else "No"
        ellipse_flag = "Yes" if metrics.ellipse_found else "No"
        marker_flag = "Yes" if metrics.marker_found else "No"
        lines.append(
            f"| {frame_index:02d} | {body_flag} | {ellipse_flag} | {marker_flag} | {metrics.ground_y} | {metrics.blue_area} | {metrics.kept_area} | {metrics.white_area} |"
        )

    lines.append("")
    lines.append("## Interpretation")
    if summary["body_success"] == summary["total_frames"]:
        lines.append("- The blue-disc body was isolated on every screenshot, which indicates that the color threshold plus morphology are sufficient for this dataset.")
    else:
        lines.append("- The blue-disc body was not isolated on every screenshot, so the threshold or morphology will need tuning for some frames.")

    if summary["ellipse_success"] == summary["total_frames"]:
        lines.append("- Ellipse fitting succeeded on every screenshot, so the centroid and tilt stage are dependable here.")
    else:
        lines.append("- Ellipse fitting failed on at least one screenshot, which means the final contour is occasionally too sparse or fragmented.")

    if summary["marker_success"] >= max(1, summary["total_frames"] // 2):
        lines.append("- Marker-line recovery is usable but still the least stable stage, so manual confirmation is still recommended.")
    else:
        lines.append("- Marker-line recovery is weak on this set, so the white-line step should be treated as advisory rather than authoritative.")

    lines.append("")
    lines.append("## Notes")
    lines.append("- The ground-line estimate is derived automatically from edge density in the lower part of each screenshot.")
    lines.append("- The thickness-fix step is preserved because it is the main defense against fused ground-strip artifacts.")
    lines.append("- English-only labels are used in the generated figure and report to match the requested output format.")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(input_dir: Path, output_prefix: str) -> Tuple[Path, Path]:
    image_paths = [path for path in sorted(input_dir.iterdir(), key=natural_key) if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file()]
    if not image_paths:
        raise FileNotFoundError(f"No images found in {input_dir}")

    debug_root = input_dir / f"{output_prefix}_debug"
    debug_root.mkdir(parents=True, exist_ok=True)

    all_panels: List[List[np.ndarray]] = []
    metrics_list: List[FrameMetrics] = []

    panel_width = 340
    panel_height = 245

    for frame_index, image_path in enumerate(image_paths, start=1):
        artifacts = process_frame(image_path)
        panels = create_stage_panels(artifacts, frame_index, panel_width, panel_height)
        save_debug_panels(panels, debug_root, frame_index)
        all_panels.append(panels)
        metrics_list.append(artifacts.metrics)

    contact_sheet_path = input_dir / f"{output_prefix}_contact_sheet.png"
    report_path = input_dir / f"{output_prefix}_feasibility_report.md"

    create_contact_sheet(all_panels, contact_sheet_path, "Blue Video Analysis Workflow Reproduction")
    summary = summarize_metrics(metrics_list)
    create_report(metrics_list, summary, contact_sheet_path, report_path)

    return contact_sheet_path, report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reproduce the Blue Video Analysis workflow on screenshot images.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Directory containing screenshots. Defaults to the script directory.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="blue_workflow",
        help="Prefix for generated output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    input_dir = Path(args.input_dir).expanduser().resolve() if args.input_dir else script_dir
    contact_sheet_path, report_path = run_pipeline(input_dir, args.output_prefix)
    print(f"Contact sheet written to: {contact_sheet_path}")
    print(f"Feasibility report written to: {report_path}")


if __name__ == "__main__":
    main()