from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from Video_Analysi_Code.tracker_models import WINDOW_CALIB


SUPPORTED_FLIPO_COLORS = ("pink", "dark_red", "dark_green", "white", "blue")


def prompt_video_path() -> str:
    while True:
        raw = input("Enter absolute video path: ").strip().strip('"')
        if not raw:
            print("Path cannot be empty.")
            continue

        if not Path(raw).is_absolute():
            print("Please provide an absolute path.")
            continue

        candidate = Path(raw)
        if not candidate.exists() or not candidate.is_file():
            print("File does not exist. Please try again.")
            continue

        return str(candidate)


def prompt_folder_path() -> str:
    while True:
        raw = input("Enter absolute folder path: ").strip().strip('"')
        if not raw:
            print("Path cannot be empty.")
            continue

        if not Path(raw).is_absolute():
            print("Please provide an absolute path.")
            continue

        candidate = Path(raw)
        if not candidate.exists() or not candidate.is_dir():
            print("Folder does not exist. Please try again.")
            continue

        return str(candidate)


def prompt_float(prompt_text: str, minimum: float = 0.0) -> float:
    while True:
        raw = input(prompt_text).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a numeric value.")
            continue

        if value <= minimum:
            print(f"Value must be greater than {minimum}.")
            continue

        return value


def prompt_yes_no(message: str, default: bool = True) -> bool:
    default_hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{message} [{default_hint}]: ").strip().lower()
        if raw == "":
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please type 'y' or 'n'.")


def prompt_flipo_color(video_path: str) -> str:
    print("\nSelect Flipo Flip color for this video:")
    print(f"  video: {Path(video_path).name}")
    print("  1) pink")
    print("  2) dark_red")
    print("  3) dark_green")
    print("  4) white")
    print("  5) blue")

    alias_map = {
        "1": "pink",
        "2": "dark_red",
        "3": "dark_green",
        "4": "white",
        "5": "blue",
        "pink": "pink",
        "dark_red": "dark_red",
        "red": "dark_red",
        "dark_green": "dark_green",
        "green": "dark_green",
        "white": "white",
        "blue": "blue",
    }

    while True:
        raw = input("Enter color choice [1-5]: ").strip().lower()
        if raw in alias_map:
            return alias_map[raw]
        print("Invalid color. Please choose one of: pink, dark_red, dark_green, white, blue.")


def prompt_frame_index(total_frames: int, current_index: int) -> Optional[int]:
    max_index = max(total_frames - 1, 0)
    prompt = (
        f"Enter calibration frame index [0-{max_index}] "
        f"(blank keeps {current_index}): "
    )
    raw = input(prompt).strip()
    if raw == "":
        return current_index

    try:
        value = int(raw)
    except ValueError:
        print("Please enter an integer frame index.")
        return None

    if value < 0 or value > max_index:
        print(f"Frame index must be within [0, {max_index}].")
        return None

    return value


def prompt_reuse_or_calibrate(message: str) -> str:
    while True:
        raw = input(f"{message} [r/c] (default r): ").strip().lower()
        if raw in {"", "r", "reuse"}:
            return "reuse"
        if raw in {"c", "calibrate", "recalibrate"}:
            return "calibrate"
        print("Please type 'r' to reuse or 'c' to calibrate.")


def get_first_frame(video_path: str) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    ok, frame = capture.read()
    capture.release()

    if not ok:
        raise RuntimeError("Cannot read first frame from video.")

    return frame


def select_calibration_frame(video_path: str) -> Optional[Tuple[np.ndarray, int]]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = 1

    slider_max = max(total_frames - 1, 0)
    selected_index = 0
    displayed_index = -1
    displayed_frame: Optional[np.ndarray] = None

    def on_trackbar(position: int) -> None:
        nonlocal selected_index
        selected_index = position

    cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Frame", WINDOW_CALIB, 0, slider_max, on_trackbar)

    while True:
        if displayed_frame is None or displayed_index != selected_index:
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(selected_index))
            ok, frame = capture.read()
            if not ok:
                if selected_index == 0:
                    cv2.destroyWindow(WINDOW_CALIB)
                    capture.release()
                    raise RuntimeError("Cannot decode frame 0 from video.")
                print(f"Cannot decode frame {selected_index}. Try another frame.")
                selected_index = 0
                cv2.setTrackbarPos("Frame", WINDOW_CALIB, selected_index)
                continue

            displayed_frame = frame
            actual_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            if actual_index < 0:
                actual_index = selected_index
            displayed_index = actual_index

            if displayed_index != selected_index:
                selected_index = displayed_index
                cv2.setTrackbarPos("Frame", WINDOW_CALIB, selected_index)

        canvas = displayed_frame.copy()
        cv2.putText(
            canvas,
            (
                f"Select calibration frame: {displayed_index}/{max(total_frames - 1, 0)}"
            ),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "ENTER: confirm  I: input frame index  ESC/Q: cancel",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_CALIB, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(WINDOW_CALIB)
            capture.release()
            return None

        if key in (13, 10):
            frame_out = displayed_frame.copy()
            frame_index_out = displayed_index if displayed_index >= 0 else selected_index
            cv2.destroyWindow(WINDOW_CALIB)
            capture.release()
            return frame_out, int(frame_index_out)

        if key == ord("i"):
            manual_index = prompt_frame_index(total_frames, selected_index)
            if manual_index is not None:
                selected_index = manual_index
                cv2.setTrackbarPos("Frame", WINDOW_CALIB, selected_index)


def select_two_points(frame: np.ndarray) -> Optional[List[Tuple[int, int]]]:
    points: List[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_CALIB, on_mouse)

    while True:
        canvas = frame.copy()

        for idx, (px, py) in enumerate(points):
            cv2.circle(canvas, (px, py), 6, (0, 255, 255), -1)
            cv2.putText(
                canvas,
                f"P{idx + 1}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        if len(points) == 2:
            cv2.line(canvas, points[0], points[1], (255, 255, 0), 2, cv2.LINE_AA)
            instruction = "ENTER: confirm  R: reset  ESC/Q: cancel"
        else:
            instruction = "Click two calibration points. R: reset  ESC/Q: cancel"

        cv2.putText(
            canvas,
            instruction,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_CALIB, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(WINDOW_CALIB)
            return None
        if key == ord("r"):
            points.clear()
        if key in (13, 10) and len(points) == 2:
            cv2.destroyWindow(WINDOW_CALIB)
            return points


def select_ground_line(frame: np.ndarray) -> Optional[int]:
    """Select ground line for blue object filtering. Returns Y coordinate of ground line."""
    points: List[Tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(WINDOW_CALIB, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_CALIB, on_mouse)

    while True:
        canvas = frame.copy()

        for idx, (px, py) in enumerate(points):
            cv2.circle(canvas, (px, py), 6, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f"Ground Point {idx + 1}",
                (px + 8, py - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        if len(points) == 2:
            # Draw horizontal line at the Y coordinate of the first point
            ground_y = points[0][1]
            cv2.line(canvas, (0, ground_y), (frame.shape[1], ground_y), (255, 0, 0), 2, cv2.LINE_AA)
            instruction = "ENTER: confirm ground line  R: reset  ESC/Q: cancel"
        else:
            instruction = "Click on ground to set horizontal ground line. R: reset  ESC/Q: cancel"

        cv2.putText(
            canvas,
            instruction,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow(WINDOW_CALIB, canvas)
        key = cv2.waitKey(20) & 0xFF

        if key in (27, ord("q")):
            cv2.destroyWindow(WINDOW_CALIB)
            return None
        if key == ord("r"):
            points.clear()
        if key in (13, 10) and len(points) >= 1:
            ground_y = points[0][1]
            cv2.destroyWindow(WINDOW_CALIB)
            return ground_y
