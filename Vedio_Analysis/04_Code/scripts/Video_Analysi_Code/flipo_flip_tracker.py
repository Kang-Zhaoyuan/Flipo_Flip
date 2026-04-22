from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple

# Allow running this file directly: `python 04_Code/scripts/Video_Analysi_Code/flipo_flip_tracker.py`.
if __package__ is None or __package__ == "":
    _scripts_root = Path(__file__).resolve().parent.parent
    if str(_scripts_root) not in sys.path:
        sys.path.insert(0, str(_scripts_root))

from Video_Analysi_Code.calibration_io import load_config, normalize_path
from Video_Analysi_Code.calibration_ui import (
    prompt_flipo_color,
    prompt_folder_path,
    prompt_yes_no,
)
from Video_Analysi_Code.calibration_workflow import choose_calibration
from Video_Analysi_Code.color_learning import run_first_time_color_learning
from Video_Analysi_Code.color_profiles import (
    build_tracker_params_for_color,
    is_color_learned,
    load_color_profiles,
    save_color_profiles,
    upsert_color_profile,
)
from Video_Analysi_Code.csv_paths import raw_csv_output_path
from Video_Analysi_Code.path_registry import (
    color_profile_log_path,
    find_project_root,
    resolve_config_path,
)
from Video_Analysi_Code.tracker_models import TrackerParams
from Video_Analysi_Code.tracker_pipeline import run_tracker


DEFAULT_PARALLEL_WORKERS = 4
_WORKER_DEVNULL = None


def _silence_worker_streams() -> None:
    global _WORKER_DEVNULL
    if _WORKER_DEVNULL is not None:
        return

    try:
        _WORKER_DEVNULL = open(os.devnull, "w", encoding="utf-8")
        sys.stdout = _WORKER_DEVNULL
        sys.stderr = _WORKER_DEVNULL
    except OSError:
        # If devnull cannot be opened, keep default streams.
        _WORKER_DEVNULL = None


def _track_one_video(
    video_path: str,
    calibration: Dict[str, object],
    params: TrackerParams,
    project_root_str: str,
) -> Tuple[str, str, int, str]:
    _silence_worker_streams()

    project_root = Path(project_root_str)
    video_path_obj = Path(video_path)
    output_csv_path = raw_csv_output_path(project_root, video_path_obj)

    df = run_tracker(video_path, calibration, params, enable_preview=False)
    df.to_csv(output_csv_path, index=False)

    return video_path_obj.name, params.color_name, len(df), str(output_csv_path)


def main() -> None:
    base_params = TrackerParams()
    video_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".wmv",
        ".webm",
    }

    project_root = find_project_root(Path(__file__))
    config_path = resolve_config_path(project_root)
    profile_path = color_profile_log_path(project_root)
    profile_payload = load_color_profiles(profile_path, base_params)
    save_color_profiles(profile_path, profile_payload)

    folder_input = prompt_folder_path()
    folder_path = Path(normalize_path(folder_input))

    video_paths = [
        path
        for path in folder_path.iterdir()
        if path.is_file() and path.suffix.lower() in video_extensions
    ]

    if not video_paths:
        print("No video files found in the selected folder.")
        return

    config_data = load_config(config_path)
    jobs: List[Tuple[str, Dict[str, object], TrackerParams, str]] = []
    relearn_decision_by_color: Dict[str, bool] = {}

    for video_path_obj in video_paths:
        video_path = str(video_path_obj)
        color_name = prompt_flipo_color(video_path)

        should_learn_color = False
        if color_name != "pink":
            already_learned = is_color_learned(profile_payload, color_name)
            if already_learned:
                cached_decision = relearn_decision_by_color.get(color_name)
                if cached_decision is None:
                    cached_decision = prompt_yes_no(
                        (
                            f"Learned profile found for '{color_name}'. "
                            "Relearn with interactive feedback?"
                        ),
                        default=False,
                    )
                    relearn_decision_by_color[color_name] = cached_decision
                should_learn_color = cached_decision
            else:
                should_learn_color = True

        if should_learn_color:
            print(
                f"Running interactive feedback learning for color '{color_name}' "
                "before tracking."
            )
            try:
                learned_profile = run_first_time_color_learning(
                    video_path=video_path,
                    color_name=color_name,
                    base_params=base_params,
                    round_count=1,
                )
            except RuntimeError as exc:
                print(f"[SKIP] {video_path_obj.name}: {exc}")
                relearn_decision_by_color[color_name] = False
                continue

            upsert_color_profile(
                payload=profile_payload,
                color_name=color_name,
                profile_data=learned_profile,
                base_params=base_params,
            )
            save_color_profiles(profile_path, profile_payload)
            print(f"Saved learned parameters for color '{color_name}' to: {profile_path}")
            relearn_decision_by_color[color_name] = False

        params_for_video = build_tracker_params_for_color(
            base_params=base_params,
            payload=profile_payload,
            color_name=color_name,
        )
        calibration = choose_calibration(video_path, config_path, config_data)
        jobs.append((video_path, calibration, params_for_video, color_name))

    if not jobs:
        print("No videos scheduled for tracking.")
        return

    cpu_count = os.cpu_count() or 1
    worker_count = min(DEFAULT_PARALLEL_WORKERS, cpu_count, len(jobs))

    # On Windows, redirected stdout (for example piping to head) can break
    # child process stream flushing in multiprocessing spawn mode.
    if worker_count > 1 and hasattr(sys.stdout, "isatty") and not sys.stdout.isatty():
        print(
            "Detected redirected stdout; switching to single-worker mode "
            "to avoid multiprocessing pipe errors."
        )
        worker_count = 1

    if worker_count <= 1:
        for video_path, calibration, params_for_video, color_name in jobs:
            output_csv_path = raw_csv_output_path(project_root, Path(video_path))

            df = run_tracker(
                video_path,
                calibration,
                params_for_video,
                enable_preview=True,
            )
            df.to_csv(output_csv_path, index=False)

            print(f"[{color_name}] Tracking complete. Rows: {len(df)}")
            print(f"CSV saved to: {output_csv_path}")
        return

    print(
        f"Starting parallel tracking with {worker_count} workers "
        f"(CPU cores detected: {cpu_count})"
    )

    success_count = 0
    failed_count = 0

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_video = {
            executor.submit(
                _track_one_video,
                video_path,
                calibration,
                params_for_video,
                str(project_root),
            ): (video_path, color_name)
            for video_path, calibration, params_for_video, color_name in jobs
        }

        for future in as_completed(future_to_video):
            source_video, source_color = future_to_video[future]
            try:
                video_name, color_name, row_count, output_csv_path = future.result()
            except Exception as exc:
                failed_count += 1
                print(f"[FAILED] {Path(source_video).name} [{source_color}]: {exc}")
                continue

            success_count += 1
            print(f"[DONE] {video_name} [{color_name}] Rows: {row_count}")
            print(f"CSV saved to: {output_csv_path}")

    print(
        f"Batch complete. Success: {success_count}, Failed: {failed_count}."
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # Graceful interruption avoids long tracebacks during manual stop.
        try:
            print("\nInterrupted by user. Exiting tracker.")
        except OSError:
            pass

        # If stdout pipe is already closed (for example with head), avoid
        # shutdown flush errors by redirecting to devnull before exit.
        try:
            devnull = open(os.devnull, "w", encoding="utf-8")
            sys.stdout = devnull
            sys.stderr = devnull
        except OSError:
            pass

        raise SystemExit(130)
