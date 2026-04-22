from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class CsvSnapshot:
    mtime_ns: int
    size: int


def _resolve_path(project_root: Path, value: str | None, default: Path) -> Path:
    if value is None:
        return default.resolve()

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = (project_root / candidate).resolve()
    return candidate.resolve()


def _snapshot_raw_csv(raw_dir: Path) -> Dict[str, CsvSnapshot]:
    snapshot: Dict[str, CsvSnapshot] = {}
    for path in sorted(raw_dir.glob("*_raw_data.csv")):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[path.name] = CsvSnapshot(mtime_ns=stat.st_mtime_ns, size=stat.st_size)
    return snapshot


def _detect_changed_csv(
    before: Dict[str, CsvSnapshot],
    after: Dict[str, CsvSnapshot],
    raw_dir: Path,
) -> List[Path]:
    changed: List[Path] = []
    for name, after_state in sorted(after.items()):
        before_state = before.get(name)
        if before_state is None:
            changed.append((raw_dir / name).resolve())
            continue
        if (
            before_state.mtime_ns != after_state.mtime_ns
            or before_state.size != after_state.size
        ):
            changed.append((raw_dir / name).resolve())

    return changed


def _run_subprocess(command: List[str], cwd: Path) -> int:
    print(f"$ {' '.join(command)}")
    completed = subprocess.run(command, cwd=str(cwd), check=False)
    return int(completed.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Flipo Flip full pipeline in one command: "
            "video analysis -> raw CSV -> clean/rank -> plotting."
        )
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Project root. Defaults to the directory containing main.py.",
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable used for subprocess scripts.",
    )
    parser.add_argument(
        "--tracker-script",
        type=str,
        default=None,
        help=(
            "Tracker script path. Defaults to "
            "04_Code/scripts/Video_Analysi_Code/flipo_flip_tracker.py"
        ),
    )
    parser.add_argument(
        "--cleaner-script",
        type=str,
        default=None,
        help=(
            "Cleaner script path. Defaults to "
            "04_Code/scripts/Video_Analysi_Code/clean_roll_intervals.py"
        ),
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Raw CSV directory. Defaults to 01_Data/Raw.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Cleaner output directory. Defaults to 01_Data/Processed.",
    )
    parser.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help="Plot output directory. Defaults to 03_Results/Plots.",
    )
    parser.add_argument(
        "--min-roll-deg",
        type=float,
        default=180.0,
        help="Forwarded to cleaner --min-roll-deg.",
    )
    parser.add_argument(
        "--auto-relax-roll-threshold",
        action="store_true",
        help="Forwarded to cleaner --auto-relax-roll-threshold.",
    )
    parser.add_argument(
        "--relax-roll-seq",
        type=str,
        default="150,120,90,60,45",
        help="Forwarded to cleaner --relax-roll-seq.",
    )
    parser.add_argument(
        "--red-ignore-theta-mode",
        type=str,
        choices=("auto", "on", "off"),
        default="auto",
        help="Forwarded to cleaner --red-ignore-theta-mode.",
    )
    parser.add_argument(
        "--red-start-y-max",
        type=float,
        default=838.0,
        help="Forwarded to cleaner --red-start-y-max.",
    )
    parser.add_argument(
        "--red-end-y-min",
        type=float,
        default=935.0,
        help="Forwarded to cleaner --red-end-y-min.",
    )
    parser.add_argument(
        "--red-wobble-smoothing-alpha",
        type=float,
        default=0.25,
        help="Forwarded to cleaner --red-wobble-smoothing-alpha.",
    )
    parser.add_argument(
        "--red-forward-dx-threshold",
        type=float,
        default=0.5,
        help="Forwarded to cleaner --red-forward-dx-threshold.",
    )
    parser.add_argument(
        "--red-cycle-reset-threshold-deg",
        type=float,
        default=35.0,
        help="Forwarded to cleaner --red-cycle-reset-threshold-deg.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if args.min_roll_deg <= 0.0:
        print(f"--min-roll-deg must be positive. Got: {args.min_roll_deg}")
        return 1

    default_project_root = Path(__file__).resolve().parent
    project_root = _resolve_path(default_project_root, args.project_root, default_project_root)

    tracker_default = (
        project_root
        / "04_Code"
        / "scripts"
        / "Video_Analysi_Code"
        / "flipo_flip_tracker.py"
    )
    cleaner_default = (
        project_root
        / "04_Code"
        / "scripts"
        / "Video_Analysi_Code"
        / "clean_roll_intervals.py"
    )
    raw_default = project_root / "01_Data" / "Raw"
    output_default = project_root / "01_Data" / "Processed"
    plots_default = project_root / "03_Results" / "Plots"

    tracker_script = _resolve_path(project_root, args.tracker_script, tracker_default)
    cleaner_script = _resolve_path(project_root, args.cleaner_script, cleaner_default)
    raw_dir = _resolve_path(project_root, args.raw_dir, raw_default)
    output_dir = _resolve_path(project_root, args.output_dir, output_default)
    plots_dir = _resolve_path(project_root, args.plots_dir, plots_default)

    if not tracker_script.exists() or not tracker_script.is_file():
        print(f"Tracker script not found: {tracker_script}")
        return 1
    if not cleaner_script.exists() or not cleaner_script.is_file():
        print(f"Cleaner script not found: {cleaner_script}")
        return 1

    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 88)
    print("Step 1/4: Analyze video and output raw CSV (interactive tracker)")
    before = _snapshot_raw_csv(raw_dir)

    tracker_command = [args.python_exe, str(tracker_script)]
    tracker_exit = _run_subprocess(tracker_command, cwd=project_root)
    if tracker_exit != 0:
        print(f"Tracker failed with exit code: {tracker_exit}")
        return tracker_exit

    print("=" * 88)
    print("Step 2/4: Detect raw CSV files produced/updated in this run")
    after = _snapshot_raw_csv(raw_dir)
    changed_csv_paths = _detect_changed_csv(before=before, after=after, raw_dir=raw_dir)

    if not changed_csv_paths:
        print("No new or updated raw CSV files were detected. Nothing to clean.")
        return 0

    print(f"Detected {len(changed_csv_paths)} raw CSV file(s) to process:")
    for path in changed_csv_paths:
        print(f" - {path}")

    print("=" * 88)
    print("Step 3/4 + Step 4/4: Clean/rank CSV and plot based on ranking")
    print(
        "Cleaner threshold config: "
        + f"min_roll_deg={args.min_roll_deg:.3f}, "
        + f"auto_relax={args.auto_relax_roll_threshold}, "
        + f"red_mode={args.red_ignore_theta_mode}, "
        + f"red_start_y_max={args.red_start_y_max:.3f}, "
        + f"red_end_y_min={args.red_end_y_min:.3f}"
    )

    failed_paths: List[Path] = []
    for index, csv_path in enumerate(changed_csv_paths, start=1):
        print("-" * 88)
        print(f"[{index}/{len(changed_csv_paths)}] Processing {csv_path.name}")

        cleaner_command = [
            args.python_exe,
            str(cleaner_script),
            "--csv-path",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--plots-dir",
            str(plots_dir),
            "--min-roll-deg",
            str(args.min_roll_deg),
            "--red-ignore-theta-mode",
            args.red_ignore_theta_mode,
            "--red-start-y-max",
            str(args.red_start_y_max),
            "--red-end-y-min",
            str(args.red_end_y_min),
            "--red-wobble-smoothing-alpha",
            str(args.red_wobble_smoothing_alpha),
            "--red-forward-dx-threshold",
            str(args.red_forward_dx_threshold),
            "--red-cycle-reset-threshold-deg",
            str(args.red_cycle_reset_threshold_deg),
        ]
        if args.auto_relax_roll_threshold:
            cleaner_command.append("--auto-relax-roll-threshold")
            cleaner_command.extend(["--relax-roll-seq", args.relax_roll_seq])

        cleaner_exit = _run_subprocess(cleaner_command, cwd=project_root)
        if cleaner_exit != 0:
            print(f"Failed: {csv_path} (exit code {cleaner_exit})")
            failed_paths.append(csv_path)

    print("=" * 88)
    print("Pipeline summary")
    print(f"Raw CSV selected for cleaning: {len(changed_csv_paths)}")
    print(f"Successful clean/rank/plot: {len(changed_csv_paths) - len(failed_paths)}")
    print(f"Failed clean/rank/plot: {len(failed_paths)}")
    print(f"Processed outputs dir: {output_dir}")
    print(f"Plots outputs dir: {plots_dir}")

    if failed_paths:
        print("Failed CSV files:")
        for path in failed_paths:
            print(f" - {path}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
