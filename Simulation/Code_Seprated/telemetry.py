import csv
import math
import os
from datetime import datetime


def _quat_to_pitch_y_rad(quat_wxyz):
	"""Return rotation around world Y axis (pitch) from quaternion [w, x, y, z]."""
	w, x, y, z = [float(v) for v in quat_wxyz]
	sin_pitch = 2.0 * (w * y - z * x)
	sin_pitch = max(-1.0, min(1.0, sin_pitch))
	return math.asin(sin_pitch)


def capture_sample(data, event_flag=0):
	"""Capture one telemetry sample after mj_step."""
	t = float(data.time)
	x = float(data.qpos[0])
	y = float(data.qpos[1])
	z = float(data.qpos[2])
	quat = data.qpos[3:7]
	angle_y = _quat_to_pitch_y_rad(quat)
	vx = float(data.qvel[0])
	vz = float(data.qvel[2])
	wy = float(data.qvel[4])
	speed_xz = math.sqrt(vx * vx + vz * vz)
	return {
		"time": t,
		"x": x,
		"y": y,
		"z": z,
		"angle_y_rad": angle_y,
		"angle_y_deg": math.degrees(angle_y),
		"wy": wy,
		"speed_xz": speed_xz,
		"event_flag": int(event_flag),
	}


def make_run_id(prefix="run"):
	ts = datetime.now().strftime("%Y%m%d_%H%M%S")
	return f"{prefix}_{ts}"


def ensure_dir(path):
	os.makedirs(path, exist_ok=True)


def save_telemetry_csv(samples, output_dir, run_id):
	ensure_dir(output_dir)
	csv_path = os.path.join(output_dir, f"{run_id}_telemetry.csv")
	fieldnames = [
		"time",
		"x",
		"y",
		"z",
		"angle_y_rad",
		"angle_y_deg",
		"wy",
		"speed_xz",
		"event_flag",
	]
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		for row in samples:
			writer.writerow(row)
	return csv_path
