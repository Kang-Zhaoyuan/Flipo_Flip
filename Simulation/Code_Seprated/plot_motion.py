import os


def generate_motion_plots(samples, output_dir, run_id, meta_text=""):
	"""Generate trajectory and angle charts from telemetry samples."""
	if not samples:
		return []

	try:
		import matplotlib.pyplot as plt
		import numpy as np
		from matplotlib.collections import LineCollection
	except ImportError:
		print("[WARN] matplotlib/numpy unavailable. Skip plot generation.")
		return []

	os.makedirs(output_dir, exist_ok=True)

	t = np.array([s["time"] for s in samples], dtype=float)
	x = np.array([s["x"] for s in samples], dtype=float)
	z = np.array([s["z"] for s in samples], dtype=float)
	angle_deg = np.array([s["angle_y_deg"] for s in samples], dtype=float)
	wy = np.array([s["wy"] for s in samples], dtype=float)
	speed = np.array([s["speed_xz"] for s in samples], dtype=float)
	event_idx = next((i for i, s in enumerate(samples) if s["event_flag"] == 1), None)

	# Figure A: XoZ projection with color mapped to speed (blue->red)
	fig_a, ax_a = plt.subplots(figsize=(7.5, 6.0), dpi=140)
	if len(x) >= 2:
		points = np.column_stack([x, z]).reshape(-1, 1, 2)
		segments = np.concatenate([points[:-1], points[1:]], axis=1)
		segment_speed = 0.5 * (speed[:-1] + speed[1:])
		lc = LineCollection(segments, cmap="coolwarm", linewidths=2.0)
		lc.set_array(segment_speed)
		ax_a.add_collection(lc)
		fig_a.colorbar(lc, ax=ax_a, label="speed_xz (m/s)")
	ax_a.scatter([x[0]], [z[0]], c="green", s=50, label="start", zorder=3)
	ax_a.scatter([x[-1]], [z[-1]], c="black", s=50, label="end", zorder=3)
	if event_idx is not None:
		ax_a.scatter([x[event_idx]], [z[event_idx]], c="gold", edgecolors="black", s=70, label="flick", zorder=4)
	ax_a.set_xlabel("x (m)")
	ax_a.set_ylabel("z (m)")
	ax_a.set_title("Figure A: XoZ trajectory colored by speed")
	if meta_text:
		ax_a.text(0.01, 0.01, meta_text, transform=ax_a.transAxes, fontsize=8, va="bottom")
	ax_a.grid(True, alpha=0.3)
	ax_a.axis("equal")
	ax_a.legend(loc="best")
	fig_a.tight_layout()
	fig_a_path = os.path.join(output_dir, f"{run_id}_figureA_xoz_speed.png")
	fig_a.savefig(fig_a_path)
	plt.close(fig_a)

	# Figure B: angle_y and wy vs time
	fig_b, (ax_b1, ax_b2) = plt.subplots(2, 1, figsize=(9.0, 6.5), dpi=140, sharex=True)
	ax_b1.plot(t, angle_deg, color="tab:blue", linewidth=1.5)
	ax_b1.set_ylabel("angle_y (deg)")
	ax_b1.set_title("Figure B1: Y-axis rotation angle")
	ax_b1.grid(True, alpha=0.3)
	ax_b2.plot(t, wy, color="tab:red", linewidth=1.5)
	ax_b2.set_ylabel("wy (rad/s)")
	ax_b2.set_xlabel("time (s)")
	ax_b2.set_title("Figure B2: Angular velocity around Y")
	ax_b2.grid(True, alpha=0.3)
	if event_idx is not None:
		t_event = t[event_idx]
		ax_b1.axvline(t_event, color="goldenrod", linestyle="--", linewidth=1.2)
		ax_b2.axvline(t_event, color="goldenrod", linestyle="--", linewidth=1.2)
	fig_b.tight_layout()
	fig_b_path = os.path.join(output_dir, f"{run_id}_figureB_angle_wy.png")
	fig_b.savefig(fig_b_path)
	plt.close(fig_b)

	# Figure C: z vs time
	fig_c, ax_c = plt.subplots(figsize=(8.5, 4.5), dpi=140)
	ax_c.plot(t, z, color="tab:purple", linewidth=1.5)
	ax_c.set_xlabel("time (s)")
	ax_c.set_ylabel("z (m)")
	ax_c.set_title("Figure C: Height over time")
	if event_idx is not None:
		ax_c.axvline(t[event_idx], color="goldenrod", linestyle="--", linewidth=1.2, label="flick")
		ax_c.legend(loc="best")
	ax_c.grid(True, alpha=0.3)
	fig_c.tight_layout()
	fig_c_path = os.path.join(output_dir, f"{run_id}_figureC_z_time.png")
	fig_c.savefig(fig_c_path)
	plt.close(fig_c)

	return [fig_a_path, fig_b_path, fig_c_path]
