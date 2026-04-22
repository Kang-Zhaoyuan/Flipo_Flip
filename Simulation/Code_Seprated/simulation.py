import time

import mujoco
import mujoco_viewer

from Simulation.Code_Seprated.config import SETTLE_TIME
from Simulation.Code_Seprated.flick import apply_manual_flick
from Simulation.Code_Seprated.telemetry import capture_sample


def run_simulation(model, data, v_x, v_z, omega, playback_speed):
	viewer = mujoco_viewer.MujocoViewer(model, data)
	start_time = time.time()
	flick_applied = False
	samples = []

	while viewer.is_alive:
		time_elapsed = (time.time() - start_time) * playback_speed

		while data.time < time_elapsed:
			prev_time = data.time
			mujoco.mj_step(model, data)
			event_flag = 0

			if (not flick_applied) and (prev_time < SETTLE_TIME <= data.time):
				v_x_applied, v_y_applied, v_z_applied, omega_y_applied = apply_manual_flick(
					data, v_x, v_z, omega
				)
				flick_applied = True
				event_flag = 1
				print(f"\n[!] MANUAL FLICK APPLIED at t={data.time:.3f}s")
				print(f"    -> injected v_x    : {v_x_applied:.6f} m/s")
				print(f"    -> injected v_y    : {v_y_applied:.6f} m/s")
				print(f"    -> injected v_z    : {v_z_applied:.6f} m/s")
				print(f"    -> injected omega_y: {omega_y_applied:.6f} rad/s")

			samples.append(capture_sample(data, event_flag=event_flag))

		viewer.render()

	viewer.close()
	return samples
