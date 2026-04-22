def compute_flick_velocities_from_omega(omega, y_length, h_ratio, l_ratio):
	"""Compute v_x and v_z from omega and the requested H/L proportion.

	Relationship:
		v_x = omega * (Y / 2)
		v_x / v_z = H / L
	"""
	if h_ratio == 0:
		raise ValueError("H ratio must be non-zero.")

	v_x = float(omega) * (float(y_length) / 2.0)
	v_z = v_x * (float(l_ratio) / float(h_ratio))
	return float(v_x), float(v_z)


def apply_manual_flick(data, v_x, v_z, omega):
	"""Inject translational and angular velocities.

	qvel layout (free joint): [vx, vy, vz, wx, wy, wz]
	"""
	omega_y_applied = abs(float(omega))
	v_y = 0.0

	data.qvel[0] = float(v_x)
	data.qvel[1] = v_y
	data.qvel[2] = float(v_z)
	data.qvel[3] = 0.0
	data.qvel[4] = omega_y_applied

	return float(v_x), v_y, float(v_z), omega_y_applied
