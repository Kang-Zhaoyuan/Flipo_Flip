from Simulation.Code_Seprated.config import DEFAULT_OMEGA, DEFAULT_PLAYBACK_SPEED, MATERIALS


def select_model(stl_files):
	print("Found the following models in All_3D_Models:")
	for i, file_name in enumerate(stl_files):
		print(f"[{i}] {file_name}")

	try:
		choice = int(input("Enter the number of the model to simulate: "))
		if choice < 0 or choice >= len(stl_files):
			raise ValueError
	except ValueError:
		print("Invalid input, exiting.")
		raise SystemExit

	return stl_files[choice]


def select_material():
	print("\nSelect the material for the Flipo Flop (used to calculate mass/inertia):")
	for key, (name, density) in MATERIALS.items():
		print(f"[{key}] {name} (Density: {density} kg/m^3)")

	try:
		mat_choice = int(input("Enter the material number: "))
		if mat_choice not in MATERIALS:
			raise ValueError
	except ValueError:
		print("Invalid input, defaulting to PETG.")
		mat_choice = 2

	return MATERIALS[mat_choice]


def input_omega():
	print("\n--- Stage 4: Flick Parameters ---")
	print("Auto flick mode: input omega, then v_x and v_z are computed automatically.")
	try:
		wx_input = input(f"Enter angular speed omega in rad/s [Default: {DEFAULT_OMEGA}]: ")
		omega = float(wx_input) if wx_input.strip() else float(DEFAULT_OMEGA)
	except ValueError:
		print(f"Invalid input. Using default omega={DEFAULT_OMEGA}")
		omega = float(DEFAULT_OMEGA)
	return omega


def get_playback_speed():
	print("\n--- Stage 5: Playback Speed ---")
	print("Set simulation playback speed (e.g., 0.5=half, 1.0=normal, 2.0=double).")
	user_input = input(f"Enter playback speed multiplier [Default: {DEFAULT_PLAYBACK_SPEED}]: ")

	if not user_input.strip():
		return float(DEFAULT_PLAYBACK_SPEED)

	try:
		speed = float(user_input)
		if speed <= 0:
			raise ValueError
		return speed
	except ValueError:
		print(f"Invalid speed. Using default: {DEFAULT_PLAYBACK_SPEED}x")
		return float(DEFAULT_PLAYBACK_SPEED)


def print_geometry_info(raw_w, raw_l, raw_h, geom_width, geom_length, geom_height):
	print("\n--- Geometry Info (Scaled) ---")
	print(f"Width/Thickness (X): {geom_width:.6f} m")
	print(f"Length          (Y): {geom_length:.6f} m")
	print(f"Height          (Z): {geom_height:.6f} m")
	print(f"raw_w (X): {raw_w:.6f}")
	print(f"raw_l (Y): {raw_l:.6f}")
	print(f"raw_h (Z): {raw_h:.6f}")


def print_flick_computation(h_ratio, l_ratio, geom_length, v_x, v_z):
	print(f"H ratio parameter: {h_ratio:.10f}")
	print(f"L ratio parameter: {l_ratio:.10f}")
	print(f"Y (scaled model length): {geom_length:.6f} m")
	print(f"Computed v_x from omega: {v_x:.6f} m/s")
	print(f"Computed v_z from H/L ratio: {v_z:.6f} m/s")
	print(f"Check v_x / v_z: {(v_x / v_z):.6f} (target H/L={h_ratio / l_ratio:.6f})")


def print_simulation_ready(selected_stl, converted_for_mujoco, material_name, target_width, playback_speed):
	print("\n--- Simulation Ready ---")
	print(f"Model: {selected_stl}")
	if converted_for_mujoco:
		print("Mesh input: ASCII STL auto-converted to binary for MuJoCo")
	print(f"Material: {material_name}")
	print(f"Target physical width: {target_width:.4f} m")
	print(f"Playback speed: {playback_speed:.2f}x")
	print("The simulation will wait 1.5s for the toy to settle, then apply manual flick.")
	print("------------------------\n")
