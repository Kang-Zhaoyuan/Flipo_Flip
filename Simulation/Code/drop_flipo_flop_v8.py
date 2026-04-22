import mujoco
import mujoco_viewer
import time
import os
import sys
import numpy as np


def _is_likely_ascii_stl(stl_path, probe_bytes=512):
	"""Heuristic check for ASCII STL by probing file header text."""
	try:
		with open(stl_path, "rb") as f:
			header = f.read(probe_bytes)
	except OSError:
		return False

	header_l = header.lower()
	return header_l.startswith(b"solid") and (b"facet" in header_l)


def ensure_mujoco_compatible_stl(stl_path):
	"""Convert ASCII STL to binary STL for MuJoCo if needed.

	Returns:
		tuple: (mesh_path_to_use, converted_flag)
	"""
	if not _is_likely_ascii_stl(stl_path):
		return stl_path, False

	try:
		from stl import mesh, Mode
	except ImportError:
		print("[WARN] Selected STL looks like ASCII. MuJoCo may fail to decode it.")
		print("[WARN] Install numpy-stl to auto-convert: pip install numpy-stl")
		return stl_path, False

	converted_path = os.path.splitext(stl_path)[0] + "_mujoco_binary.stl"
	try:
		src_mesh = mesh.Mesh.from_file(stl_path)
		src_mesh.save(converted_path, mode=Mode.BINARY)
		print(f"[INFO] Converted ASCII STL -> Binary STL: {converted_path}")
		return converted_path, True
	except Exception as e:
		print(f"[WARN] Failed to convert ASCII STL with numpy-stl: {e}")
		return stl_path, False


def get_mesh_dimensions_and_scale(stl_path, target_length=0.35):
		"""Load mesh once and return raw/scaled dimensions and scale factor.

		Returns:
				tuple: (raw_w, raw_l, raw_h, scale_factor, width, length, height)
		"""
		temp_xml = f"""
<mujoco>
	<asset>
		<mesh name="flipo_mesh" file="{stl_path}" scale="1 1 1"/>
	</asset>
	<worldbody>
		<body pos="0 0 0">
			<geom type="mesh" mesh="flipo_mesh"/>
		</body>
	</worldbody>
</mujoco>
"""

		temp_model = mujoco.MjModel.from_xml_string(temp_xml)
		verts = np.asarray(temp_model.mesh_vert, dtype=float)
		if verts.ndim == 1:
				verts = verts.reshape(-1, 3)

		vert_adr = temp_model.mesh_vertadr[0]
		vert_num = temp_model.mesh_vertnum[0]
		mesh_verts = verts[vert_adr : vert_adr + vert_num]

		raw_w = np.max(mesh_verts[:, 0]) - np.min(mesh_verts[:, 0])
		raw_l = np.max(mesh_verts[:, 1]) - np.min(mesh_verts[:, 1])
		raw_h = np.max(mesh_verts[:, 2]) - np.min(mesh_verts[:, 2])

		if raw_l <= 0:
				raise RuntimeError("Mesh length on Y-axis is zero; cannot derive scale_factor.")

		scale_factor = target_length / raw_l
		width = raw_w * scale_factor
		length = raw_l * scale_factor
		height = raw_h * scale_factor
		return raw_w, raw_l, raw_h, scale_factor, width, length, height


def apply_manual_flick(data, v_x, v_z, omega):
	"""Inject manually specified translational and angular velocities.

	qvel layout (free joint): [vx, vy, vz, wx, wy, wz]

	Returns:
		tuple: (v_x, v_y, v_z, omega_y_applied)
	"""
	omega_y_applied = abs(float(omega))
	v_y = 0.0

	data.qvel[0] = float(v_x)
	data.qvel[1] = v_y
	data.qvel[2] = float(v_z)
	data.qvel[3] = 0.0
	data.qvel[4] = omega_y_applied

	return float(v_x), v_y, float(v_z), omega_y_applied


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


def get_playback_speed(default_speed=1.0):
	"""Get a user-friendly playback speed multiplier.

	Examples:
		0.5 -> half speed
		1.0 -> real-time
		2.0 -> double speed
	"""
	print("\n--- Stage 5: Playback Speed ---")
	print("Set simulation playback speed (e.g., 0.5=half, 1.0=normal, 2.0=double).")
	user_input = input(f"Enter playback speed multiplier [Default: {default_speed}]: ")

	if not user_input.strip():
		return float(default_speed)

	try:
		speed = float(user_input)
		if speed <= 0:
			raise ValueError
		return speed
	except ValueError:
		print(f"Invalid speed. Using default: {default_speed}x")
		return float(default_speed)


# 1. Automatic path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
	sys.path.append(project_root)

models_dir = os.path.join(project_root, "All_3D_Models")

if not os.path.exists(models_dir):
	print(f"Directory not found: {models_dir}")
	sys.exit()

stl_files = [f for f in os.listdir(models_dir) if f.lower().endswith(".stl")]

if not stl_files:
	print("No .stl files found in the All_3D_Models directory.")
	sys.exit()

# 2. Interactive Model Selection
print("Found the following models in All_3D_Models:")
for i, file_name in enumerate(stl_files):
	print(f"[{i}] {file_name}")

try:
	choice = int(input("Enter the number of the model to simulate: "))
	if choice < 0 or choice >= len(stl_files):
		raise ValueError
except ValueError:
	print("Invalid input, exiting.")
	sys.exit()

selected_stl = stl_files[choice]
stl_path = os.path.join(models_dir, selected_stl).replace("\\", "/")
mesh_path, converted_for_mujoco = ensure_mujoco_compatible_stl(stl_path)

# 3. Material and Density Selection
materials = {
	1: ("TPU", 1200),
	2: ("PETG", 1270),
	3: ("Nylon", 1140),
	4: ("Aluminum", 2700),
	5: ("Iron/Steel", 7870),
}

print("\nSelect the material for the Flipo Flop (used to calculate mass/inertia):")
for key, (name, density) in materials.items():
	print(f"[{key}] {name} (Density: {density} kg/m^3)")

try:
	mat_choice = int(input("Enter the material number: "))
	if mat_choice not in materials:
		raise ValueError
except ValueError:
	print("Invalid input, defaulting to PETG.")
	mat_choice = 2

selected_material_name, selected_density = materials[mat_choice]

try:
	(
		raw_w,
		raw_l,
		raw_h,
		scale_factor,
		geom_width,
		geom_length,
		geom_height,
	) = get_mesh_dimensions_and_scale(mesh_path, target_length=0.35)
except Exception as e:
	print(f"Failed to load model geometry. Error: {e}")
	sys.exit()

print("\n--- Geometry Info (Scaled) ---")
print(f"Width/Thickness (X): {geom_width:.6f} m")
print(f"Length          (Y): {geom_length:.6f} m")
print(f"Height          (Z): {geom_height:.6f} m")
print(f"raw_w (X): {raw_w:.6f}")
print(f"raw_l (Y): {raw_l:.6f}")
print(f"raw_h (Z): {raw_h:.6f}")

# 4. Flick Parameters (auto-computed from omega)
print("\n--- Stage 4: Flick Parameters ---")
print("Auto flick mode: input omega, then v_x and v_z are computed automatically.")

# User-defined proportion parameters (dimensionless ratios)
H = 35.0
L = 1.4479973796

try:
	wx_input = input("Enter angular speed omega in rad/s [Default: 20.0]: ")
	omega = float(wx_input) if wx_input.strip() else 20.0
except ValueError:
	print("Invalid input. Using default omega=20.0")
	omega = 20.0

v_x, v_z = compute_flick_velocities_from_omega(
	omega=omega,
	y_length=geom_length,
	h_ratio=H,
	l_ratio=L,
)

print(f"H ratio parameter: {H:.10f}")
print(f"L ratio parameter: {L:.10f}")
print(f"Y (scaled model length): {geom_length:.6f} m")
print(f"Computed v_x from omega: {v_x:.6f} m/s")
print(f"Computed v_z from H/L ratio: {v_z:.6f} m/s")
print(f"Check v_x / v_z: {(v_x / v_z):.6f} (target H/L={H / L:.6f})")

# 5. Playback Speed
playback_speed = get_playback_speed(default_speed=1.0)

# 6. Dynamic Scaling (reused from geometry query)
target_width = geom_width
target_height = 1.5 * target_width

print("\n--- Simulation Ready ---")
print(f"Model: {selected_stl}")
if converted_for_mujoco:
	print("Mesh input: ASCII STL auto-converted to binary for MuJoCo")
print(f"Material: {selected_material_name} ({selected_density} kg/m^3)")
print(f"Target physical width: {target_width:.4f} m")
print(f"Playback speed: {playback_speed:.2f}x")
print("The simulation will wait 1.5s for the toy to settle, then apply manual flick.")
print("------------------------\n")

# 7. Build the Final XML
final_xml = f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
	<quality shadowsize="4096"/>
  </visual>

  <asset>
	<mesh name="flipo_mesh" file="{mesh_path}" scale="{scale_factor} {scale_factor} {scale_factor}"/>
  </asset>

  <worldbody>
	<light diffuse=".5 .5 .5" pos="0 0 5" dir="0 0 -1"/>

	<geom type="plane" size="3 3 0.1" rgba=".8 .8 .8 1"
		  solimp="0.9 0.99 0.001" solref="0.02 1"/>

	<body pos="0 0 {target_height}">
	  <joint type="free" damping="0.0013"/>

	  <geom type="mesh" mesh="flipo_mesh" rgba="0.2 0.6 0.8 1"
			contype="0" conaffinity="0" density="{selected_density}"/>

	  <geom type="mesh" mesh="flipo_mesh" rgba="1 1 1 0"
			mass="0" condim="4" friction="1 0.1 0.01"/>
	</body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(final_xml)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

# 8. Simulation Loop with Manual Flick Injection
start_time = time.time()
flick_applied = False
settle_time = 1.5

while viewer.is_alive:
	# Speed control: simulation time follows wall time scaled by playback_speed.
	time_elapsed = (time.time() - start_time) * playback_speed

	while data.time < time_elapsed:
		prev_time = data.time
		mujoco.mj_step(model, data)

		# Trigger once when simulation time crosses settle_time.
		if (not flick_applied) and (prev_time < settle_time <= data.time):
			v_x_applied, v_y_applied, v_z_applied, omega_y_applied = apply_manual_flick(
				data, v_x, v_z, omega
			)
			flick_applied = True
			print(f"\n[!] MANUAL FLICK APPLIED at t={data.time:.3f}s")
			print(f"    -> injected v_x    : {v_x_applied:.6f} m/s")
			print(f"    -> injected v_y    : {v_y_applied:.6f} m/s")
			print(f"    -> injected v_z    : {v_z_applied:.6f} m/s")
			print(f"    -> injected omega_y: {omega_y_applied:.6f} rad/s")

	viewer.render()

viewer.close()
