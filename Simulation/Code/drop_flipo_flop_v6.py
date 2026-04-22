import mujoco
import mujoco_viewer
import time
import os
import sys
import numpy as np


def get_mesh_dimensions_and_scale(stl_path, target_width=0.04):
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

        if raw_w <= 0:
                raise RuntimeError("Mesh width on X-axis is zero; cannot derive scale_factor.")

        scale_factor = target_width / raw_w
        width = raw_w * scale_factor
        length = raw_l * scale_factor
        height = raw_h * scale_factor
        return raw_w, raw_l, raw_h, scale_factor, width, length, height


def apply_no_slip_flick(model, data, omega_x):
    """Inject a no-slip flick by matching CoM velocity with pivot kinematics.

    Physical model:
        - Flipo Flop lies flat on the XY plane, thickness along X-axis (small).
        - The pivot edge runs along the Y-axis (long edge, length = L_scaled).
        - omega_x: angular velocity about the X-axis (the roll axis).

    No-slip condition at the front-bottom pivot edge:
        v_x = omega_mag * (H / 2)   <- CoM advances along X; arm = half-height (Z)
        v_z = omega_mag * (W / 2)   <- CoM lifts along Z;    arm = half-thickness (X)

    qvel layout (free joint): [vx, vy, vz, wx, wy, wz]

    Returns:
        tuple: (v_x, v_z, W, H, scale_factor)
            v_x        -- injected X velocity (forward, m/s)
            v_z        -- injected Z velocity (upward lift, m/s)
            W          -- scaled thickness along X (m)
            H          -- scaled height along Z (m)
            scale_factor
    """
    target_width = 0.04

    # Find the first mesh geom used by the compiled model.
    mesh_geom_id = None
    for geom_id in range(model.ngeom):
        if (
            model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH
            and model.geom_dataid[geom_id] >= 0
        ):
            mesh_geom_id = geom_id
            break

    if mesh_geom_id is None:
        raise RuntimeError("No mesh geom found for no-slip flick calculation.")

    mesh_id = model.geom_dataid[mesh_geom_id]

    verts = np.asarray(model.mesh_vert, dtype=float)
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)

    vert_adr = model.mesh_vertadr[mesh_id]
    vert_num = model.mesh_vertnum[mesh_id]
    mesh_verts = verts[vert_adr : vert_adr + vert_num]

    raw_w = np.max(mesh_verts[:, 0]) - np.min(mesh_verts[:, 0])  # thickness (X, small)
    raw_L = np.max(mesh_verts[:, 1]) - np.min(mesh_verts[:, 1])  # length (Y, pivot axis)
    raw_H = np.max(mesh_verts[:, 2]) - np.min(mesh_verts[:, 2])  # height (Z)

    if raw_w <= 0:
        raise RuntimeError("Mesh width on X-axis is zero; cannot derive scale_factor.")

    scale_factor = target_width / raw_w
    W = raw_w * scale_factor   # scaled thickness along X
    H = raw_H * scale_factor   # scaled height along Z

    omega_mag = abs(float(omega_x))

    # No-slip condition:
    #   v_x: CoM moves forward (X) as the toy rolls over the pivot edge.
    #        Arm = distance from CoM to pivot edge in Z direction = H / 2.
    v_x = omega_mag * (H / 2.0)

    #   v_z: CoM lifts upward (Z) as the toy tips over the pivot edge.
    #        Arm = distance from CoM to pivot edge in X direction = W / 2.
    #        This is small because W (thickness) is small.
    v_z = omega_mag * (W / 2.0)

    # Inject velocities into the free-joint state vector.
    # qvel layout: [vx, vy, vz, wx, wy, wz]
    data.qvel[0] = v_x   # forward (X)
    data.qvel[1] = 0.0   # no lateral motion (Y)
    data.qvel[2] = v_z   # upward lift (Z)
    data.qvel[3] = -omega_mag  # roll about X-axis (negative = forward flip)

    return v_x, v_z, W, H, scale_factor


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
    ) = get_mesh_dimensions_and_scale(stl_path, target_width=0.04)
except Exception as e:
    print(f"Failed to load model geometry. Error: {e}")
    sys.exit()

print("\n--- Geometry Info (Scaled) ---")
print(f"Width/Thickness (X): {geom_width:.6f} m   <- pivot arm for v_z")
print(f"Length          (Y): {geom_length:.6f} m   <- pivot edge direction")
print(f"Height          (Z): {geom_height:.6f} m   <- pivot arm for v_x")
print(f"raw_w (X): {raw_w:.6f}")
print(f"raw_l (Y): {raw_l:.6f}")
print(f"raw_h (Z): {raw_h:.6f}")

# 4. Flick Parameter (omega_x only; v_x and v_z are auto-derived)
print("\n--- Stage 4: Flick Parameters ---")
print("No-slip flick mode: v_x and v_z will be computed from omega_x and mesh size.")
print("  v_x (forward) = omega * (H/2),  v_z (lift) = omega * (W/2)  [W << H]")
try:
    wx_input = input("Enter forward angular spin (omega_x) in rad/s [Default: 20.0]: ")
    omega_x = float(wx_input) if wx_input.strip() else 20.0
except ValueError:
    print("Invalid input. Using default omega_x=20.0")
    omega_x = 20.0

# 5. Playback Speed
playback_speed = get_playback_speed(default_speed=1.0)

# 6. Dynamic Scaling (reused from geometry query)
target_width = geom_width
target_height = 1.5 * target_width

print("\n--- Simulation Ready ---")
print(f"Model: {selected_stl}")
print(f"Material: {selected_material_name} ({selected_density} kg/m^3)")
print(f"Target physical width: {target_width:.4f} m")
print(f"Playback speed: {playback_speed:.2f}x")
print("The simulation will wait 1.5s for the toy to settle, then apply no-slip flick.")
print("------------------------\n")

# 7. Build the Final XML
final_xml = f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <visual>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <mesh name="flipo_mesh" file="{stl_path}" scale="{scale_factor} {scale_factor} {scale_factor}"/>
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

# 8. Simulation Loop with No-Slip Flick Injection
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
            try:
                v_x, v_z, W, H, used_scale = apply_no_slip_flick(model, data, omega_x)
                flick_applied = True
                print(f"\n[!] NO-SLIP FLICK APPLIED at t={data.time:.3f}s")
                print(f"    -> omega_x       : {-abs(omega_x):.4f} rad/s")
                print(f"    -> W (thickness X): {W:.6f} m  (arm for v_z)")
                print(f"    -> H (height Z)   : {H:.6f} m  (arm for v_x)")
                print(f"    -> scale_factor   : {used_scale:.6f}")
                print(f"    -> injected v_x   : {v_x:.6f} m/s  (forward)")
                print(f"    -> injected v_z   : {v_z:.6f} m/s  (upward lift, should be small)")
            except RuntimeError as err:
                flick_applied = True
                print(f"\n[!] Flick injection failed: {err}")

    viewer.render()

viewer.close()
