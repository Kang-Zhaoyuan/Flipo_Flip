import mujoco
import mujoco_viewer
import time
import os
import sys
import numpy as np

# 1. Automatic path configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

models_dir = os.path.join(project_root, "All_3D_Models")

if not os.path.exists(models_dir):
    print(f"Directory not found: {models_dir}")
    sys.exit()

stl_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.stl')]

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
stl_path = os.path.join(models_dir, selected_stl).replace('\\', '/')

# 3. Material and Density Selection
materials = {
    1: ("TPU", 1200),
    2: ("PETG", 1270),
    3: ("Nylon", 1140),
    4: ("Aluminum", 2700),
    5: ("Iron/Steel", 7870)
}

print("\nSelect the material for the Flipo Flop (used to calculate mass/inertia):")
for key, (name, density) in materials.items():
    print(f"[{key}] {name} (Density: {density} kg/m³)")

try:
    mat_choice = int(input("Enter the material number: "))
    if mat_choice not in materials:
        raise ValueError
except ValueError:
    print("Invalid input, defaulting to PETG.")
    mat_choice = 2

selected_material_name, selected_density = materials[mat_choice]

# 4. Flick Parameters (Stage 4 Addition)
print("\n--- Stage 4: Flick Parameters ---")
print("We will let the toy settle on the floor, then apply an instant 'flick'.")
print("Press Enter to use the recommended default values, or type your own.")
try:
    vx_input = input("Enter forward velocity (v_x) in m/s [Default: 1.5]: ")
    v_x = float(vx_input) if vx_input.strip() else 1.5
    
    wy_input = input("Enter angular spin (omega_y) in rad/s [Default: 20.0]: ")
    omega_y = float(wy_input) if wy_input.strip() else 20.0
except ValueError:
    print("Invalid input. Using defaults: v_x=1.5, omega_y=20.0")
    v_x, omega_y = 1.5, 20.0

# 5. Dynamic Scaling Calculation
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

try:
    temp_model = mujoco.MjModel.from_xml_string(temp_xml)
except Exception as e:
    print(f"Failed to load model. Error: {e}")
    sys.exit()

verts = np.array(temp_model.mesh_vert)
if verts.ndim == 1:
    verts = verts.reshape(-1, 3)

vert_adr = temp_model.mesh_vertadr[0]
vert_num = temp_model.mesh_vertnum[0]
mesh_verts = verts[vert_adr : vert_adr + vert_num]

raw_h = np.max(mesh_verts[:, 0]) - np.min(mesh_verts[:, 0])

# Force the object X-axis width to exactly 4 centimeters (0.04 meters)
target_width = 0.04
scale_factor = target_width / raw_h
# Lowered drop height so it reaches the ground and settles faster
target_height = 1.5 * target_width 

print(f"\n--- Simulation Ready ---")
print(f"Model: {selected_stl}")
print(f"Material: {selected_material_name} ({selected_density} kg/m³)")
print(f"Target physical width: {target_width:.4f} m")
print(f"The simulation will wait 1.5s for the toy to settle, then flick it.")
print(f"------------------------\n")

# 6. Build the Final XML
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
      <joint type="free" damping="0.001"/>
      
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

# 7. Simulation Loop with State Injection
start_time = time.time()
flick_applied = False
settle_time = 1.5  # Wait 1.5 seconds of physics time before flicking

while viewer.is_alive:
    time_elapsed = time.time() - start_time
    
    while data.time < time_elapsed:
        mujoco.mj_step(model, data)
        
        # Inject the impulse (flick) at exactly the settle_time marker
        if data.time >= settle_time and not flick_applied:
            # data.qvel is an array mapping to [vx, vy, vz, wx, wy, wz] for a free joint
            # We apply forward linear velocity (X) and forward pitch (Y)
            data.qvel[0] = v_x
            data.qvel[4] = omega_y
            
            flick_applied = True
            print(f"\n[!] FLICK APPLIED at t={data.time:.2f}s")
            print(f"    -> v_x: {v_x} m/s")
            print(f"    -> omega_y: {omega_y} rad/s")

    viewer.render()

viewer.close()