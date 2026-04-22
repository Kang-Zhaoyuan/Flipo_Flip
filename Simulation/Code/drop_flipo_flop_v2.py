import mujoco
import mujoco_viewer
import time
import os
import sys
import numpy as np

# 自动处理路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

models_dir = os.path.join(project_root, "All_3D_Models")

if not os.path.exists(models_dir):
    print(f"找不到文件夹: {models_dir}")
    sys.exit()

stl_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.stl')]

if not stl_files:
    print("未在 All_3D_Models 文件夹中找到任何 .stl 文件。")
    sys.exit()

# 1. 选择模型
print("在 All_3D_Models 中找到了以下模型:")
for i, file_name in enumerate(stl_files):
    print(f"[{i}] {file_name}")

try:
    choice = int(input("请输入要仿真的模型编号: "))
    if choice < 0 or choice >= len(stl_files):
        raise ValueError
except ValueError:
    print("无效的输入，已退出。")
    sys.exit()

selected_stl = stl_files[choice]
stl_path = os.path.join(models_dir, selected_stl).replace('\\', '/')

# 2. 材质与密度选择菜单
materials = {
    1: ("TPU", 1200),
    2: ("PETG", 1270),
    3: ("Nylon (尼龙)", 1140),
    4: ("Aluminum (铝合金)", 2700),
    5: ("Iron (铁)", 7870)
}

print("\n请选择 Flipo_Flop 的材质（用于计算质量和惯性）：")
for key, (name, density) in materials.items():
    print(f"[{key}] {name} (密度: {density} kg/m³)")

try:
    mat_choice = int(input("请输入材质编号: "))
    if mat_choice not in materials:
        raise ValueError
except ValueError:
    print("无效的输入，默认使用 PETG。")
    mat_choice = 2

selected_material_name, selected_density = materials[mat_choice]

# 3. 动态缩放计算
# 首先以 1:1 比例加载网格以获取原始尺寸
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
    print(f"无法加载模型。错误: {e}")
    sys.exit()

verts = np.array(temp_model.mesh_vert)
if verts.ndim == 1:
    verts = verts.reshape(-1, 3)

vert_adr = temp_model.mesh_vertadr[0]
vert_num = temp_model.mesh_vertnum[0]
mesh_verts = verts[vert_adr : vert_adr + vert_num]

raw_h = np.max(mesh_verts[:, 0]) - np.min(mesh_verts[:, 0])

# 强制将模型 X 轴宽度缩放至 4 厘米 (0.04 米)
target_width = 0.04
scale_factor = target_width / raw_h
target_height = 4 * target_width

print(f"\n--- 仿真参数确认 ---")
print(f"模型: {selected_stl}")
print(f"材质: {selected_material_name} ({selected_density} kg/m³)")
print(f"原始 X 尺寸: {raw_h:.4f} m, 缩放系数: {scale_factor:.6f}")
print(f"实际物理宽度: {target_width:.4f} m")
print(f"初始掉落高度: {target_height:.4f} m")
print(f"--------------------\n")

# 4. 创建最终的 XML
# 将视觉呈现 (Visual) 与 物理碰撞 (Collision) 分离
final_xml = f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  
  <asset>
    <mesh name="flipo_mesh" file="{stl_path}" scale="{scale_factor} {scale_factor} {scale_factor}"/>
  </asset>
  
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 5" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".8 .8 .8 1"/>
    
    <body pos="0 0 {target_height}">
      <freejoint/>
      
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

start_time = time.time()

while viewer.is_alive:
    time_elapsed = time.time() - start_time
    while data.time < time_elapsed:
        mujoco.mj_step(model, data)
    viewer.render()

viewer.close()