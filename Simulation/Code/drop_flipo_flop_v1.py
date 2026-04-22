# Currently, the code simply throws the .stl file onto the flat surface. 
# This is the initial version.

import mujoco
import mujoco_viewer
import time
import os
import sys
import numpy as np

# 1. 路径自动配置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

models_dir = os.path.join(project_root, "All_3D_Models")

# 检查模型文件夹是否存在
if not os.path.exists(models_dir):
    print(f"找不到文件夹: {models_dir}")
    sys.exit()

# 获取所有的 .stl 文件
stl_files = [f for f in os.listdir(models_dir) if f.lower().endswith('.stl')]

if not stl_files:
    print("未在 All_3D_Models 文件夹中找到任何 .stl 文件。")
    sys.exit()

# 2. 用户交互选择模型
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
stl_path = os.path.join(models_dir, selected_stl)

# 将路径转换为正斜杠，避免在生成 XML 字符串时由于反斜杠转义报错
stl_path_str = stl_path.replace('\\', '/')

# 3. 计算包围盒 X 尺寸 (h)
# 我们首先生成一个临时的 XML，用于让 MuJoCo 加载网格并应用 scale 进行编译
temp_xml = f"""
<mujoco>
  <asset>
    <mesh name="flipo_mesh" file="{stl_path_str}" scale="0.001 0.001 0.001"/>
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
    print(f"无法加载模型，可能是 STL 文件损坏。错误: {e}")
    sys.exit()

# 获取网格顶点数据并重塑为 (N, 3) 的二维数组
verts = np.array(temp_model.mesh_vert)
if verts.ndim == 1:
    verts = verts.reshape(-1, 3)

# 根据 mesh 的地址和数量提取特定网格的顶点
vert_adr = temp_model.mesh_vertadr[0]
vert_num = temp_model.mesh_vertnum[0]
mesh_verts = verts[vert_adr : vert_adr + vert_num]

# 提取 X 坐标，计算尺寸 h
x_coords = mesh_verts[:, 0]
h = np.max(x_coords) - np.min(x_coords)
target_height = 4 * h

print(f"\n加载成功: {selected_stl}")
print(f"模型 X 轴尺寸 (h): {h:.4f} 米")
print(f"初始掉落高度 (4*h): {target_height:.4f} 米\n")

# 4. 创建最终的带物理特性的完整仿真环境
final_xml = f"""
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <asset>
    <mesh name="flipo_mesh" file="{stl_path_str}" scale="0.001 0.001 0.001"/>
  </asset>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 5" dir="0 0 -1"/>
    <geom type="plane" size="2 2 0.1" rgba=".8 .8 .8 1"/>
    
    <body pos="0 0 {target_height}">
      <freejoint/>
      <geom type="mesh" mesh="flipo_mesh" rgba="0.2 0.6 0.8 1" condim="3" friction="1 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(final_xml)
data = mujoco.MjData(model)
viewer = mujoco_viewer.MujocoViewer(model, data)

start_time = time.time()

# 5. 开始带有真实时间同步的仿真主循环
while viewer.is_alive:
    time_elapsed = time.time() - start_time
    while data.time < time_elapsed:
        mujoco.mj_step(model, data)
    viewer.render()

viewer.close()