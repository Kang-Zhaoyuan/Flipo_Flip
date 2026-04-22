import mujoco
import mujoco_viewer 
import time
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

xml = """
<mujoco>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="1 1 0.1" rgba=".9 .9 .9 1"/>
    <body pos="0 0 1">
      <joint type="free"/>
      <geom type="box" size=".1 .2 .3" rgba="0 .9 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

try:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    viewer = mujoco_viewer.MujocoViewer(model, data)

    # 记录现实世界的开始时间
    start_time = time.time()

    while viewer.is_alive:
        # 计算自启动以来经过了多少现实时间
        time_elapsed = time.time() - start_time
        
        # 运行物理引擎，直到模拟时间追上现实时间
        while data.time < time_elapsed:
            mujoco.mj_step(model, data)
            
        # 将当前状态渲染到屏幕上
        viewer.render()

    viewer.close()
except Exception as e:
    print(f"运行出错: {e}")
    print("请检查是否已在虚拟环境中执行: pip install mujoco mujoco-python-viewer")