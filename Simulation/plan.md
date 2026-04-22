# drop_flipo_flop_v5 功能实现说明（Plan）

## 1. 背景与目标
- 研究对象是 Flipo Flop 滚动玩具。
- 几何输入来自 All_3D_Models 目录下的 STL 模型文件。
- 本程序通过 MuJoCo 构建统一可比的滚动仿真环境，并在落地后施加一次 flick 脉冲观察滚动状态。

## 2. 程序实现目标
- 支持从 All_3D_Models 中选择任意 STL 模型进行仿真。
- 支持按材料密度切换质量属性（TPU、PETG、Nylon、Aluminum、Steel）。
- 对不同几何模型做统一尺度归一化，保证跨模型对比公平。
- 在同一物理场景下，通过 v_y 和 omega_x 控制前向滚动行为。

## 3. 关键流程（按执行顺序）
1. 扫描并展示 STL 模型列表，用户选择模型。
2. 用户选择材料类型，得到 density。
3. 用户输入 flick 参数：v_y（线速度）与 omega_x（角速度）。
4. 临时加载 mesh 获取顶点范围，计算 scale_factor 使模型宽度统一到 0.04 m。
5. 构建正式 XML：地面、重力、free joint、mesh 几何与碰撞参数。
6. 进入仿真循环；当时间达到 settle_time（1.5 s）后，一次性写入 qvel 脉冲。
7. 持续步进与渲染，观察滚动与翻转状态。

## 4. STL 处理机制
- 程序先用临时模型读取 mesh 顶点，再基于 X 轴尺寸计算原始宽度 raw_h。
- 归一化公式：
  scale_factor = target_width / raw_h
- 其中 target_width = 0.04 m。
- 正式模型中使用统一 scale_factor 加载 STL，降低“模型原始尺寸差异”对结果的干扰。

## 5. 物理参数与含义
- gravity = 0 0 -9.81：标准重力场。
- timestep = 0.002：仿真时间步长。
- density：决定刚体质量与惯性（由材料菜单给定）。
- friction = 1 0.1 0.01：滑动/滚动/扭转摩擦。
- damping = 0.001：自由关节阻尼，抑制过度振荡。
- settle_time = 1.5：flick 注入等待时间，确保先落地再施加脉冲。

## 6. v_y 与 omega_x 的功能定位
- v_y：Y 方向线速度（m/s），用于给玩具前向推进。
- omega_x：绕 X 轴角速度（rad/s），用于触发翻滚。
- 在 flick 时刻写入：
  - data.qvel[1] = v_y
  - data.qvel[3] = -omega_x
- 这表示采用“Y 向平动 + X 轴旋转”的组合来驱动滚动动作。

## 7. 设计要点
- 采用“双 geom”结构：
  - 一个 geom 承担质量属性（density），关闭接触。
  - 一个 geom 承担接触与摩擦（mass=0），用于碰撞求解。
- 该设计将“质量建模”和“接触建模”分离，调参更清晰。

## 8. 与旧版本的关键差异（摘要）
- v1-v3：主要是基础掉落与参数完善，尚未形成当前 flick 轴向方案。
- v4：引入 flick 控制框架。
- v5：将控制轴重映射为 v_y + omega_x，匹配当前前向滚动研究需求。

## 9. 实施与验证计划
1. 先固定一个 STL 模型，扫描 v_y 和 omega_x 的小范围参数网格。
2. 再切换不同材料 density，比较滚动稳定性和翻转频率变化。
3. 最后更换 STL 模型，验证归一化后不同外形的状态差异。

## 10. 成功判据
- 在相同场景和归一化尺度下，不同模型可稳定复现可比较的滚动状态。
- v_y 与 omega_x 调整能产生可解释的状态变化趋势。
- 材料密度变化对滚动表现有一致、可观测的影响。

## 11. 代码证据入口
- 主实现文件：Code/drop_flipo_flop_v5.py
- 参考版本：Code/drop_flipo_flop_v4.py, Code/drop_flipo_flop_v3.py
- 模型输入目录：All_3D_Models
