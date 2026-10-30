# 阶段一：仿真执行基础与验收

2026-10-05 已实现，可直接用项目现有 Python 环境运行。语义和动作计划由人工指定，尚未接入 LLM/VLM、PDDL、多 Agent 协商或自动重规划。本阶段验证结构化任务能否可靠执行，为后续 Agent 提供工具接口与观测结果。

## 已实现什么

- 确定性 GridWorld，提供 `reset()`、`observe()`、`step(action)`。
- 严格的 `WorldState`、`SemanticFrame`、`Goal`、`Plan`、`ActionCall` 输入边界；拒绝多余字段、错误类型和不一致状态。
- 五种白名单工具；检查动作参数、位置、持物和容器状态。
- 完整 `ExecutionTrace`，记录原始指令、人工目标/计划、初始状态、每步前后状态、导航路径、错误和最终状态。
- 回放命令：从日志中的初始状态重新执行全部已接受动作，比较各步和最终目标。
- 六个内置场景、自动测试、终端逐步网格回放和 UTF-8 文件输出。

现有语音页面和 ASR/LLM/TTS 配置继续可用。本阶段无新增依赖；不需要启动 Ollama、下载视觉模型、启用麦克风或安装 ROS/WSL。

## 场景和动作约定

默认地图为 7×5，左侧客厅、右侧厨房，中间墙壁仅在 `(3,2)` 留通道。坐标从零开始，x 向右、y 向下。

```text
. . . # . . .
. T . # . S .
R . . . . . .
. . . # . B .
. . . # . . .
```

`R` 是机器人，`#` 是墙；`T/S/B` 分别为桌子、架子、盒子的**交互位置**。机器人到达交互格后用 `R` 覆盖字母。设施没有实体体积，不做机械臂动力学模拟。

初始物品：红杯 `red_cup`、蓝杯 `blue_cup` 在 `table`；苹果 `apple` 在关闭的 `box`；`shelf` 为空。机器人一次只能拿一件物品，桌面、架子和盒子不设容量上限。

| 工具 | 严格参数 | 前置条件与效果 |
| --- | --- | --- |
| `navigate` | `target` | 目标设施存在、路径可达；BFS 返回最短网格路径，再移动到目标交互格 |
| `pick` | `object_id` | 物品存在、手为空、已到其设施、容器已打开；物品位置变为 `robot` |
| `place` | `object_id`, `target` | 正拿着该物品、已到目标、容器已打开；更新物品位置并清空手 |
| `open` | `target` | 目标是容器、机器人在其交互格；打开盒盖；重复打开不改变状态 |
| `inspect` | 无，使用 `{}` | 返回完整状态，不修改场景 |

`inspect` 是全观测，包括盒内物品，不模拟视觉遮挡。导航是一次高层工具调用，但日志保存逐格路径；这里的 BFS 是导航算法，不是 PDDL 任务规划器。

失败动作不改变世界状态或 `revision`，但保留之前已成功动作的效果。运行器在第一次失败后停止，后续计划不再执行。状态快照不可变，各环境实例互不共享状态。

## 在本机如何验收

在项目根目录 PowerShell 运行以下命令。当前机器已有 `.venv`；在新机器上先按 README 安装项目和 `dev` 依赖。`outputs/` 不进入 Git，可由以下命令重新生成。

### 1. 自动化验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_simulation.py -q
.\.venv\Scripts\python.exe -m pytest -q
```

当前预期为 **51 项仿真测试通过，全部 75 项测试通过**（原语音部分 24 项）。测试覆盖实际状态转移、绕墙路径、失败不修改状态、关闭容器、双重持物、无效参数、状态隔离、重置、日志不一致检测和 Windows 管道中的中文编码。

### 2. 成功演示：移动红杯

```powershell
.\.venv\Scripts\python.exe -m voice_lab.simulation run --scenario move-cup
```

观察 `red_cup@table → red_cup@robot → red_cup@shelf`，蓝杯保持在桌上。导航路径必须经过 `(3,2)` 的通道。最终显示：

```text
status=completed; goal_satisfied=true; executed=6; skipped=0
```

### 3. 失败演示：未开盒就取物

```powershell
.\.venv\Scripts\python.exe -m voice_lab.simulation run --scenario closed-box
```

预期错误码为 `CONTAINER_CLOSED`，苹果仍在盒里，盒盖仍关闭，手为空；第 3、4 步没有执行。最终显示：

```text
status=failed; goal_satisfied=false; executed=2; skipped=2
```

这是故意安排的失败场景，命令退出码 `1` 表示任务未完成。`run` 退出码 `0` 表示完成，`2` 表示输入/文件错误；所有动作成功但目标未满足会显示 `incomplete`，同样退出 `1`。

### 4. 验证日志能否复现

```powershell
.\.venv\Scripts\python.exe -m voice_lab.simulation replay outputs/simulation/move-cup/trace.json
.\.venv\Scripts\python.exe -m voice_lab.simulation replay outputs/simulation/closed-box/trace.json
```

两条都应显示 `REPLAY VERIFIED` 并退出 `0`。失败任务也能正确复现。若只修改日志中的某一步结果、路径或最终状态，回放应提示“不一致”并退出 `2`。这验证执行的一致性，不是对日志来源的签名认证。

每次运行保存三个文件：

| 文件 | 用途 |
| --- | --- |
| `outputs/simulation/<场景>/scenario.json` | 可编辑、可再次输入的完整初始状态、目标和人工动作计划 |
| `outputs/simulation/<场景>/trace.json` | 逐动作结果及世界状态，供程序回放验证 |
| `outputs/simulation/<场景>/replay.txt` | 可直接阅读的网格、路径和状态变化 |

同一输出目录再次运行会覆盖上次结果。保留不同版本时指定 `--output outputs/simulation/my-run-01`。

## 更多场景和自定义输入

将 `run --scenario` 后的名字替换为以下任意项：

| 场景 | 预期最终状态 | 关键验收点 |
| --- | --- | --- |
| `move-cup` | `completed` | 红杯到架子，手为空，蓝杯不动 |
| `open-box` | `completed` | 打开盒子后取出苹果，放到桌上 |
| `closed-box` | `failed / CONTAINER_CLOSED` | 未开盒不能拿到苹果 |
| `missing-object` | `failed / UNKNOWN_OBJECT` | 不存在的绿色杯子不能被拿起 |
| `blocked-path` | `failed / UNREACHABLE` | 堵住通道后不能穿墙；红杯仍在机器人手里 |
| `unknown-tool` | `failed / UNKNOWN_TOOL` | 未注册的 `teleport` 不执行，初始状态保持不变 |

查看工具参数：

```powershell
.\.venv\Scripts\python.exe -m voice_lab.simulation tools
```

成功演示后，可以复制 `scenario.json` 到 `data/` 并编辑，再用自定义输入运行：

```powershell
Copy-Item outputs/simulation/move-cup/scenario.json data/my-simulation.json
.\.venv\Scripts\python.exe -m voice_lab.simulation run --scenario-file data/my-simulation.json --output outputs/simulation/custom
```

例如，把 `semantic_frame.goal.object_id` 和计划中所有 `object_id` 的 `red_cup` 改为 `blue_cup`，并同步修改 `transcript`，即可测试蓝杯搬运。**只改中文指令不会改变任务**：阶段一使用人工填写的目标和动作，自动理解文字将在阶段二接入。

结构化调用示例：

```python
from voice_lab.simulation import GridWorld

world = GridWorld()
initial = world.reset()
result = world.step({"id": "step-1", "tool": "navigate", "arguments": {"target": "table"}})
assert result.success
observation = world.observe()
```

## 完成标准与下一步需要你做的事

阶段一完成的条件是：自动测试通过、成功案例达到真实状态目标、失败案例保持失败动作之前的状态、日志可重放且能发现不一致。这些是执行层的验收条件；20 条中文指令的语义准确率、多 Agent 协同收益和重规划能力仍需后续阶段评测。

你现在需要做的是：

1. 跑一遍上述成功、失败、回放命令，查看 `replay.txt`，确认“搬杯子/开盒取物”符合毕设演示场景。
2. 若场景需要调整，提供期望的对象、场所、动作和完成条件。下一阶段可以沿用这些接口扩展。
3. 为阶段二准备约 10–20 条你实际会说的中文指令及期望结果，包括歧义指令和不可执行指令。例如“拿那个杯子”应追问颜色，“拿绿色杯子”应说明当前场景没有该物品。

本阶段无需你采集新录音或下载模型。阶段二再让语义 Agent 生成 `SemanticFrame`，由规划层产出 `Plan`，用当前工具执行并记录结果。

## 文件定位

| 文件 | 职责 |
| --- | --- |
| `voice_lab/simulation/schemas.py` | 严格结构与状态一致性检查 |
| `voice_lab/simulation/world.py` | 场景、路径搜索、动作前置条件和原子状态转移 |
| `voice_lab/simulation/tools.py` | 工具白名单与参数描述 |
| `voice_lab/simulation/runner.py` | 执行、停止、最终目标判断、保存和回放 |
| `voice_lab/simulation/scenarios.py` | 六个人工场景 |
| `voice_lab/simulation/__main__.py` | 命令行入口 |
| `tests/test_simulation.py` | 仿真与 CLI 的行为测试 |
