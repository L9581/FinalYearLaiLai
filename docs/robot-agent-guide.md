# 当前阶段：结构化指令与模拟巡检 Agent

2026-10-10：依据 [现阶段目标与实施计划](现阶段目标与实施计划.md) 实现了第一版可运行闭环。入口是 `python -m voice_lab.robot_agent`。这个模块采用一个 Agent，支持文本和已有 ASR 的 WAV 输入，在三个离散地点上执行模拟动作。

当前链路：自然语言 → Ollama 结构化解析 → 原文约束核对与任务校验 → LLM 初步计划 → 每轮一个决定 → 模拟动作 → 实际状态反馈 → 程序核对完成条件。随时可用控制台暂停、恢复。

旧的 `voice_lab.simulation` 是人工计划的网格实验，与本次巡检任务的接口含义不同。本指南对应新的 `voice_lab.robot_agent`。

## 先运行一个完整任务

在项目根目录的 PowerShell 中执行：

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent run "先去 A 点检查，再去 B 点检查，最后回到出发点"
```

使用 `config.toml` 中的 Ollama 地址和模型，本机验证使用 `qwen2.5:3b`。需要本机 Ollama 服务运行。可用 `--config config.local.toml` 或 `--model 模型名` 覆盖设置。文本与离线演示只使用 Python 标准库，没有新增依赖；新环境至少需要 Python 3.11。

运行时显示结构化结果、初步计划、每轮 Thinking / Action / Observation。Thinking 只展示模型的简短决定说明；实际进度以 Observation 和程序完成检查为准。

## 使用 Streamlit 控制页面

命令行验收通过后，可以启动文字版页面：

```powershell
.\start_robot_agent.ps1
```

或者直接执行：

```powershell
.\.venv\Scripts\python.exe -m streamlit run robot_agent_app.py --server.port 8503
```

浏览器打开 `http://127.0.0.1:8503`。页面默认选择“离线演示（规则 Agent）”，不调用 Ollama，适合先检查界面和暂停恢复。选择 “Ollama LLM Agent” 后，输入配置文件和模型名即可使用真实解析与规划。

页面操作：

1. 输入自然语言任务，选择出发点和模拟动作延时。
2. 点击“开始任务”，查看结构化任务、初步计划和执行日志。
3. 动作运行时点击“暂停任务”；状态变为“正在暂停”后会显示“已暂停”。
4. 点击“恢复任务”，控制器读取最新状态后继续执行。
5. 完成或暂停后点击“开始新任务”清除当前页面任务。

页面每秒刷新状态。日志仍保存到 `outputs/robot-agent/ui/`，格式与命令行相同。页面目前是文字优先版本，麦克风 ASR 接入仍使用命令行 `--wav` 入口，下一步再接入页面录音。

### 口语巡检指令与补充信息（2026-10-19）

“在附近转一圈检查检查”可以表达一次巡检意图，但当前环境只有离散的基地、A 点、B 点，没有距离或“附近”的范围。系统会先询问地点，不会自动猜测一条路线。“转一圈”不自动解释成循环巡逻、圆形移动或返回起点；需要返回时请明确说“回来”。

测试方法：

1. 选择 **Ollama LLM Agent**，将“注入移动失败次数”设为 `0`。
2. 输入 `在附近转一圈检查检查`，点击“开始任务”。应出现巡检地点追问，此时不执行动作。
3. 在“补充信息”输入 `A 点和 B 点`，点击“提交补充”。系统应继续原来的检查任务。
4. 如果要同时测试顺序和返回，在第三步改为输入 `先检查 A 点，再检查 B 点，最后回来`。最终应检查 A、B 并返回本次出发点。

“提交补充”保留原任务、模型和模拟设置；若要换任务，请修改上方指令，再点击“开始任务”。日志保留最初的口语指令和后续补充。离线演示固定执行 A/B 检查示例，不解析输入文字。

命令行也支持同样的澄清流程：

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent run "在附近转一圈检查检查"
```

出现追问后输入具体地点和返回要求。仅重复“就附近”仍会要求明确范围。`检查 A 点附近` 也不会被直接当作 `检查 A 点`；拍照等额外能力仍不支持。

输入以下命令并按回车：

| 命令 | 行为 |
| --- | --- |
| `p` | 人工暂停，立即禁止后续动作调度 |
| `r` | 从已确认停止的状态恢复原任务 |
| `s` | 显示当前状态、任务出发点和检查记录 |
| `q` | 停止并保存进度，然后退出 |
| Ctrl+C | 执行期间暂停；输入/解析阶段取消且不启动动作 |

`pausing` 表示收到暂停但尚未确认后端停止，`paused` 表示已确认静止。文本解析尚未完成时可用 Ctrl+C 取消；`p/r/s/q` 控制入口在任务开始后启用。为方便观察移动中暂停，可加 `--delay 3`。

## 第一步：理解并验证结构化任务

对应代码：`voice_lab/robot_agent/schemas.py`。

| 字段 | 约定 |
| --- | --- |
| `status` | `ready` / `needs_clarification` / `unsupported` |
| `task.task_type` | `navigate`（一个目标）或 `inspect_locations`（1～3 个不同目标） |
| `task.targets` | `home` / `point_a` / `point_b`，不接受其他地点 |
| `constraints.visit_in_order` | 是否必须保持目标顺序 |
| `constraints.return_to_start` | 是否返回本次任务开始时的位置 |
| `completion_rules` | 由程序补齐；如果输入已提供，拒绝不一致或降低标准的条件 |
| `clarification_question` | 可执行时为 `null`；否则说明需要补充的信息或不支持原因 |

`navigate` 的完成条件是到达目标；要求返回时，还必须有本任务到达目标的事实并最终位于出发点。检查任务只统计本任务 ID 下的检查记录，核对目标、顺序和返回条件。

两份完整示例在 `data/agent_task_navigate.json`、`data/agent_task_inspect.json`。先执行：

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent validate data/agent_task_inspect.json
```

把目标改成 `point_c`、把布尔值改成字符串或将 `all_targets_inspected` 改成 `false`，校验应指出字段错误。也可以仅提供 `task` 对象。缺少 `completion_rules` 时程序自动补齐。

## 第二步：只验证自然语言解析

对应代码：`model.py` 的 `TaskParser`、`grounding.py`。

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent parse "先检查 B 点，再检查 A 点，最后回来"
.\.venv\Scripts\python.exe -m voice_lab.robot_agent parse "去检查一下"
```

第二条应追问地点；输入 `A 点` 后应保留“检查”意图。缺少地点不能默认选择基地，`去 C 点` 应澄清，拍照、取物等请求不进入执行。

解析使用 Ollama JSON Schema 输出，再做严格程序校验；格式错误最多修正一次。模型只提取任务类型、地点、约束，完成条件由程序生成。

为处理本阶段有限的中文指令，程序还核对原文中的地点别名及明确的顺序、返回、否定表达。例如 `A/A点/A区/point_a`，`依次/先/再/然后/最后/接着`，`回来/回去/返回出发点/回到起点/返回原地`。这是 **LLM 加有限规则核对** 的实现；不是任意自然语言都能保证正确。具体匹配范围见 `grounding.py`。复杂的否定地点表达会要求用户重新明确目标。

澄清上下文只保留待补充的这项任务，最多六轮。`parse.json` 记录原始文本、输入来源、模型原始结构、修正错误和最终规范结构。

## 第三步：验证模拟接口

对应代码：`backend.py` 的 `ActionBackend` 协议与 `SimulatedBackend`。

| 动作 | 参数 | 行为 |
| --- | --- | --- |
| `navigate_to` | `target` | 延时结束且未取消后才更新位置 |
| `inspect` | `location` | 必须在目标地点；生成带 task/action/record ID 的检查记录 |
| `get_status` | `{}` | 查询实际位置、运动状态、活动动作、检查记录 |
| `stop` | `{}` | 请求取消并等待停止确认，返回统一动作反馈 |

反馈包含 `action_id/action/status/data/error/started_at/ended_at`；状态为 `succeeded/failed/cancelled`。时间戳为 UTC，并带时区；北京时间需要加 8 小时。停止或超时未提交的模拟移动不会算作到达。

无需模型的接口调用：

```python
from threading import Event
from voice_lab.robot_agent.backend import SimulatedBackend

backend = SimulatedBackend(delay=0.2)
for name, arguments in [
    ("navigate_to", {"target": "point_a"}),
    ("inspect", {"location": "point_a"}),
    ("navigate_to", {"target": "home"}),
]:
    result = backend.execute({"name": name, "arguments": arguments},
                             task_id="manual-example", cancel=Event(), timeout=10)
    print(result)
```

也可运行离线示例：

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent demo --non-interactive
```

`demo` 使用人工结构化任务和根据反馈决策的规则替身，专门测试控制流程。真实 LLM 使用 `run`。

## 第四步：观察逐轮执行

对应代码：`model.py` 的 `LLMPlanner`、`controller.py`。

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent run "依次检查 A、B，再回来" --non-interactive
```

初步计划单独调用一次模型。随后每轮只接收一个动作决定，执行后读取真实模拟结果及状态，再传入下一轮。日志保留完整历史；模型请求使用简化后的最近反馈和任务累计进度，以减少过时状态的干扰。

执行前检查动作白名单、参数、地点是否属于本任务、检查顺序、重复检查等。无效决定作为下一轮反馈，不调用后端。模型声称完成也必须通过程序检查；模拟后端已经满足条件时，程序直接结束。

出发点在 `start()` 时保存，`resume()` 不覆盖：

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent run "检查 A 点，完成后回来" --start point_b
```

该任务最后必须回到 `point_b`。

## 第五步：暂停与恢复

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent run "依次检查 A、B，再回来" --delay 3
```

在移动期间输入 `p`，等待出现 `paused`；输入 `s` 查看保留的任务；输入 `r` 继续。恢复先读取最新状态，使用原任务 ID、出发点和已完成检查。

控制器与模型调用使用独立线程及每次运行独有的取消信号。暂停不依赖模型同意。等待模型时暂停后，迟到的响应只留在该调用自己的返回队列，不能调度动作或覆盖恢复后的状态。本地 HTTP 请求不保证能中止服务端推理，但它的结果会被丢弃。

目前恢复支持**同一进程内** `p → r`；`q` 或关闭终端后，日志和快照仍在，但尚未提供从磁盘恢复进程的命令。

## 第六步：失败、超时与停止条件

```powershell
# 注入第一次移动失败，检查下一轮是否使用失败反馈
.\.venv\Scripts\python.exe -m voice_lab.robot_agent run "检查 A 点" --fail-next 1

# 离线、稳定地验证连续失败会暂停
.\.venv\Scripts\python.exe -m voice_lab.robot_agent demo --fail-next 10 --non-interactive

# 观察动作超时后的状态核对和暂停
.\.venv\Scripts\python.exe -m voice_lab.robot_agent demo --delay 2 --action-timeout 0.1 --non-interactive
```

默认同一动作最多重试两次（共三次调用）；模型校验错误最多连续出现三次；六轮没有完成任何目标则暂停；每次连续运行最多二十轮。人工恢复会获得新的重试/轮数预算，但任务累计轮次、记录和出发点保留。

可配置：`--max-retries`、`--max-rounds`、`--max-no-progress`、`--model-timeout`、`--action-timeout`。动作超时后先查询状态并停止，不直接假定动作没有发生。模拟后端取消前未提交的动作不产生移动或检查效果。

检查结果可用 `--abnormal point_a` 设置异常。异常是有效的检查结果，不代表接口调用失败；当前任务要求“完成检查”，不包含排除故障或处理异常。

## 第七步：复现验收与查看日志

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m voice_lab.robot_agent evaluate
.\.venv\Scripts\python.exe -m voice_lab.robot_agent evaluate --samples data/agent_parse_unseen.json
.\.venv\Scripts\python.exe scripts/verify_robot_agent.py --live --repetitions 3
.\.venv\Scripts\python.exe -m pytest tests/test_robot_agent_app.py -q
```

不用真实模型时去掉验收脚本的 `--live`。脚本保存正常执行、移动中暂停、等待模型时暂停、动作之间暂停、恢复、单次和连续失败的独立记录。输出目录若已存在会拒绝覆盖；默认生成新目录。

每个任务保存：

- `parse.json`：控制台解析记录（验收脚本将解析记录放在 `task_started` 事件中）。
- `events.jsonl`：原文、任务、初步计划、逐轮决定、动作、Observation、校验失败、暂停恢复及最终完成检查。
- `snapshot.json`：最新任务、原出发点、当前位置、进度和暂停原因。

默认输出位于 `outputs/robot-agent/`；也可指定一个新的 `--output` 目录。返回码：`0` 完成/解析成功，`1` 已暂停或验收未全通过，`2` 输入需要澄清/不支持/错误，输入解析阶段 Ctrl+C 返回 `130`。

本机实际结果与证据链接见 [模拟 Agent 验收记录](robot-agent-verification.md)。

## 接入已有 ASR

```powershell
.\.venv\Scripts\python.exe -m voice_lab.robot_agent run --wav "你的任务录音.wav" --config config.local.toml
```

该入口复用现有音频预处理与 ASR，转写后直接进入任务解析，日志用 `source=asr` 标记。不会调用聊天回答或 TTS。当前已经测试接线与转写传递；尚未为巡检指令录制真实语音做端到端验收，也未在原 Streamlit 页面增加任务按钮。

## 后续机器狗阶段

新增符合 `ActionBackend` 协议的后端，提供实际状态、可取消动作和可靠停止确认。Agent 与控制器只依赖该协议。当前模拟的位置是离散地点；真实机器狗是否支持按地点导航，需要结合 SDK、定位和地图能力确定。提供设备后，先验证单动作、停止反馈和超时语义，再验证多步任务。
