# 语音交互中间层多 Agent 与仿真验证计划

2026-10-05 进度：阶段一的 `GridWorld`、严格状态结构、五种工具、执行日志和回放已实现，操作与验收见 [阶段一说明](phase-one-simulation.md)。本阶段语义/动作均为人工输入；多 Agent 推理、PDDL、自动重规划与视觉模型按后续阶段继续推进。

## 当前基线

现有程序已经跑通 `录音/WAV → ASR → Ollama → TTS`：

- `voice_lab/pipeline.py` 的 `VoicePipeline` 负责整条语音链路。
- `voice_lab/backends.py` 的 `OllamaLLM` 只返回自然语言，没有结构化意图、工具调用或世界状态。
- `app.py` 只提供语音、WAV 和文字输入，还没有图像输入。
- `total_aim.md` 已经提出 VLM、PDDL、技能调用和执行后重规划，但这些模块尚未落地。

因此新层应放在 ASR 和 TTS 之间，并保留普通问答回退路径：

```text
语音/文字 ─→ ASR ─→ AgentOrchestrator ─→ 最终答复 ─→ TTS
                         │
                         ├─ 语义 Agent
                         ├─ 场景/状态 Agent
                         ├─ 规划 Agent（PDDL）
                         ├─ 技能执行 Agent
                         └─ 监控/批评 Agent
                              │
                              └─ 仿真环境
```

第一阶段的“VLM”可以先接收仿真器提供的结构化场景 JSON；这样先验证任务理解、规划、工具调用和失败恢复。第二阶段再把仿真器截图接入真正的视觉模型。当前 `qwen2.5:3b` 是文本模型，不能直接承担图像理解。

## 推荐的最小闭环

先实现一个 Python 进程内的确定性 GridWorld，不引入 ROS 或大型机器人仿真器。场景可以包含机器人、厨房/客厅、桌子、杯子、苹果和盒子，动作限定为：

`navigate`、`pick`、`place`、`open`、`inspect`。

推荐的第一条演示任务是：

> “把桌上的红色杯子拿到厨房的架子上。”

它同时覆盖对象指代、空间关系、导航、抓取、放置和执行结果检查。仿真器需要提供 `reset()`、`observe()`、`step(action)` 三个接口，并对非法动作返回明确错误，不直接修改隐藏状态。

Agent 之间先用进程内 Python 类和严格 JSON 通信，不要一开始拆成多个服务。多 Agent 的边界由角色、输入输出 schema 和独立日志体现；可以先共用一个 Ollama 模型，后续再按资源情况拆模型。

建议的调用顺序是：

1. 语义 Agent 将转写文本和对话历史转成 `SemanticFrame`，包含 `goal`、对象、约束、置信度和是否需要澄清。
2. 场景 Agent 将 `SemanticFrame` 中的对象与仿真器 `WorldState` 对齐；歧义时返回候选对象和澄清问题。
3. 规划 Agent 根据状态和目标生成 PDDL problem，并用 Python 可安装的规划器先求出动作序列。规划结果必须经过 schema 校验。
4. 技能执行 Agent 只能从白名单工具注册表中调用工具，一次执行一个动作，把 `ToolResult` 和新的观测交给监控 Agent。
5. 监控 Agent 校验动作前置条件和预期效果；失败时更新状态并最多重规划 `N` 次，超过上限则转为语音澄清或失败说明。
6. 对普通问题或无法识别为任务的输入，继续走现有 `OllamaLLM.reply()` 回退路径。

## 需要先固定的接口

建议在 `voice_lab/agents/` 或新的 `agent_lab/` 包中定义以下数据结构，使用 dataclass 或 Pydantic，并拒绝未定义字段：

```text
TurnInput
  transcript: str
  image: optional image reference
  history: list[message]
  observation: WorldState

SemanticFrame
  intent: chat | task | clarification
  goal: symbolic goal or null
  entities: list[entity reference]
  constraints: list[str]
  confidence: float
  clarification_question: str | null

Plan
  actions: list[ActionCall]
  planner: str
  problem_id: str

ActionCall
  id: str
  tool: enum of registered tools
  arguments: typed JSON object
  expected_effects: list[predicate]

ToolResult
  success: bool
  observation: WorldState
  error_code: str | null
  message: str

ExecutionTrace
  turn_id, semantic_frame, plan, action results, replans, final status
```

LLM 只负责提出结构化结果；schema 校验、PDDL 解析、工具白名单、动作前置条件、状态转移和重试上限全部由程序负责。这样可以测试“模型说了什么”和“系统实际执行了什么”之间的差异。

## 分阶段实施

### 阶段 0：冻结基线和选择场景

保留现有语音链路，增加一组不依赖模型的 pipeline 测试作为回归基线。确定一个仿真场景、五个以内的动作和 10–20 条中文任务指令，包括成功、歧义和不可执行三类。

交付物：场景说明、动作/谓词表、任务样例、现有测试通过记录。

### 阶段 1：仿真器、工具注册表和结构化日志

实现确定性 GridWorld、`WorldState`、工具注册表和 `ExecutionTrace`。先用人工构造的 `SemanticFrame` 驱动仿真器，确认 `reset → observe → step → observe` 和非法动作处理正确。

交付物：无 LLM 也能运行的仿真回放，以及针对状态转移的单元测试。

### 阶段 2：语义 Agent 和 PDDL 规划

把现有 `OllamaLLM` 扩展为可请求 JSON 的客户端，加入超时、重试、JSON 解析失败和降级处理。语义 Agent 只生成 `SemanticFrame`；规划 Agent 把它转为 PDDL problem 并产生计划。第一版建议使用 Python 易安装的规划器，等闭环稳定后再比较 Fast Downward。

交付物：同一目标在固定初始状态下能稳定生成合法计划；规划失败时能给出可读原因。

### 阶段 3：执行、监控和重规划

逐动作调用仿真器，注入三类故障：目标物不存在、导航失败、抓取失败。监控 Agent 必须记录失败原因、更新状态并重新规划，达到次数上限后停止，不得声称任务已完成。

交付物：成功率、非法工具调用数、重规划次数、失败原因和完整 trace。

### 阶段 4：接回语音和 TTS

在 `VoicePipeline` 中把 `self.llm.reply()` 替换为 `orchestrator.handle()`；普通聊天返回文本，任务请求返回执行摘要和下一步问题。给 Streamlit 增加任务执行进度和 trace 展示，最后仍由现有 TTS 播报最终摘要。

交付物：语音输入可以触发仿真任务，执行失败时不会生成误导性的“已完成”语音。

### 阶段 5：视觉模型接入（在前四阶段通过后）

让仿真器输出截图或帧，并增加 `VisionObservationAdapter`。视觉模型只负责从图像提取对象、属性和空间关系，仍由符号状态和规划器决定动作。视觉结果必须与结构化场景做一致性检查，不能让 VLM 直接执行任意代码。

交付物：同一任务分别用 JSON 观测和截图观测运行，报告视觉识别错误如何影响规划成功率。

## 后续完整任务闭环的验收指标

以下为阶段二及以后接入模型、规划和重规划后的暂定目标，不是阶段一已测得的结果；阶段一按状态转移、错误处理和回放一致性验收：

- 20 条任务指令中，语义结构化成功率至少 80%。
- 固定初始状态下，计划合法率至少 90%。
- 注入故障后，系统不执行未注册工具，且能在限定次数内重规划。
- 任务完成率、平均动作数、平均响应时间、LLM 调用次数和重规划次数全部写入 JSONL。
- 失败任务必须保存原始转写、结构化结果、计划、工具结果和最终答复，便于写报告。

## 后续阶段需要准备的事项

1. 确认第一条任务场景。建议直接采用“把红色杯子从桌上放到厨房架子上”，暂不扩展到真实机器人动作。
2. 确认先使用轻量 GridWorld；若后续必须展示视觉画面，再选 Webots 或 PyBullet。Windows 环境下不建议第一步就上 Isaac Sim 或 ROS2。
3. 准备 10–20 条中文指令及期望结果，至少包含 3 条歧义指令、3 条不可执行指令和 3 条带数字/位置描述的指令。
4. 运行 `ollama list`，确认现有 `qwen2.5:3b` 可用；如果要做视觉阶段，再另外准备一个支持图像输入的本地 VLM，并记录模型名、显存和接口形式。
5. 确认本阶段是否只在 Windows + Python 内完成。若要使用 Fast Downward、ROS2 或其他 Linux 优先工具，需要提前决定是否接受 WSL2/独立 Linux 环境。
6. 确认评价优先级：先看规划与执行成功率，还是先看 VLM 的视觉识别效果。建议先验收前者。

阶段一已采用 Windows + Python GridWorld 和人工计划实现；使用现有环境即可验收，不需要额外下载模型。接下来准备中文任务样例，再进入阶段二的 Ollama 结构化语义和 PDDL 规划。
