# 模拟巡检 Agent 验收记录

## 2026-10-19：口语巡检与网页澄清修正

- 原句“在附近转一圈检查检查”原本会得到含糊的随机检查追问。现改为说明只有基地/A/B，并要求选择地点；补充地点后保留巡检意图及原先的返回要求。
- 页面区分新任务与补充答复，澄清期间保留原模型和模拟设置；任务日志保留最初指令；解析错误会直接显示；离线演示明确提示不解析文字。
- 全项目自动测试：**146 passed**，包含范围未明时不执行、补充信息衔接、新任务清除旧上下文及解析错误显示等回归。
- 真实 Ollama qwen2.5:3b：新增 [8 条附近巡检案例](../data/agent_parse_nearby.json) **8/8**，旧开发案例 **20/20**。[新案例结果](../outputs/robot-agent/acceptance/nearby-20261019-v2/evaluation.json)、[旧案例回归](../outputs/robot-agent/acceptance/nearby-regression-20261019/evaluation.json)。首次新增样例测试为 7/8，修正澄清上下文提示后复测全部通过；这组样例属于开发回归集。
- Streamlit AppTest 使用真实 Ollama 完成“在附近转一圈检查检查”→ 追问 → “先检查 A 点，再检查 B 点，最后回来”→ `completed`；最终检查 A/B 并回到 home。[完整任务日志](../outputs/robot-agent/ui/20261019-210935-146796/events.jsonl)、[结果快照](../outputs/robot-agent/acceptance/nearby-ui-20261019.json)。本次未进行人工浏览器操作。

以上仍是有限地点的模拟巡检，没有新增自动探索、距离感知或真实地图导航能力。

## 2026-10-10：初始命令行验收

日期：2026-10-10（Asia/Shanghai）。Python 3.12.8；本机 Ollama `qwen2.5:3b`；动作执行使用 Python 模拟后端。模型输出 temperature=0、上下文 8192 tokens，未训练模型。

## 自动测试

`.\.venv\Scripts\python.exe -m pytest -q`：**127 passed**。

其中本次新增 52 项测试；已有 75 项语音与网格仿真测试通过。新增测试覆盖结构边界与完成条件、澄清修正、原文约束核对、动作效果、停止确认、本次任务记录隔离、顺序、失败重试、假完成拒绝、超时、无进展，以及暂停后迟到响应和恢复竞争。另以真实控制台子进程验证 `p → paused → r → completed`，并检查输入端仍打开时 Python 能正常退出。

## 真实模型解析

| 数据 | 结果 | 记录 |
| --- | --- | --- |
| 20 条开发案例，包含两条多轮澄清 | 20/20 | [开发集输出](../outputs/robot-agent/acceptance/parser-final/evaluation.json) |
| 6 条扩展表述，已用于调试 | 6/6 | [扩展表述输出](../outputs/robot-agent/acceptance/parser-variants-final/evaluation.json) |
| 完成修正后新增的 6 条表述 | 6/6 | [新增表述输出](../outputs/robot-agent/acceptance/parser-unseen/evaluation.json) |

上述结果评估的是 **模型解析 + 程序核对** 的最终结构，不是裸模型准确率。原始模型输出保留在 `turns[].attempts` 中。样例数量小，不代表对任意表述的可靠性。

`data/agent_parse_holdout.json` 是早期命名；其中的失败已被用于改进，因此不再视为独立留出集。`agent_parse_unseen.json` 是最终修改后新增并一次运行的六条输入。

## 执行与人工控制

运行：

```powershell
.\.venv\Scripts\python.exe scripts/verify_robot_agent.py --live --repetitions 3 --output outputs/robot-agent/acceptance/final-runs
```

**22/22 个状态检查通过**，包括 13 个使用规则替身的执行/暂停检查，以及 9 次真实 LLM 闭环任务。[完整报告](../outputs/robot-agent/acceptance/final-runs/report.json)。

| 场景 | 执行来源 | 实际结果 |
| --- | --- | --- |
| 前往 A、检查 A、依次检查 A/B、检查后返回 | 人工结构化任务 + 规则替身 | 完成条件全部通过 |
| 从 B 出发，检查 A 后返回 | 规则替身 + 真实 LLM | 最后位于 B |
| 缺少目标 / 未知目标 | 真实解析流程 | 澄清，不进入动作执行 |
| 第一次移动失败 | 规则替身 + 真实 LLM | 根据失败反馈重试后完成 |
| 连续移动失败 | 规则替身 | 三次失败后暂停，没有假报完成 |
| 移动中暂停 | 规则替身 | 动作取消，位置未提前改变，后续动作不启动 |
| 等待模型回复时暂停 | 可控延迟的模型替身 | 不等待模型返回，迟到响应不执行；恢复后正常完成 |
| 两个动作之间暂停 | 可控延迟的模型替身 | 已完成检查保留，恢复不重复检查 |

真实 LLM 的三类任务各重复三次，全部完成：

Streamlit 页面测试：`tests/test_robot_agent_app.py` **2 passed**，覆盖页面启动、任务日志、暂停和恢复。页面使用离线规则 Agent 测试，因此不依赖 Ollama。

- 检查 A/B 后返回：3/3，每次 7 轮决定、6 次接口调用。其中有 1 次错误移动决定被校验拦截，模型利用下一轮反馈修正；另有 1 次额外状态查询。
- 从 B 出发检查 A 后返回：3/3，每次 3 次接口调用，最终返回 B。
- 注入第一次移动失败后检查 A：3/3，每次 3 次接口调用，第一次失败被保留在下一轮上下文中。

代表记录：

- [真实 LLM 正常任务](../outputs/robot-agent/acceptance/final-runs/live-normal-1/events.jsonl)
- [真实 LLM 返回 B 点](../outputs/robot-agent/acceptance/final-runs/live-return-to-b-1/events.jsonl)
- [真实 LLM 失败后重试](../outputs/robot-agent/acceptance/final-runs/live-single-failure-1/events.jsonl)
- [移动中暂停与恢复](../outputs/robot-agent/acceptance/final-runs/10-motion-pause-resume/events.jsonl)
- [等待模型时暂停与恢复](../outputs/robot-agent/acceptance/final-runs/11-model-pause-resume/events.jsonl)
- [连续失败暂停](../outputs/robot-agent/acceptance/final-runs/09-repeated-failure/events.jsonl)

## 开发中发现的问题与剩余限制

最初的模型提示/结构格式会出现把未知或缺失地点替换成已知地点、把到达当成完成检查、混淆完成要求与当前进度等错误。已通过简化模型输入、程序生成完成条件、原文约束核对及执行前后校验修正。早期失败日志保留在 `outputs/robot-agent/acceptance/llm-normal*` 与 `parser-dev*` 中；最终结果以上面的 `final-runs`、`parser-final` 为准。

暂停的竞争测试使用确定性同步点和模型替身，验证控制器对迟到回复的处理；没有声称人工在真实 Ollama 推理的每个时点都操作过控制台。真实模型仍可能给出不合理决定，因此所有动作持续经过校验、进展和轮数限制。

当前支持进程内暂停恢复，未提供关闭进程后的任务恢复命令。ASR 入口已接好并测试，真实巡检录音、现场麦克风、Streamlit 任务界面、真实机器人均不在本次实测结果中。检查异常只记录结果，不包含排障动作。

`outputs/` 被 Git 忽略，上述证据保存在本机；复制项目到其他电脑时可用验收命令重新生成。
