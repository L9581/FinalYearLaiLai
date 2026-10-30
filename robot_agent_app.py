"""Streamlit control panel for the text-first simulated robot Agent.

This page deliberately uses the existing ``voice_lab.robot_agent`` controller;
the UI is only a view and a human control surface. It does not add robot SDK
calls or bypass the controller's action validation.
"""

from datetime import datetime
import json
from pathlib import Path
import tomllib

import streamlit as st

from voice_lab.robot_agent.backend import SimulatedBackend
from voice_lab.robot_agent.controller import TaskController
from voice_lab.robot_agent.model import DemoPlanner, LLMPlanner, OllamaJSON, TaskParser
from voice_lab.robot_agent.schemas import LOCATIONS


ROOT = Path(__file__).resolve().parent


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def reset_task():
    # A finished/paused controller is retained in the log view until this button
    # is pressed. The backend is recreated for every new task.
    for key in ("controller", "parser", "parse_result", "task_text", "clarification", "last_error",
                "clarification_answer", "pending_settings", "original_text", "output_dir"):
        st.session_state.pop(key, None)


def load_llm_config(path):
    with Path(path).open("rb") as file:
        config = tomllib.load(file)["llm"]
    return config


def begin_task(text, *, mode, start, delay, fail_next, config_path, model_name, followup=False):
    if followup:
        # A clarification belongs to the original model, simulation settings,
        # and task. Changing widgets must not silently change that task.
        settings = st.session_state.pending_settings
        mode, start, delay, fail_next, config_path, model_name = (
            settings[key] for key in ("mode", "start", "delay", "fail_next", "config_path", "model_name"))
    else:
        reset_task()
        st.session_state.original_text = text
        st.session_state.pending_settings = dict(mode=mode, start=start, delay=delay, fail_next=fail_next,
                                                 config_path=config_path, model_name=model_name)
    st.session_state.last_error = None
    if mode == "离线演示（规则 Agent）":
        parser = None
        parsed = {"status": "ready", "task": {
            "task_type": "inspect_locations", "targets": ["point_a", "point_b"],
            "constraints": {"visit_in_order": True, "return_to_start": True},
            "completion_rules": {"all_targets_inspected": True, "at_start": True},
        }, "clarification_question": None}
        planner = DemoPlanner()
    else:
        config = load_llm_config(config_path)
        if model_name.strip():
            config["model"] = model_name.strip()
        client = OllamaJSON(config)
        parser = st.session_state.parser if followup else TaskParser(client)
        parsed = parser.parse(text, source="text")
        st.session_state.parser = parser
        if parsed["status"] != "ready":
            st.session_state.parse_result = parsed
            st.session_state.clarification = parsed.get("clarification_question")
            return
        planner = LLMPlanner(client)

    output = ROOT / "outputs" / "robot-agent" / "ui" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backend = SimulatedBackend(start=start, delay=float(delay))
    if fail_next:
        backend.fail_next(count=int(fail_next))
    controller = TaskController(backend, planner, output=output)
    original_text = st.session_state.original_text
    controller.start(parsed["task"], original_text=original_text,
                     parse_turns=parser.turns if parser else [])
    st.session_state.update(controller=controller, parse_result=parsed, task_text=original_text,
                            clarification=None, last_error=None, output_dir=str(output))


def render_runtime():
    controller = st.session_state.get("controller")
    if controller is None:
        return
    snapshot = controller.snapshot()
    state = snapshot["state"]
    labels = {"idle": "未开始", "running": "运行中", "pausing": "正在暂停",
              "paused": "已暂停", "completed": "已完成"}
    st.subheader("执行控制")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("任务状态", labels.get(state, state))
    col2.metric("当前位置", snapshot["observation"]["current_location"])
    col3.metric("执行轮次", snapshot["round"])
    col4.metric("已完成目标", f"{len(snapshot['progress'].get('done_targets', []))}/{len(snapshot['task']['targets'])}")

    if state == "running":
        if st.button("暂停任务", type="secondary", key="pause_task"):
            controller.pause("人工暂停")
            st.rerun(scope="fragment")
    elif state == "paused":
        if st.button("恢复任务", type="primary", key="resume_task"):
            try:
                controller.resume()
            except RuntimeError as exc:
                st.session_state.last_error = str(exc)
            st.rerun(scope="fragment")
    if state in ("paused", "completed"):
        if st.button("开始新任务", key="new_task"):
            reset_task()
            st.rerun(scope="app")

    st.progress(min(1.0, len(snapshot["progress"].get("done_targets", [])) /
                      max(1, len(snapshot["task"]["targets"]))))
    if snapshot.get("reason"):
        st.info(f"暂停原因：{snapshot['reason']}")
    if st.session_state.get("last_error"):
        st.error(st.session_state.last_error)

    with st.expander("当前结构化任务", expanded=False):
        st.json(snapshot["task"])
    with st.expander("初步计划", expanded=True):
        if snapshot["initial_plan"]:
            st.write(" → ".join(snapshot["initial_plan"]))
        else:
            st.caption("等待 Agent 生成计划")
    events = controller.events_since(0)
    with st.expander(f"执行日志（{len(events)} 条）", expanded=True):
        for event in events[-40:]:
            kind = event["kind"]
            if kind == "thinking":
                st.markdown(f"**第 {event['round']} 轮 Thinking**：{event['decision_summary']}")
            elif kind == "action_started":
                st.code("Action：" + pretty(event["action"]), language="json")
            elif kind == "observation":
                result = event["result"]
                st.markdown(f"Observation：`{result['status']}`，位置 `{event['observation']['current_location']}`，"
                            f"已完成 `{event['progress']['done_targets']}`")
                if result.get("error"):
                    st.warning(str(result["error"]))
            elif kind in {"validation_error", "execution_error"}:
                st.warning(f"{kind}：{event.get('observation', event.get('error'))}")
            elif kind in {"paused", "pausing", "resumed", "completed"}:
                st.info(f"{kind}：{event.get('reason', '状态已更新')}")


@st.fragment(run_every="1s")
def runtime_fragment():
    render_runtime()


def main():
    st.set_page_config(page_title="模拟巡检 Agent", page_icon="🤖", layout="wide")
    st.title("模拟巡检 Agent")
    st.caption("自然语言结构化 → Agent 逐轮规划 → 模拟动作 → Observation → 人工控制")

    controller = st.session_state.get("controller")
    if controller is None:
        st.subheader("开始一个模拟任务")
        mode = st.radio("规划模式", ["离线演示（规则 Agent）", "Ollama LLM Agent"], horizontal=True,
                        help="离线演示不调用模型；Ollama 模式使用 config.toml 中的本地模型。")
        if mode == "离线演示（规则 Agent）":
            st.info("离线演示固定执行“依次检查 A、B 并返回”，不会理解或使用下方文字。测试自然语言请切换到 Ollama LLM Agent。")
        else:
            st.caption("支持基地、A 点和 B 点。说“在附近转一圈检查检查”时，会先询问巡检地点。")
        task_text = st.text_area("自然语言任务", value="依次检查 A、B，再回来", height=90,
                                 help="当前支持：前往地点、单点或多点检查、按顺序执行、返回出发点。")
        col1, col2, col3 = st.columns(3)
        start = col1.selectbox("任务出发点", LOCATIONS, index=0,
                               format_func={"home": "home（基地）", "point_a": "point_a（A点）", "point_b": "point_b（B点）"}.get)
        delay = col2.number_input("每个动作延时（秒）", min_value=0.0, max_value=30.0, value=0.5, step=0.5)
        fail_next = col3.number_input("注入移动失败次数", min_value=0, max_value=10, value=0, step=1,
                                      help="用于验证错误反馈和重试。")
        config_path = str(ROOT / "config.toml")
        model_name = ""
        if mode == "Ollama LLM Agent":
            config_path = st.text_input("配置文件", value=config_path)
            model_name = st.text_input("模型名（留空使用配置）", value="")
        if st.button("开始任务", type="primary", disabled=not task_text.strip()):
            try:
                begin_task(task_text.strip(), mode=mode, start=start, delay=delay, fail_next=fail_next,
                           config_path=config_path, model_name=model_name)
            except Exception as exc:
                st.session_state.last_error = str(exc)
            st.rerun()
        if st.session_state.get("parse_result") and not st.session_state.get("controller"):
            parsed = st.session_state.parse_result
            if parsed["status"] == "needs_clarification":
                st.warning(parsed["clarification_question"])
                st.caption("在下方补充原任务的信息，点击“提交补充”。若要换一个任务，请修改上方指令后点击“开始任务”。")
                answer = st.text_input("补充信息", key="clarification_answer")
                if st.button("提交补充", disabled=not answer.strip()):
                    try:
                        begin_task(answer.strip(), mode="Ollama LLM Agent", start=start, delay=delay,
                                   fail_next=fail_next, config_path=config_path, model_name=model_name, followup=True)
                    except Exception as exc:
                        st.session_state.last_error = str(exc)
                    st.rerun()
            elif parsed["status"] == "unsupported":
                st.error(parsed["clarification_question"])
        if st.session_state.get("last_error"):
            st.error(st.session_state.last_error)
    else:
        runtime_fragment()


if __name__ == "__main__":
    main()
