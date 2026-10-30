from pathlib import Path
from copy import deepcopy
from uuid import uuid4

from streamlit.testing.v1 import AppTest

from voice_lab.robot_agent.controller import TaskController
from voice_lab.robot_agent.model import DemoPlanner


def fake_model(monkeypatch, tmp_path, *results):
    class Client:
        def __init__(self):
            self.results = iter(results)
            self.messages = []

        def generate(self, messages, schema):
            self.messages.append(deepcopy(messages))
            result = next(self.results)
            if isinstance(result, Exception):
                raise result
            return result

    client = Client()
    monkeypatch.setattr("voice_lab.robot_agent.model.OllamaJSON", lambda _: client)
    monkeypatch.setattr("voice_lab.robot_agent.model.LLMPlanner", lambda _: DemoPlanner())

    def make_controller(*args, **kwargs):
        kwargs["output"] = tmp_path / uuid4().hex
        return TaskController(*args, **kwargs)

    monkeypatch.setattr("voice_lab.robot_agent.controller.TaskController", make_controller)
    return client


def run_app():
    return AppTest.from_file(str(Path(__file__).parents[1] / "robot_agent_app.py"), default_timeout=20).run()


def test_offline_ui_starts_and_renders_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = run_app()
    assert not app.exception
    assert app.title[0].value == "模拟巡检 Agent"
    app.radio[0].set_value("离线演示（规则 Agent）")
    app.number_input[1].set_value(0)
    next(button for button in app.button if button.label == "开始任务").click().run()
    assert not app.exception
    # The controller runs in a worker. AppTest can rerun the fragment to observe it.
    for _ in range(12):
        app.run()
    assert not app.exception
    assert any("完成" in metric.label or "已完成" in metric.label for metric in app.metric)
    assert any("执行日志" in exp.label for exp in app.expander)


def test_offline_ui_pause_and_resume(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = run_app()
    app.radio[0].set_value("离线演示（规则 Agent）")
    app.number_input[1].set_value(0.2)
    next(button for button in app.button if button.label == "开始任务").click().run()
    # Pause is available as soon as the controller enters running; wait for the
    # worker to publish the control surface, then click it.
    for _ in range(10):
        app.run()
        if any(button.label == "暂停任务" for button in app.button):
            break
    pause = next(button for button in app.button if button.label == "暂停任务")
    pause.click().run()
    for _ in range(5):
        app.run()
    assert not app.exception
    assert any("恢复任务" == button.label for button in app.button)
    next(button for button in app.button if button.label == "恢复任务").click().run()
    for _ in range(20):
        app.run()
    assert not app.exception


def test_ui_nearby_clarification_retains_intent_and_original_text(tmp_path, monkeypatch):
    clarify = {"status": "unsupported", "task": None, "clarification_question": "不支持附近巡检"}
    ready = {"status": "ready", "task": {"task_type": "inspect_locations", "targets": ["point_b"],
             "constraints": {"visit_in_order": False, "return_to_start": True}}, "clarification_question": None}
    client = fake_model(monkeypatch, tmp_path, clarify, ready)
    app = run_app()
    app.radio[0].set_value("Ollama LLM Agent").run()
    app.text_area[0].set_value("在附近转一圈检查检查，完成后回来")
    app.number_input[0].set_value(0)
    next(b for b in app.button if b.label == "开始任务").click().run()
    assert not app.exception
    assert "controller" not in app.session_state
    assert any("附近" in w.value and "A 点" in w.value for w in app.warning)
    # Change a widget while clarifying: the original task's settings still apply.
    app.selectbox[0].set_value("point_a").run()
    app.text_input(key="clarification_answer").set_value("B 点").run()
    next(b for b in app.button if b.label == "提交补充").click().run()
    assert not app.exception
    controller = app.session_state["controller"]
    snapshot = controller.wait(3)
    assert snapshot["state"] == "completed"
    assert snapshot["task"]["task_type"] == "inspect_locations"
    assert snapshot["start_location"] == "home"
    assert snapshot["progress"]["done_targets"] == ["point_b"]
    first = controller.events_since(0)[0]
    assert first["original_text"] == "在附近转一圈检查检查，完成后回来"
    assert [t["text"] for t in first["parse_turns"]] == [first["original_text"], "B 点"]
    assert any(m["content"] == first["original_text"] for m in client.messages[-1])
    app.run()
    assert len(client.messages) == 2


def test_new_request_drops_pending_clarification(tmp_path, monkeypatch):
    clarify = {"status": "needs_clarification", "task": None, "clarification_question": "去哪巡检？"}
    ready = {"status": "ready", "task": {"task_type": "navigate", "targets": ["point_a"],
             "constraints": {"visit_in_order": False, "return_to_start": False}}, "clarification_question": None}
    client = fake_model(monkeypatch, tmp_path, clarify, ready)
    app = run_app()
    app.radio[0].set_value("Ollama LLM Agent").run()
    app.text_area[0].set_value("在附近转一圈检查检查，完成后回来")
    app.number_input[0].set_value(0)
    next(b for b in app.button if b.label == "开始任务").click().run()
    app.text_area[0].set_value("去 A 点")
    next(b for b in app.button if b.label == "开始任务").click().run()
    assert not app.exception
    snapshot = app.session_state["controller"].wait(3)
    assert snapshot["state"] == "completed"
    assert not snapshot["task"]["constraints"]["return_to_start"]
    assert [m["content"] for m in client.messages[-1] if m["role"] == "user"] == ["去 A 点"]


def test_parser_failure_is_visible_before_a_controller_exists(tmp_path, monkeypatch):
    fake_model(monkeypatch, tmp_path, RuntimeError("Ollama 连接失败"))
    app = run_app()
    app.radio[0].set_value("Ollama LLM Agent").run()
    next(b for b in app.button if b.label == "开始任务").click().run()
    assert not app.exception
    assert "controller" not in app.session_state
    assert any("Ollama 连接失败" in e.value for e in app.error)
