from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
from threading import Event, Thread
from time import monotonic, sleep

import pytest

from voice_lab.robot_agent.backend import SimulatedBackend
from voice_lab.robot_agent.controller import TaskController
from voice_lab.robot_agent.model import DemoPlanner, OllamaJSON, TaskParser
from voice_lab.robot_agent.grounding import ground_parse
from voice_lab.robot_agent.schemas import SchemaError, read_json, validate_parse, validate_task


def task(kind="inspect_locations", targets=("point_a",), ordered=False, returns=False):
    return validate_task({"task_type": kind, "targets": list(targets),
                          "constraints": {"visit_in_order": ordered, "return_to_start": returns}})


def action(name, **arguments):
    return {"name": name, "arguments": arguments}


def decision(value):
    return {"decision_summary": "根据观测选择动作", "remaining_plan": [], "next_action": value, "request_pause": None}


def eventually(predicate, seconds=2):
    end = monotonic() + seconds
    while monotonic() < end:
        if predicate():
            return
        sleep(0.005)
    assert predicate(), "等待预期状态超时"


class FakeClient:
    def __init__(self, *results):
        self.results = iter(results)
        self.messages = []

    def generate(self, messages, schema):
        self.messages.append(deepcopy(messages))
        value = next(self.results)
        if isinstance(value, Exception):
            raise value
        return value


class RecordingPlanner(DemoPlanner):
    def __init__(self):
        self.contexts = []

    def decide(self, context):
        self.contexts.append(deepcopy(context))
        return super().decide(context)


class ConstantPlanner(DemoPlanner):
    def __init__(self, value):
        self.value = value

    def decide(self, context):
        return self.value


@pytest.mark.parametrize("mutate", [
    lambda t: t.update(task_type="dance"),
    lambda t: t.update(targets=[]),
    lambda t: t.update(targets=["point_c"]),
    lambda t: t.update(targets=["point_a", "point_a"]),
    lambda t: t["constraints"].update(return_to_start="true"),
    lambda t: t["completion_rules"].update(all_targets_inspected=False),
    lambda t: t["completion_rules"].update(at_start=False),
    lambda t: t.update(allow_unsafe=True),
    lambda t: t.update(task_type="navigate", targets=["point_a", "point_b"]),
])
def test_task_boundary_rejects_invalid_or_weakened_goal(mutate):
    data = task(returns=True)
    mutate(data)
    with pytest.raises(SchemaError):
        validate_task(data)


def test_missing_completion_rules_are_derived_and_json_duplicates_rejected():
    data = task("navigate", returns=True)
    del data["completion_rules"]
    assert validate_task(data)["completion_rules"] == {"all_targets_reached": True, "at_start": True}
    for value in ('{"a":1,"a":2}', '{"a":NaN}', '```json\n{}\n```'):
        with pytest.raises(SchemaError):
            read_json(value)


@pytest.mark.parametrize("status", ["needs_clarification", "unsupported"])
def test_unready_response_cannot_contain_executable_task(status):
    with pytest.raises(SchemaError):
        validate_parse({"status": status, "task": task(), "clarification_question": "去哪里？"})


def test_parser_repair_is_bounded_and_clarification_retains_original_request():
    clarify = {"status": "needs_clarification", "task": None, "clarification_question": "请提供检查地点。"}
    ready = {"status": "ready", "task": task(returns=True), "clarification_question": None}
    client = FakeClient({"wrong": 1}, clarify, ready)
    parser = TaskParser(client)
    assert parser.parse("去检查一下，结束后回来")["status"] == "needs_clarification"
    assert parser.parse("A 点")["task"]["constraints"]["return_to_start"]
    assert any(m["content"] == "去检查一下，结束后回来" for m in client.messages[-1])
    assert parser.pending == []
    assert len(parser.turns[0]["errors"]) == 1
    bad = FakeClient({}, {})
    with pytest.raises(SchemaError, match="修正后仍"):
        TaskParser(bad).parse("去 A 点")
    assert len(bad.messages) == 2


def test_ollama_transport_requests_structured_output(monkeypatch):
    payloads = []

    def respond(request, timeout):
        payloads.append(json.loads(request.data))
        return BytesIO(json.dumps({"message": {"content": '{"ok":true}'}}).encode())

    monkeypatch.setattr("voice_lab.robot_agent.model.urlopen", respond)
    client = OllamaJSON({"base_url": "http://localhost:11434", "model": "example"})
    schema = {"type": "object"}
    assert client.generate([], schema) == {"ok": True}
    assert payloads[0]["format"] == schema
    assert payloads[0]["options"]["num_predict"] > 256


@pytest.mark.parametrize("utterance,status", [("去检查一下", "needs_clarification"),
    ("去 C 点", "needs_clarification"), ("去 A 点拍照", "unsupported"),
    ("先检查A点，再把杯子带回", "unsupported"), ("不要去 A 点", "needs_clarification")])
def test_hallucinated_ready_task_is_grounded_in_user_input(utterance, status):
    ready = {"status": "ready", "task": task(), "clarification_question": None}
    parsed = TaskParser(FakeClient(ready)).parse(utterance)
    assert parsed["status"] == status and parsed["task"] is None


@pytest.mark.parametrize("utterance,ordered,returns", [
    ("检查 A 点后回到起点", False, True), ("依次检查 A 和 B，不用回来", True, False),
    ("检查 A，回到本次出发点", False, True), ("回基地", False, False),
    ("去 A 点，然后回来", True, True), ("先检查 A 和 B，顺序不限", False, False),
])
def test_explicit_constraints_are_preserved_even_if_model_flags_are_wrong(utterance, ordered, returns):
    ready = {"status": "ready", "task": task(targets=("home",) if utterance == "回基地" else ("point_a",)),
             "clarification_question": None}
    result = ground_parse(ready, [utterance])
    assert result["task"]["constraints"] == {"visit_in_order": ordered, "return_to_start": returns}
    assert result["task"]["completion_rules"]["at_start"] == returns


@pytest.mark.parametrize("utterance", ["在附近转一圈检查检查", "去周边巡检一下", "在周围巡查一遍", "检查 A 点附近"])
@pytest.mark.parametrize("model_status", ["ready", "needs_clarification", "unsupported"])
def test_nearby_region_needs_explicit_scope_despite_model_guess(utterance, model_status):
    raw = {"status": model_status, "task": task() if model_status == "ready" else None,
           "clarification_question": None if model_status == "ready" else "选择随机巡检吗？"}
    parser = TaskParser(FakeClient(raw))
    result = parser.parse(utterance)
    assert result["status"] == "needs_clarification" and result["task"] is None
    assert "还没有定义“附近”的范围" in result["clarification_question"]
    assert "A 点" in result["clarification_question"] and "随机" not in result["clarification_question"]
    assert parser.pending[0]["content"] == utterance


def test_nearby_clarification_uses_user_answer_only_not_suggested_places():
    clarify = {"status": "needs_clarification", "task": None, "clarification_question": "去哪里？"}
    ready = {"status": "ready", "task": task(targets=("point_b",)), "clarification_question": None}
    parser = TaskParser(FakeClient(clarify, clarify, ready))
    parser.parse("在周围巡查一遍，完成后回来")
    assert parser.parse("就附近")["status"] == "needs_clarification"
    result = parser.parse("B 点")
    assert result["task"] == task(targets=("point_b",), returns=True)
    assert parser.pending == []
    assert len(parser.turns) == 3


def test_nearby_request_does_not_mask_unsupported_actions():
    ready = {"status": "ready", "task": task(), "clarification_question": None}
    result = TaskParser(FakeClient(ready)).parse("在附近转一圈检查检查，顺便拍照")
    assert result["status"] == "unsupported" and result["task"] is None


def test_actions_check_position_failure_and_actual_result():
    backend = SimulatedBackend(delay=0, inspection_results={"point_a": "abnormal"})
    execute = lambda a: backend.execute(a, task_id="task-1", cancel=Event())
    assert execute(action("inspect", location="point_a"))["error"]["code"] == "wrong_location"
    backend.fail_next()
    assert execute(action("navigate_to", target="point_a"))["status"] == "failed"
    assert backend.get_status()["current_location"] == "home"
    assert execute(action("navigate_to", target="point_a"))["status"] == "succeeded"
    result = execute(action("inspect", location="point_a"))
    assert result["data"]["inspection"]["result"] == "abnormal"
    assert result["data"]["inspection"]["task_id"] == "task-1"
    assert execute(action("shell", command="bad"))["status"] == "failed"
    assert len(backend.get_status()["inspection_records"]) == 1


def test_stop_acknowledges_cancellation_before_arrival():
    backend = SimulatedBackend(delay=2)
    results = []
    worker = Thread(target=lambda: results.append(backend.execute(action("navigate_to", target="point_a"), task_id="t", cancel=Event())))
    worker.start()
    eventually(lambda: backend.get_status()["motion_state"] == "moving")
    assert backend.get_status()["current_location"] == "home"
    assert backend.stop()["status"] == "succeeded"
    worker.join(1)
    assert results[0]["status"] == "cancelled"
    assert backend.get_status()["active_action"] is None
    assert backend.get_status()["current_location"] == "home"


@pytest.mark.parametrize("spec,start,final,records", [
    (task("navigate"), "home", "point_a", []),
    (task(), "home", "point_a", ["point_a"]),
    (task(targets=("point_a", "point_b"), ordered=True), "home", "point_b", ["point_a", "point_b"]),
    (task(targets=("point_a", "point_b"), ordered=True, returns=True), "home", "home", ["point_a", "point_b"]),
    (task(returns=True), "point_b", "point_b", ["point_a"]),
    (task("navigate", returns=True), "point_b", "point_b", []),
])
def test_acceptance_tasks_use_fresh_feedback_and_program_completion(spec, start, final, records, tmp_path):
    planner = RecordingPlanner()
    controller = TaskController(SimulatedBackend(start, delay=0), planner, output=tmp_path)
    controller.start(spec, original_text="test")
    result = controller.wait(3)
    assert result["state"] == "completed"
    assert result["observation"]["current_location"] == final
    assert [r["location"] for r in result["progress"]["inspection_records"]] == records
    actions = [e for e in controller.events if e["kind"] == "action_started"]
    observations = [e for e in controller.events if e["kind"] == "observation"]
    assert len(actions) == len(observations) == len(planner.contexts)
    for index in range(1, len(planner.contexts)):
        assert planner.contexts[index]["observation"] == observations[index - 1]["observation"]
        assert planner.contexts[index]["history"][-1] == observations[index - 1]["result"]
    assert json.loads((tmp_path / "snapshot.json").read_text(encoding="utf-8"))["state"] == "completed"
    saved = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert saved == controller.events


def test_single_failure_is_in_next_decision_and_retried():
    backend, planner = SimulatedBackend(delay=0), RecordingPlanner()
    backend.fail_next()
    controller = TaskController(backend, planner)
    controller.start(task("navigate"))
    assert controller.wait(2)["state"] == "completed"
    assert planner.contexts[1]["history"][-1]["error"]["code"] == "injected_failure"
    assert len(planner.contexts) == 2


def test_continuous_failures_pause_at_retry_budget():
    backend = SimulatedBackend(delay=0)
    backend.fail_next(count=10)
    controller = TaskController(backend, DemoPlanner(), max_retries=2)
    controller.start(task())
    result = controller.wait(2)
    assert result["state"] == "paused" and "重试" in result["reason"]
    assert len(controller.history) == 3
    assert not result["progress"]["completed"]


def test_pause_during_motion_then_resume_keeps_start_and_progress():
    backend = SimulatedBackend("point_b", delay=1)
    controller = TaskController(backend, DemoPlanner())
    controller.start(task(returns=True))
    eventually(lambda: backend.get_status()["motion_state"] == "moving")
    controller.pause()
    result = controller.wait(2)
    assert result["state"] == "paused"
    assert controller.history[-1]["status"] == "cancelled"
    assert result["observation"]["current_location"] == "point_b"
    assert len([e for e in controller.events if e["kind"] == "action_started"]) == 1
    backend.delay = 0
    controller.resume()
    result = controller.wait(2)
    assert result["state"] == "completed"
    assert result["start_location"] == result["observation"]["current_location"] == "point_b"
    assert len(result["progress"]["inspection_records"]) == 1


@pytest.mark.parametrize("phase", ["plan", "decide"])
def test_pause_during_model_call_ignores_late_result_even_after_resume(phase):
    entered, release, returned = Event(), Event(), Event()

    class SlowPlanner(DemoPlanner):
        def __init__(self):
            self.block_once = True

        def blocked(self, method, context):
            if self.block_once:
                self.block_once = False
                entered.set()
                release.wait(3)
                returned.set()
            return getattr(super(), method)(context)

        def plan(self, context):
            return self.blocked("plan", context) if phase == "plan" else super().plan(context)

        def decide(self, context):
            return self.blocked("decide", context) if phase == "decide" else super().decide(context)

    controller = TaskController(SimulatedBackend(delay=0), SlowPlanner())
    try:
        controller.start(task("navigate"))
        assert entered.wait(2)
        started = monotonic()
        controller.pause()
        assert controller.wait(1)["state"] == "paused"
        assert monotonic() - started < 1
        assert not any(e["kind"] == "action_started" for e in controller.events)
        controller.resume()
        assert controller.wait(2)["state"] == "completed"
        count = len(controller.events)
        release.set()
        assert returned.wait(1)
        assert len(controller.events) == count
        assert len(controller.history) == 1
    finally:
        release.set()


def test_pause_between_actions_resume_does_not_repeat_inspection():
    entered, release = Event(), Event()

    class BetweenPlanner(DemoPlanner):
        def decide(self, context):
            if context["progress"]["done_targets"] == ["point_a"] and not entered.is_set():
                entered.set()
                release.wait(2)
            return super().decide(context)

    controller = TaskController(SimulatedBackend(delay=0), BetweenPlanner())
    try:
        controller.start(task(targets=("point_a", "point_b"), ordered=True, returns=True))
        assert entered.wait(2)
        controller.pause()
        result = controller.wait(1)
        assert result["state"] == "paused"
        assert result["progress"]["done_targets"] == ["point_a"]
        controller.resume()
        result = controller.wait(2)
        assert result["state"] == "completed"
        assert [r["location"] for r in result["progress"]["inspection_records"]] == ["point_a", "point_b"]
    finally:
        release.set()


@pytest.mark.parametrize("value", [
    decision(None),
    decision(action("shell", command="hello")),
    decision(action("navigate_to", target="point_b")),
    decision(action("inspect", location="point_b")),
    {"bad": "output"},
])
def test_invalid_decisions_never_complete_or_dispatch(value):
    controller = TaskController(SimulatedBackend(delay=0), ConstantPlanner(value))
    controller.start(task())
    result = controller.wait(2)
    assert result["state"] == "paused"
    assert not result["progress"]["completed"]
    assert not any(e["kind"] == "action_started" for e in controller.events)


def test_wrong_inspection_order_is_rejected_before_backend():
    controller = TaskController(SimulatedBackend("point_b", delay=0), ConstantPlanner(decision(action("inspect", location="point_b"))))
    controller.start(task(targets=("point_a", "point_b"), ordered=True))
    assert controller.wait(2)["state"] == "paused"
    assert not controller.backend.get_status()["inspection_records"]
    assert "point_a" in controller.history[0]["error"]["message"]


@pytest.mark.parametrize("max_rounds,max_no_progress,rounds,reason", [(20, 3, 3, "没有任务进展"), (2, 6, 2, "最大轮数")])
def test_unproductive_status_polling_stops(max_rounds, max_no_progress, rounds, reason):
    controller = TaskController(SimulatedBackend(delay=0), ConstantPlanner(decision(action("get_status"))),
                                max_rounds=max_rounds, max_no_progress=max_no_progress)
    controller.start(task())
    result = controller.wait(2)
    assert result["state"] == "paused" and reason in result["reason"]
    assert result["round"] == rounds


def test_action_timeout_queries_state_then_pauses():
    controller = TaskController(SimulatedBackend(delay=1), DemoPlanner(), action_timeout=0.02)
    controller.start(task())
    result = controller.wait(2)
    assert result["state"] == "paused" and "超时" in result["reason"]
    assert result["observation"]["current_location"] == "home"
    assert any(e["kind"] == "timeout_state_checked" for e in controller.events)
    assert len(controller.history) == 1


def test_model_timeout_pauses_without_actions():
    release = Event()

    class HangingPlanner(DemoPlanner):
        def decide(self, context):
            release.wait(2)
            return super().decide(context)

    controller = TaskController(SimulatedBackend(delay=0), HangingPlanner(), model_timeout=0.03)
    try:
        controller.start(task())
        result = controller.wait(1)
        assert result["state"] == "paused" and "模型请求超时" in result["reason"]
        assert not controller.history
    finally:
        release.set()


def test_prior_task_inspections_do_not_satisfy_new_task():
    backend = SimulatedBackend("point_a", delay=0)
    backend.execute(action("inspect", location="point_a"), task_id="old", cancel=Event())
    controller = TaskController(backend, DemoPlanner())
    controller.start(task())
    result = controller.wait(2)
    assert result["state"] == "completed"
    assert len(result["observation"]["inspection_records"]) == 2
    assert len(result["progress"]["inspection_records"]) == 1
    assert result["progress"]["inspection_records"][0]["task_id"] == result["task_id"]


def test_asr_bridge_reuses_transcription_without_chat_or_tts(tmp_path, monkeypatch):
    from voice_lab.robot_agent.__main__ import transcribe_wav
    from voice_lab.audio import Audio, wav_bytes
    import numpy as np

    class ASR:
        def transcribe(self, audio):
            assert audio.sample_rate == 16000
            return "去 A 点"

    monkeypatch.setattr("voice_lab.backends.create_asr", lambda _: ASR())
    wav = tmp_path / "input.wav"
    wav.write_bytes(wav_bytes(Audio(np.ones(1600, dtype=np.float32) * 0.1, 16000)))
    assert transcribe_wav(wav, Path(__file__).parents[1] / "config.toml") == "去 A 点"


def test_interactive_cli_pauses_resumes_and_exits_with_open_stdin(tmp_path):
    import subprocess
    import sys
    from threading import Timer

    process = subprocess.Popen([sys.executable, "-m", "voice_lab.robot_agent", "demo", "--delay", "0.1",
                                "--output", str(tmp_path / "cli")], cwd=Path(__file__).parents[1],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8")
    timer = Timer(10, process.kill)
    timer.start()
    lines, paused, resumed = [], False, False
    try:
        for line in process.stdout:
            lines.append(line)
            if not paused and "  Action" in line:
                process.stdin.write("p\n")
                process.stdin.flush()
                paused = True
            if '"kind": "paused"' in line:
                process.stdin.write("r\n")
                process.stdin.flush()
                resumed = True
        process.wait(timeout=2)
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
        process.wait(timeout=2)
        process.stdin.close()
        process.stdout.close()
    assert process.returncode == 0, "".join(lines[-5:])
    assert paused and resumed
    saved = json.loads((tmp_path / "cli" / "snapshot.json").read_text(encoding="utf-8"))
    assert saved["state"] == "completed"
