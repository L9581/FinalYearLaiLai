from dataclasses import FrozenInstanceError, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from voice_lab.simulation import ActionCall, Goal, GridWorld, Plan, WorldState, default_world
from voice_lab.simulation.__main__ import main
from voice_lab.simulation.runner import (Scenario, read_json, run_scenario, save_trace, verify_trace)
from voice_lab.simulation.schemas import Position, SchemaError
from voice_lab.simulation.scenarios import make_scenario


def call(world, tool, **arguments):
    return world.step(ActionCall("test-action", tool, arguments))


def test_transfer_changes_real_state_and_preserves_blue_cup():
    world = GridWorld()
    original = world.observe()
    assert call(world, "navigate", target="table").success
    assert call(world, "pick", object_id="red_cup").success
    held = world.observe()
    assert held.holding == "red_cup"
    assert next(o for o in held.objects if o.id == "red_cup").location == "robot"
    navigation = call(world, "navigate", target="shelf")
    assert navigation.success
    assert navigation.path[0] == Position(1, 1)
    assert navigation.path[-1] == Position(5, 1)
    assert Position(3, 2) in navigation.path  # The only doorway.
    assert len(navigation.path) == 7  # Shortest route must go around the wall.
    assert not set(navigation.path).intersection(original.blocked)
    assert all(abs(a.x - b.x) + abs(a.y - b.y) == 1 for a, b in zip(navigation.path, navigation.path[1:]))
    assert call(world, "place", object_id="red_cup", target="shelf").success
    assert Goal("red_cup", "shelf").satisfied_by(world.observe())
    assert Goal("blue_cup", "table").satisfied_by(world.observe())
    assert world.observe().holding is None
    assert world.observe().revision == 4
    # Earlier observations must remain snapshots rather than views of the current world.
    assert Goal("red_cup", "table").satisfied_by(original)
    assert held.holding == "red_cup"


def test_closed_container_requires_open_for_both_pick_and_place():
    world = GridWorld()
    call(world, "navigate", target="box")
    before = world.observe()
    assert call(world, "pick", object_id="apple").error_code == "CONTAINER_CLOSED"
    assert world.observe() == before
    assert call(world, "open", target="box").success
    opened = world.observe()
    assert call(world, "open", target="box").success  # Already open is idempotent.
    assert world.observe() == opened
    assert call(world, "pick", object_id="apple").success
    call(world, "navigate", target="table")
    assert call(world, "place", object_id="apple", target="table").success
    assert Goal("apple", "table").satisfied_by(world.observe())

    world.reset()
    call(world, "navigate", target="table")
    call(world, "pick", object_id="red_cup")
    call(world, "navigate", target="box")
    before = world.observe()
    assert call(world, "place", object_id="red_cup", target="box").error_code == "CONTAINER_CLOSED"
    assert world.observe() == before
    call(world, "open", target="box")
    assert call(world, "place", object_id="red_cup", target="box").success
    assert Goal("red_cup", "box").satisfied_by(world.observe())


@pytest.mark.parametrize("actions,code", [
    ([("pick", {"object_id": "red_cup"})], "NOT_AT_TARGET"),
    ([("pick", {"object_id": "green_cup"})], "UNKNOWN_OBJECT"),
    ([("navigate", {"target": "moon"})], "UNKNOWN_TARGET"),
    ([("place", {"object_id": "red_cup", "target": "shelf"})], "NOT_HOLDING"),
    ([("open", {"target": "table"})], "NOT_CONTAINER"),
    ([("open", {"target": "box"})], "NOT_AT_TARGET"),
    ([("navigate", {"target": "table"}), ("pick", {"object_id": "red_cup"}),
      ("pick", {"object_id": "blue_cup"})], "HAND_FULL"),
    ([("navigate", {"target": "table"}), ("pick", {"object_id": "red_cup"}),
      ("place", {"object_id": "red_cup", "target": "shelf"})], "NOT_AT_TARGET"),
    ([("navigate", {"target": "table"}), ("pick", {"object_id": "red_cup"}),
      ("place", {"object_id": "blue_cup", "target": "table"})], "NOT_HOLDING"),
    ([("teleport", {"target": "shelf"})], "UNKNOWN_TOOL"),
    ([("inspect", {"extra": "ignore me"})], "INVALID_ARGUMENTS"),
    ([("navigate", {})], "INVALID_ARGUMENTS"),
])
def test_rejected_action_is_atomic(actions, code):
    world = GridWorld()
    for tool, args in actions[:-1]:
        assert call(world, tool, **args).success
    before = world.observe()
    tool, args = actions[-1]
    result = call(world, tool, **args)
    assert not result.success
    assert result.error_code == code
    assert result.observation == before == world.observe()
    assert result.path == ()


@pytest.mark.parametrize("payload", [
    None, [], {"id": "x", "tool": "inspect"},
    {"id": "x", "tool": "inspect", "arguments": {}, "extra": True},
    {"id": "x", "tool": "pick", "arguments": {"object_id": 3}},
    {"id": "x", "tool": "pick", "arguments": {"object_id": ""}},
    {"id": "x", "tool": "pick", "arguments": ["red_cup"]},
])
def test_malformed_call_is_rejected_without_changing_state(payload):
    world = GridWorld()
    before = world.observe()
    result = world.step(payload)
    assert not result.success
    assert result.error_code == "INVALID_ACTION"
    assert world.observe() == before


def test_action_arguments_revalidated_after_external_mutation():
    world = GridWorld()
    action = ActionCall("x", "navigate", {"target": "table"})
    action.arguments["target"] = float("nan")
    result = world.step(action)
    assert result.error_code == "INVALID_ACTION"
    assert world.observe() == default_world()


def test_inspect_reset_and_world_isolation():
    world = GridWorld()
    other = GridWorld()
    initial = world.observe()
    with pytest.raises(FrozenInstanceError):
        initial.objects[0].location = "shelf"
    json_copy = initial.to_dict()
    json_copy["objects"][0]["location"] = "shelf"
    assert world.observe() == initial
    call(world, "navigate", target="table")
    changed = world.observe()
    assert call(world, "inspect").observation == changed
    assert call(world, "navigate", target="table").observation == changed
    assert other.observe() == initial
    assert world.reset() == initial
    assert call(world, "navigate", target="table").observation == changed


@pytest.mark.parametrize("field,value", [
    ("width", True), ("height", 0), ("revision", -1),
    ("holding", "red_cup"), ("robot", {"x": 3, "y": 0}),
    ("robot", {"x": 7, "y": 0}), ("robot", {"x": 1.0, "y": 2}),
    ("extra", "not allowed"),
])
def test_world_schema_rejects_inconsistent_state(field, value):
    data = default_world().to_dict()
    data[field] = value
    with pytest.raises(SchemaError):
        WorldState.from_dict(data)


def test_world_schema_checks_nested_fields_and_locations():
    for change in (lambda d: d["objects"][0].update(location="unknown"),
                   lambda d: d["objects"][0].update(id="blue_cup"),
                   lambda d: d["objects"][0].update(hidden=True),
                   lambda d: d["fixtures"][0].update(is_open=False),
                   lambda d: d["fixtures"][0].update(is_open=1)):
        data = default_world().to_dict()
        change(data)
        with pytest.raises(SchemaError):
            WorldState.from_dict(data)
    assert WorldState.from_dict(default_world().to_dict()) == default_world()


@pytest.mark.parametrize("name,status,code,count", [
    ("move-cup", "completed", None, 6),
    ("open-box", "completed", None, 5),
    ("closed-box", "failed", "CONTAINER_CLOSED", 2),
    ("missing-object", "failed", "UNKNOWN_OBJECT", 2),
    ("blocked-path", "failed", "UNREACHABLE", 4),
    ("unknown-tool", "failed", "UNKNOWN_TOOL", 1),
])
def test_scenarios_check_goal_stop_on_failure_and_replay(tmp_path, name, status, code, count):
    scenario = make_scenario(name)
    trace = run_scenario(scenario)
    assert trace.status == status
    assert trace.goal_satisfied == (status == "completed")
    assert len(trace.events) == count
    assert trace.events[-1].result.error_code == code
    if code:
        assert trace.events[-1].before == trace.final_state
    if name == "blocked-path":
        assert trace.final_state.holding == "red_cup"
        assert trace.final_state.robot == Position(1, 1)
    assert run_scenario(scenario, execution_id=trace.execution_id) == trace
    save_trace(trace, tmp_path)
    assert verify_trace(read_json(tmp_path / "trace.json")) == trace
    assert Scenario.from_dict(read_json(tmp_path / "scenario.json")) == scenario
    assert f"status={status}" in (tmp_path / "replay.txt").read_text(encoding="utf-8")


def test_successful_actions_alone_do_not_mean_task_completed():
    scenario = make_scenario()
    empty = replace(scenario, plan=Plan(()))
    assert run_scenario(empty).status == "incomplete"
    short = replace(scenario, plan=Plan(scenario.plan.actions[:3]))
    trace = run_scenario(short)
    assert all(e.result.success for e in trace.events)
    assert trace.status == "incomplete"
    assert not trace.goal_satisfied
    assert trace.final_state.holding == "red_cup"
    already_done = replace(empty, semantic_frame=replace(empty.semantic_frame, goal=Goal("red_cup", "table")))
    assert run_scenario(already_done).status == "completed"


@pytest.mark.parametrize("part", ["state", "route", "result", "status", "missing_event", "extra_event", "bool_as_int"])
def test_replay_detects_inconsistent_trace(part):
    data = run_scenario(make_scenario()).to_dict()
    if part == "state":
        data["events"][2]["result"]["observation"]["holding"] = None
    elif part == "route":
        data["events"][3]["result"]["path"] = []
    elif part == "result":
        data["events"][0]["result"]["message"] = "fake"
    elif part == "status":
        data["status"] = "failed"
    elif part == "missing_event":
        data["events"].pop()
    elif part == "extra_event":
        data["events"].append(data["events"][-1])
    else:
        data["goal_satisfied"] = 1
    with pytest.raises(SchemaError, match="回放不一致"):
        verify_trace(data)


def test_scenario_schema_rejects_extra_fields_duplicate_steps_and_fake_model_source():
    scenario = make_scenario()
    for change in (lambda d: d.update(extra=True),
                   lambda d: d["plan"]["actions"].append(d["plan"]["actions"][0]),
                   lambda d: d["semantic_frame"].update(source="vlm"),
                   lambda d: d["semantic_frame"]["goal"].update(extra=True)):
        data = scenario.to_dict()
        change(data)
        with pytest.raises(SchemaError):
            Scenario.from_dict(data)


@pytest.mark.parametrize("raw", ['{"id":"a","id":"b"}', '{"value":NaN}'])
def test_non_strict_json_is_rejected(tmp_path, raw):
    path = tmp_path / "bad.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(SchemaError):
        read_json(path)


def test_cli_saved_case_custom_input_and_failure_exit_codes(tmp_path, capsys):
    success = tmp_path / "success"
    failure = tmp_path / "failure"
    assert main(["run", "--output", str(success)]) == 0
    assert main(["run", "--scenario-file", str(success / "scenario.json"), "--output", str(tmp_path / "copy")]) == 0
    assert main(["run", "--scenario", "closed-box", "--output", str(failure)]) == 1
    assert main(["replay", str(failure / "trace.json")]) == 0
    assert "REPLAY VERIFIED" in capsys.readouterr().out
    data = read_json(success / "trace.json")
    data["status"] = "failed"
    (success / "trace.json").write_text(json.dumps(data), encoding="utf-8")
    assert main(["replay", str(success / "trace.json")]) == 2
    assert "回放不一致" in capsys.readouterr().err
    assert main(["tools"]) == 0
    assert {v["name"] for v in json.loads(capsys.readouterr().out)} == {"navigate", "pick", "place", "open", "inspect"}


def test_module_cli_emits_readable_utf8_through_windows_style_pipe():
    env = {**os.environ, "PYTHONIOENCODING": "gbk"}
    result = subprocess.run([sys.executable, "-m", "voice_lab.simulation", "tools"],
                            cwd=Path(__file__).parents[1], env=env, capture_output=True, check=True)
    descriptions = json.loads(result.stdout.decode("utf-8"))
    assert "网格" in descriptions[0]["description"]
