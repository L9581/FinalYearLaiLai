"""Small, explicitly hand-authored scenarios for execution-layer validation."""

from dataclasses import replace

from .runner import Scenario
from .schemas import ActionCall, Goal, Plan, Position, SemanticFrame
from .world import default_world


SCENARIO_NAMES = ("move-cup", "open-box", "closed-box", "missing-object", "blocked-path", "unknown-tool")


def make_scenario(name: str = "move-cup") -> Scenario:
    initial = default_world()
    goal = Goal("red_cup", "shelf")
    transcript = "把桌上的红色杯子拿到厨房的架子上。"
    description = "人工动作计划：检查 → 到桌子 → 拿红杯 → 到架子 → 放下 → 检查。"
    calls = [("inspect", {}), ("navigate", {"target": "table"}),
             ("pick", {"object_id": "red_cup"}), ("navigate", {"target": "shelf"}),
             ("place", {"object_id": "red_cup", "target": "shelf"}), ("inspect", {})]
    if name in ("open-box", "closed-box"):
        transcript = "把厨房盒子里的苹果放到桌子上。"
        goal = Goal("apple", "table")
        calls = [("navigate", {"target": "box"})]
        if name == "open-box":
            calls.append(("open", {"target": "box"}))
        calls.extend([("pick", {"object_id": "apple"}), ("navigate", {"target": "table"}),
                      ("place", {"object_id": "apple", "target": "table"})])
        description = ("人工计划包含开盒动作，应完成任务。" if name == "open-box"
                       else "故意漏掉开盒动作，应返回 CONTAINER_CLOSED 并停止后续动作。")
    elif name == "missing-object":
        transcript = "把桌上的绿色杯子拿到厨房架子上。"
        goal = Goal("green_cup", "shelf")
        calls = [("navigate", {"target": "table"}), ("pick", {"object_id": "green_cup"})]
        description = "绿色杯子不存在，应返回 UNKNOWN_OBJECT。"
    elif name == "blocked-path":
        initial = replace(initial, blocked=initial.blocked + (Position(3, 2),))
        description = "堵住两侧之间的通道；拿起杯子后导航应返回 UNREACHABLE，杯子仍在手里。"
    elif name == "unknown-tool":
        calls = [("teleport", {"target": "shelf"})]
        description = "teleport 未注册，应返回 UNKNOWN_TOOL，状态完全不变。"
    elif name != "move-cup":
        raise ValueError(f"unknown scenario: {name}")
    actions = tuple(ActionCall(f"step-{i}", tool, args) for i, (tool, args) in enumerate(calls, 1))
    return Scenario(name, description, initial, SemanticFrame(transcript, "task", goal), Plan(actions))
