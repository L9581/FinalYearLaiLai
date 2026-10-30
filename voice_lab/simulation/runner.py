"""Manual scenario execution and independently re-executed trace verification."""

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

from .schemas import (ActionCall, JsonRecord, Plan, SchemaError, SemanticFrame,
                      ToolResult, WorldState, exact, text)
from .world import GridWorld, render_text


@dataclass(frozen=True)
class Scenario(JsonRecord):
    id: str
    description: str
    initial_state: WorldState
    semantic_frame: SemanticFrame
    plan: Plan

    def __post_init__(self):
        text(self.id, "scenario.id")
        text(self.description, "scenario.description")
        # Scenario ids are also used as default output directory names.
        if any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.id):
            raise SchemaError("scenario.id must contain only lowercase ASCII letters, digits, - or _")
        if not isinstance(self.initial_state, WorldState) or not isinstance(self.semantic_frame, SemanticFrame):
            raise SchemaError("invalid scenario state or semantic frame")
        if not isinstance(self.plan, Plan):
            raise SchemaError("invalid scenario plan")

    @classmethod
    def from_dict(cls, data):
        exact(data, ("id", "description", "initial_state", "semantic_frame", "plan"))
        return cls(data["id"], data["description"], WorldState.from_dict(data["initial_state"]),
                   SemanticFrame.from_dict(data["semantic_frame"]), Plan.from_dict(data["plan"]))


@dataclass(frozen=True)
class ExecutionEvent(JsonRecord):
    action: ActionCall
    before: WorldState
    result: ToolResult


@dataclass(frozen=True)
class ExecutionTrace(JsonRecord):
    schema_version: int
    execution_id: str
    scenario: Scenario
    events: tuple[ExecutionEvent, ...]
    final_state: WorldState
    status: str
    goal_satisfied: bool


def run_scenario(scenario: Scenario, *, execution_id: str | None = None) -> ExecutionTrace:
    # Snapshot mutable JSON argument dictionaries, and validate the complete boundary.
    scenario = Scenario.from_dict(scenario.to_dict())
    execution_id = execution_id if execution_id is not None else str(uuid4())
    text(execution_id, "execution_id")
    world = GridWorld(scenario.initial_state)
    world.reset()
    events = []
    for action in scenario.plan.actions:
        before = world.observe()
        result = world.step(action)
        events.append(ExecutionEvent(action, before, result))
        if not result.success:
            break
    final = world.observe()
    goal_satisfied = scenario.semantic_frame.goal.satisfied_by(final)
    failed = any(not event.result.success for event in events)
    status = "failed" if failed else "completed" if goal_satisfied else "incomplete"
    return ExecutionTrace(1, execution_id, scenario, tuple(events), final, status, goal_satisfied)


def read_json(path: str | Path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise SchemaError(f"duplicate JSON field: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise SchemaError(f"non-finite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"),
                      object_pairs_hook=unique_object, parse_constant=reject_constant)


def verify_trace(data: dict) -> ExecutionTrace:
    """Recompute every before/after state, result, route and final goal from scratch."""
    exact(data, ("schema_version", "execution_id", "scenario", "events", "final_state", "status", "goal_satisfied"))
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        raise SchemaError("unsupported trace schema version")
    scenario = Scenario.from_dict(data["scenario"])
    actual = run_scenario(scenario, execution_id=data["execution_id"])
    # Compare canonical JSON, not Python == (which considers True == 1).
    try:
        saved = json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)
        replayed = json.dumps(actual.to_dict(), sort_keys=True, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SchemaError("trace is not valid JSON") from exc
    if saved != replayed:
        raise SchemaError("回放不一致：动作结果、路径、状态或最终目标与重新执行的结果不同")
    return actual


def render_trace(trace: ExecutionTrace) -> str:
    lines = [f"Scenario: {trace.scenario.id}", trace.scenario.description,
             f"指令（人工标注）: {trace.scenario.semantic_frame.transcript}",
             "语义和动作计划来源: manual（未调用 LLM/VLM/PDDL）", "\nInitial state:",
             render_text(trace.scenario.initial_state)]
    for index, event in enumerate(trace.events, 1):
        result = event.result
        args = json.dumps(event.action.arguments, ensure_ascii=False)
        lines.extend((f"\nStep {index}: {event.action.tool} {args}",
                      f"{'OK' if result.success else result.error_code}: {result.message}"))
        if result.path:
            lines.append("Route: " + " -> ".join(f"({p.x},{p.y})" for p in result.path))
        lines.append(render_text(result.observation))
    skipped = len(trace.scenario.plan.actions) - len(trace.events)
    lines.append(f"\nstatus={trace.status}; goal_satisfied={str(trace.goal_satisfied).lower()}; "
                 f"executed={len(trace.events)}; skipped={skipped}")
    return "\n".join(lines) + "\n"


def save_trace(trace: ExecutionTrace, directory: str | Path):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, record in (("scenario.json", trace.scenario), ("trace.json", trace)):
        (directory / name).write_text(json.dumps(record.to_dict(), ensure_ascii=False, indent=2,
                                                allow_nan=False) + "\n", encoding="utf-8")
    (directory / "replay.txt").write_text(render_trace(trace), encoding="utf-8")
