"""Strict JSON boundaries and immutable snapshots for the simulation layer."""

from dataclasses import asdict, dataclass, fields
import json


class SchemaError(ValueError):
    pass


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{name} must be a non-empty string")


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise SchemaError(f"{name} must be an integer >= {minimum}")


def exact(data, names):
    if not isinstance(data, dict) or set(data) != set(names):
        raise SchemaError(f"expected exactly these fields: {', '.join(sorted(names))}")
    return data


class JsonRecord:
    def to_dict(self):
        # JSON lists rather than Python tuples also make replay comparison exact.
        return json.loads(json.dumps(asdict(self), ensure_ascii=False, allow_nan=False))

    @classmethod
    def from_dict(cls, data):
        return cls(**exact(data, [f.name for f in fields(cls)]))


@dataclass(frozen=True)
class Position(JsonRecord):
    x: int
    y: int

    def __post_init__(self):
        integer(self.x, "x")
        integer(self.y, "y")


@dataclass(frozen=True)
class Fixture(JsonRecord):
    id: str
    name: str
    position: Position
    kind: str  # surface | container; position is the robot's interaction cell.
    is_open: bool

    def __post_init__(self):
        text(self.id, "fixture.id")
        text(self.name, "fixture.name")
        if not isinstance(self.position, Position):
            raise SchemaError("fixture.position must be a Position")
        if self.kind not in ("surface", "container") or type(self.is_open) is not bool:
            raise SchemaError("invalid fixture kind or is_open")
        if self.kind == "surface" and not self.is_open:
            raise SchemaError("a surface cannot be closed")

    @classmethod
    def from_dict(cls, data):
        values = dict(exact(data, ("id", "name", "position", "kind", "is_open")))
        values["position"] = Position.from_dict(values["position"])
        return cls(**values)


@dataclass(frozen=True)
class WorldObject(JsonRecord):
    id: str
    name: str
    color: str
    location: str  # fixture id, or the reserved location "robot".

    def __post_init__(self):
        for name in ("id", "name", "color", "location"):
            text(getattr(self, name), f"object.{name}")


@dataclass(frozen=True)
class WorldState(JsonRecord):
    width: int
    height: int
    blocked: tuple[Position, ...]
    robot: Position
    holding: str | None
    fixtures: tuple[Fixture, ...]
    objects: tuple[WorldObject, ...]
    revision: int = 0

    def __post_init__(self):
        integer(self.width, "width", 1)
        integer(self.height, "height", 1)
        integer(self.revision, "revision")
        if self.width * self.height > 10000:
            raise SchemaError("grid must contain at most 10000 cells")
        for name, kind in (("blocked", Position), ("fixtures", Fixture), ("objects", WorldObject)):
            values = getattr(self, name)
            if not isinstance(values, tuple) or not all(isinstance(v, kind) for v in values):
                raise SchemaError(f"{name} must be an immutable tuple of {kind.__name__}")
        if not isinstance(self.robot, Position):
            raise SchemaError("robot must be a Position")
        positions = (*self.blocked, self.robot, *(f.position for f in self.fixtures))
        if any(p.x >= self.width or p.y >= self.height for p in positions):
            raise SchemaError("position outside grid")
        if len(set(self.blocked)) != len(self.blocked):
            raise SchemaError("duplicate blocked cells")
        if self.robot in self.blocked or any(f.position in self.blocked for f in self.fixtures):
            raise SchemaError("robot and interaction cells must be walkable")
        if len({f.position for f in self.fixtures}) != len(self.fixtures):
            raise SchemaError("fixture interaction cells must be distinct")
        ids = [v.id for v in (*self.fixtures, *self.objects)]
        if len(ids) != len(set(ids)) or "robot" in ids:
            raise SchemaError("entity ids must be unique; robot is reserved")
        locations = {f.id for f in self.fixtures} | {"robot"}
        if any(o.location not in locations for o in self.objects):
            raise SchemaError("object has an unknown location")
        if self.holding is not None:
            text(self.holding, "holding")
        held = [o.id for o in self.objects if o.location == "robot"]
        if held != ([] if self.holding is None else [self.holding]):
            raise SchemaError("holding must match exactly one object at robot, or none")

    @classmethod
    def from_dict(cls, data):
        values = dict(exact(data, [f.name for f in fields(cls)]))
        for name, kind in (("blocked", Position), ("fixtures", Fixture), ("objects", WorldObject)):
            if not isinstance(values[name], list):
                raise SchemaError(f"{name} must be a JSON array")
            values[name] = tuple(kind.from_dict(v) for v in values[name])
        values["robot"] = Position.from_dict(values["robot"])
        return cls(**values)


@dataclass(frozen=True)
class Goal(JsonRecord):
    object_id: str
    target: str

    def __post_init__(self):
        text(self.object_id, "goal.object_id")
        text(self.target, "goal.target")

    def satisfied_by(self, state: WorldState) -> bool:
        return (any(f.id == self.target for f in state.fixtures)
                and any(o.id == self.object_id and o.location == self.target for o in state.objects))


@dataclass(frozen=True)
class SemanticFrame(JsonRecord):
    transcript: str
    intent: str
    goal: Goal
    source: str = "manual"

    def __post_init__(self):
        text(self.transcript, "transcript")
        if self.intent != "task" or self.source != "manual" or not isinstance(self.goal, Goal):
            raise SchemaError("phase 1 accepts only manually specified task goals")

    @classmethod
    def from_dict(cls, data):
        values = dict(exact(data, ("transcript", "intent", "goal", "source")))
        values["goal"] = Goal.from_dict(values["goal"])
        return cls(**values)


@dataclass(frozen=True)
class ActionCall(JsonRecord):
    id: str
    tool: str
    arguments: dict[str, str]

    def __post_init__(self):
        text(self.id, "action.id")
        text(self.tool, "action.tool")
        if not isinstance(self.arguments, dict):
            raise SchemaError("action.arguments must be a JSON object")
        for key, value in self.arguments.items():
            text(key, "argument name")
            text(value, f"argument {key}")
        # Do not retain the caller's dictionary. Registry checks tool-specific fields.
        object.__setattr__(self, "arguments", dict(self.arguments))


@dataclass(frozen=True)
class Plan(JsonRecord):
    actions: tuple[ActionCall, ...]
    source: str = "manual"

    def __post_init__(self):
        if self.source != "manual":
            raise SchemaError("phase 1 plans must be labelled manual")
        if not isinstance(self.actions, tuple) or not all(isinstance(a, ActionCall) for a in self.actions):
            raise SchemaError("plan.actions must be a tuple of ActionCall")
        if len(self.actions) > 100:
            raise SchemaError("plan exceeds 100 actions")
        if len({a.id for a in self.actions}) != len(self.actions):
            raise SchemaError("action ids must be unique within a plan")

    @classmethod
    def from_dict(cls, data):
        exact(data, ("actions", "source"))
        if not isinstance(data["actions"], list):
            raise SchemaError("plan.actions must be a JSON array")
        return cls(tuple(ActionCall.from_dict(a) for a in data["actions"]), data["source"])


@dataclass(frozen=True)
class ToolResult(JsonRecord):
    success: bool
    observation: WorldState
    error_code: str | None
    message: str
    path: tuple[Position, ...] = ()

    def __post_init__(self):
        if type(self.success) is not bool or not isinstance(self.observation, WorldState):
            raise SchemaError("invalid tool result")
        text(self.message, "result.message")
        if self.success and self.error_code is not None:
            raise SchemaError("successful result cannot have an error code")
        if not self.success:
            text(self.error_code, "result.error_code")
        if not isinstance(self.path, tuple) or not all(isinstance(p, Position) for p in self.path):
            raise SchemaError("result.path must be a tuple of Position")
