"""Atomic, deterministic transitions in a small symbolic grid environment."""

from collections import deque
from dataclasses import replace

from .schemas import ActionCall, Fixture, Position, SchemaError, ToolResult, WorldObject, WorldState
from .tools import ToolError, ToolRegistry


def default_world() -> WorldState:
    return WorldState(
        width=7, height=5,
        blocked=tuple(Position(3, y) for y in (0, 1, 3, 4)),
        robot=Position(0, 2), holding=None,
        fixtures=(
            Fixture("table", "客厅桌子", Position(1, 1), "surface", True),
            Fixture("shelf", "厨房架子", Position(5, 1), "surface", True),
            Fixture("box", "厨房带盖盒子", Position(5, 3), "container", False),
        ),
        objects=(
            WorldObject("red_cup", "杯子", "红色", "table"),
            WorldObject("blue_cup", "杯子", "蓝色", "table"),
            WorldObject("apple", "苹果", "红色", "box"),
        ),
    )


class GridWorld:
    def __init__(self, initial: WorldState | None = None):
        self._initial = initial if initial is not None else default_world()
        if not isinstance(self._initial, WorldState):
            raise SchemaError("initial state must be WorldState")
        self.registry = ToolRegistry()
        self._state = self._initial

    def reset(self) -> WorldState:
        self._state = self._initial
        return self.observe()

    def observe(self) -> WorldState:
        # State and every nested member are immutable; callers cannot modify the world.
        return self._state

    def step(self, action: ActionCall | dict) -> ToolResult:
        try:
            # Revalidate even dataclass inputs, since arguments is a JSON dictionary.
            raw = {"id": action.id, "tool": action.tool, "arguments": action.arguments} if isinstance(action, ActionCall) else action
            action = ActionCall.from_dict(raw)
        except SchemaError as exc:
            return ToolResult(False, self.observe(), "INVALID_ACTION", str(exc))
        try:
            self.registry.validate(action)
            # Only these handlers can execute. User-supplied strings are never eval'ed.
            handlers = {"navigate": self._navigate, "pick": self._pick, "place": self._place,
                        "open": self._open, "inspect": self._inspect}
            candidate, path, message = handlers[action.tool](**action.arguments)
        except ToolError as exc:
            return ToolResult(False, self.observe(), exc.code, str(exc))
        if candidate != self._state:
            candidate = replace(candidate, revision=self._state.revision + 1)
        # Commit only after the complete transition and all state invariants succeed.
        self._state = candidate
        return ToolResult(True, self.observe(), None, message, path)

    def _fixture(self, target):
        for fixture in self._state.fixtures:
            if fixture.id == target:
                return fixture
        raise ToolError("UNKNOWN_TARGET", f"没有设施：{target}")

    def _object(self, object_id):
        for obj in self._state.objects:
            if obj.id == object_id:
                return obj
        raise ToolError("UNKNOWN_OBJECT", f"没有物品：{object_id}")

    def _accessible(self, fixture):
        if self._state.robot != fixture.position:
            raise ToolError("NOT_AT_TARGET", f"机器人尚未到达：{fixture.id}")
        if not fixture.is_open:
            raise ToolError("CONTAINER_CLOSED", f"容器尚未打开：{fixture.id}")

    def _route(self, goal):
        start = self._state.robot
        queue = deque([start])
        previous = {start: None}
        blocked = set(self._state.blocked)
        # Fixed neighbor order makes paths reproducible.
        while queue:
            current = queue.popleft()
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = previous[current]
                return tuple(reversed(path))
            for dx, dy in ((0, -1), (1, 0), (0, 1), (-1, 0)):
                x, y = current.x + dx, current.y + dy
                if not (0 <= x < self._state.width and 0 <= y < self._state.height):
                    continue
                cell = Position(x, y)
                if cell not in blocked and cell not in previous:
                    previous[cell] = current
                    queue.append(cell)
        raise ToolError("UNREACHABLE", "目标不可达：没有可通行路径")

    def _navigate(self, target):
        fixture = self._fixture(target)
        path = self._route(fixture.position)
        return replace(self._state, robot=fixture.position), path, f"已到达 {target}"

    def _pick(self, object_id):
        obj = self._object(object_id)
        if self._state.holding is not None:
            raise ToolError("HAND_FULL", "机器人已经拿着一个物品")
        self._accessible(self._fixture(obj.location))
        objects = tuple(replace(o, location="robot") if o.id == object_id else o for o in self._state.objects)
        return replace(self._state, objects=objects, holding=object_id), (), f"已拿起 {object_id}"

    def _place(self, object_id, target):
        self._object(object_id)
        fixture = self._fixture(target)
        if self._state.holding != object_id:
            raise ToolError("NOT_HOLDING", f"机器人没有拿着：{object_id}")
        self._accessible(fixture)
        objects = tuple(replace(o, location=target) if o.id == object_id else o for o in self._state.objects)
        return replace(self._state, objects=objects, holding=None), (), f"已将 {object_id} 放到 {target}"

    def _open(self, target):
        fixture = self._fixture(target)
        if fixture.kind != "container":
            raise ToolError("NOT_CONTAINER", f"不是可打开的容器：{target}")
        if self._state.robot != fixture.position:
            raise ToolError("NOT_AT_TARGET", f"机器人尚未到达：{target}")
        fixtures = tuple(replace(f, is_open=True) if f.id == target else f for f in self._state.fixtures)
        return replace(self._state, fixtures=fixtures), (), f"容器已打开：{target}"

    def _inspect(self):
        return self._state, (), "已读取完整场景快照"


def render_text(state: WorldState) -> str:
    cells = [["." for _ in range(state.width)] for _ in range(state.height)]
    for p in state.blocked:
        cells[p.y][p.x] = "#"
    for fixture in state.fixtures:
        cells[fixture.position.y][fixture.position.x] = fixture.id[0].upper()
    cells[state.robot.y][state.robot.x] = "R"
    grid = "\n".join(" ".join(row) for row in cells)
    objects = ", ".join(f"{o.id}@{o.location}" for o in state.objects)
    return f"{grid}\nR=robot #=wall T=table S=shelf B=box\nholding={state.holding}; {objects}"
