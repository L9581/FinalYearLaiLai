"""Closed tool registry: validates arguments before any world transition."""

from dataclasses import dataclass
from types import MappingProxyType

from .schemas import ActionCall


class ToolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class ToolSpec:
    name: str
    parameters: tuple[str, ...]
    description: str

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {p: {"type": "string", "minLength": 1} for p in self.parameters},
                "required": list(self.parameters),
                "additionalProperties": False,
            },
        }


class ToolRegistry:
    def __init__(self):
        specs = (
            ToolSpec("navigate", ("target",), "沿可通行网格移动到设施的交互格"),
            ToolSpec("pick", ("object_id",), "从当前位置拿起物品；手必须为空，容器必须已打开"),
            ToolSpec("place", ("object_id", "target"), "把手持物放到当前位置的设施；容器必须已打开"),
            ToolSpec("open", ("target",), "打开当前位置的容器"),
            ToolSpec("inspect", (), "返回完整场景快照；不改变状态"),
        )
        self._specs = MappingProxyType({s.name: s for s in specs})

    def describe(self):
        return [s.to_dict() for s in self._specs.values()]

    def validate(self, action: ActionCall):
        spec = self._specs.get(action.tool)
        if spec is None:
            raise ToolError("UNKNOWN_TOOL", f"未注册工具：{action.tool}")
        if set(action.arguments) != set(spec.parameters):
            raise ToolError("INVALID_ARGUMENTS", f"{action.tool} 只接受参数：{', '.join(spec.parameters) or '(无)'}")
