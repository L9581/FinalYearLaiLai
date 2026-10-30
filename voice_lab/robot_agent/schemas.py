"""Trust boundaries shared by parsers, planners, and action backends."""

import json

LOCATIONS = ("home", "point_a", "point_b")
ALIASES = {"home": ["基地", "家", "home"], "point_a": ["A点", "A区", "A", "point_a"],
           "point_b": ["B点", "B区", "B", "point_b"]}


class SchemaError(ValueError):
    pass


def obj(properties, required=None):
    return {"type": "object", "properties": properties,
            "required": list(properties) if required is None else required, "additionalProperties": False}


STRING = {"type": "string", "minLength": 1, "maxLength": 2000}
BOOL = {"type": "boolean"}
LOCATION = {"type": "string", "enum": list(LOCATIONS)}
PLAN_SCHEMA = obj({"plan": {"type": "array", "items": STRING, "minItems": 1, "maxItems": 12}})
TASK_SCHEMA = obj({
    "task_type": {"type": "string", "enum": ["navigate", "inspect_locations"]},
    "targets": {"type": "array", "items": LOCATION, "minItems": 1, "maxItems": 3, "uniqueItems": True},
    "constraints": obj({"visit_in_order": BOOL, "return_to_start": BOOL}),
    "completion_rules": obj({"all_targets_inspected": BOOL, "all_targets_reached": BOOL,
                             "at_start": BOOL}, ["at_start"]),
})
TASK_INPUT_SCHEMA = obj({key: value for key, value in TASK_SCHEMA["properties"].items() if key != "completion_rules"})
PARSE_SCHEMA = obj({
    "status": {"type": "string", "enum": ["ready", "needs_clarification", "unsupported"]},
    "task": {"anyOf": [TASK_INPUT_SCHEMA, {"type": "null"}]},
    "clarification_question": {"anyOf": [STRING, {"type": "null"}]},
})
ACTION_SCHEMAS = {
    "navigate_to": obj({"target": LOCATION}), "inspect": obj({"location": LOCATION}),
    "get_status": obj({}), "stop": obj({}),
}
ACTION_SCHEMA = obj({"name": {"type": "string", "enum": list(ACTION_SCHEMAS)},
                     "arguments": obj({"target": LOCATION, "location": LOCATION}, [])})
DECISION_BASE = {
    "decision_summary": STRING,
    "remaining_plan": {"type": "array", "items": STRING, "maxItems": 12},
}
DECISION_SCHEMA = {"anyOf": [
    obj({**DECISION_BASE, "next_action": ACTION_SCHEMA, "request_pause": {"type": "null"}}),
    obj({**DECISION_BASE, "next_action": {"type": "null"}, "request_pause": STRING}),
]}


def fields(data, required, optional=(), path="root"):
    if not isinstance(data, dict):
        raise SchemaError(f"{path}: 必须是 JSON 对象")
    missing, extra = set(required) - data.keys(), data.keys() - set(required) - set(optional)
    if missing or extra:
        raise SchemaError(f"{path}: 缺少字段 {sorted(missing)}；未知字段 {sorted(extra)}")


def text(value, path):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise SchemaError(f"{path}: 必须是 1～2000 字的非空文本")


def location(value, path):
    if not isinstance(value, str) or value not in LOCATIONS:
        raise SchemaError(f"{path}: 未知地点，只允许 {LOCATIONS}")


def read_json(value):
    def pairs(items):
        result = {}
        for key, item in items:
            if key in result:
                raise SchemaError(f"JSON 重复字段: {key}")
            result[key] = item
        return result

    def constant(value):
        raise SchemaError(f"JSON 不允许 {value}")

    try:
        return json.loads(value, object_pairs_hook=pairs, parse_constant=constant)
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"JSON 无效: {exc}") from exc


def validate_task(data):
    fields(data, ("task_type", "targets", "constraints"), ("completion_rules",), "task")
    if data["task_type"] not in ("navigate", "inspect_locations"):
        raise SchemaError("task.task_type: 只支持 navigate / inspect_locations")
    targets = data["targets"]
    if not isinstance(targets, list) or not 1 <= len(targets) <= 3:
        raise SchemaError("task.targets: 必须包含 1～3 个地点")
    for i, target in enumerate(targets):
        location(target, f"task.targets[{i}]")
    if len(targets) != len(set(targets)):
        raise SchemaError("task.targets: 不支持重复地点任务")
    if data["task_type"] == "navigate" and len(targets) != 1:
        raise SchemaError("task.targets: navigate 当前只支持一个目标地点")
    constraints = data["constraints"]
    fields(constraints, ("visit_in_order", "return_to_start"), path="task.constraints")
    for key, value in constraints.items():
        if type(value) is not bool:
            raise SchemaError(f"task.constraints.{key}: 必须是布尔值")
    progress_key = "all_targets_inspected" if data["task_type"] == "inspect_locations" else "all_targets_reached"
    rules = {progress_key: True, "at_start": constraints["return_to_start"]}
    supplied = data.get("completion_rules", {})
    fields(supplied, (), rules, "task.completion_rules")
    for key, value in supplied.items():
        if type(value) is not bool or value != rules[key]:
            raise SchemaError(f"task.completion_rules.{key}: 必须是 {rules[key]}，不能降低完成标准")
    return {"task_type": data["task_type"], "targets": list(targets),
            "constraints": dict(constraints), "completion_rules": rules}


def validate_parse(data):
    fields(data, ("status", "task", "clarification_question"))
    if data["status"] == "ready":
        if data["clarification_question"] is not None:
            raise SchemaError("ready 状态的 clarification_question 必须是 null")
        return dict(data, task=validate_task(data["task"]))
    if data["status"] not in ("needs_clarification", "unsupported"):
        raise SchemaError("status: 未知解析状态")
    if data["task"] is not None:
        raise SchemaError("尚不可执行时 task 必须是 null")
    text(data["clarification_question"], "clarification_question")
    return dict(data)


def validate_action(data):
    fields(data, ("name", "arguments"), path="next_action")
    name = data["name"]
    if not isinstance(name, str) or name not in ACTION_SCHEMAS:
        raise SchemaError("next_action.name: 不在动作白名单内")
    arguments = data["arguments"]
    fields(arguments, ACTION_SCHEMAS[name]["properties"], path="next_action.arguments")
    for key, value in arguments.items():
        location(value, f"next_action.arguments.{key}")
    return {"name": name, "arguments": dict(arguments)}


def validate_plan(data):
    fields(data, ("plan",))
    steps = data["plan"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= 12:
        raise SchemaError("plan: 必须有 1～12 个步骤")
    for step in steps:
        text(step, "plan[]")
    return list(steps)


def validate_decision(data):
    fields(data, ("decision_summary", "remaining_plan", "next_action", "request_pause"))
    text(data["decision_summary"], "decision_summary")
    plan = data["remaining_plan"]
    if not isinstance(plan, list) or len(plan) > 12:
        raise SchemaError("remaining_plan: 必须是最多 12 项的数组")
    for step in plan:
        text(step, "remaining_plan[]")
    if data["request_pause"] is not None:
        text(data["request_pause"], "request_pause")
        if data["request_pause"].strip().lower() in ("null", "none", "无", "不需要暂停"):
            raise SchemaError("request_pause: 正常执行使用 JSON null，不能把字符串 null 当作暂停原因")
        if data["next_action"] is not None:
            raise SchemaError("请求暂停时不能同时要求执行动作")
    action = None if data["next_action"] is None else validate_action(data["next_action"])
    return dict(data, next_action=action, remaining_plan=list(plan))
