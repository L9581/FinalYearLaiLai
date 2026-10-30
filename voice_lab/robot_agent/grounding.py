"""Conservative checks for the finite Chinese command vocabulary of this stage.

The LLM extracts task intent. These checks bind locations and explicit constraints
to user text, so a well-formed hallucination cannot silently pick another place.
"""

import re

from .schemas import validate_task

LOCATION_PATTERNS = {
    "point_a": r"point_a|(?<![a-z0-9_])a(?![a-z0-9_])",
    "point_b": r"point_b|(?<![a-z0-9_])b(?![a-z0-9_])",
    "home": r"(?<![a-z])home(?![a-z])|基地|家",
}
RETURN = r"回来|回去|(?:返回|回到|回)(?:本次任务的|本次的|本次|任务的|最初的)?(?:出发点|起点|原点|原地|原位)|返航"
NEGATION = r"(?:不用|不要|不必|无需|不需要|别)\s*(?:再|返回后)?\s*$"


def ground_parse(result, user_texts):
    source = "\n".join(user_texts)
    normalized = re.sub(r"\s+", "", source).lower()
    # Explicit unsupported operations cannot be dropped from a compound command.
    if re.search(r"拍照|摄影|跳舞|跳一段舞|(?:把|将).{0,24}(?:拿|带|搬|抓|送)|定时|每隔|循环巡逻", normalized):
        return {"status": "unsupported", "task": None,
                "clarification_question": "当前只支持前往地点和模拟检查，不能执行拍照、取物、舞蹈或定时任务。"}
    unknown = [m.group(1) for m in re.finditer(r"(?<![a-z_])([a-z])(?:点|区)", normalized) if m.group(1) not in ("a", "b")]
    if unknown:
        return {"status": "needs_clarification", "task": None,
                "clarification_question": f"未知地点 {', '.join(unknown).upper()}。请指定 A 点、B 点或基地。"}
    # A relative region has no geometry in this discrete simulator. Keep the
    # inspection intent but never silently turn it into a guessed route.
    latest = user_texts[-1]
    latest_places = {key for key, pattern in LOCATION_PATTERNS.items() if re.search(pattern, latest, re.I)}
    if (re.search(r"附近|周围|周边", latest) and re.search(r"检查|巡检|巡查|巡逻", source)
            and (not latest_places or re.search(r"(?:点|区|基地|home|point_[ab])\s*(?:的)?\s*(?:附近|周围|周边)", latest, re.I))):
        return {"status": "needs_clarification", "task": None,
                "clarification_question": "我理解你想做一次巡检。当前模拟环境只有基地、A 点和 B 点，"
                "还没有定义“附近”的范围。你想检查哪些地点？可回复“A 点和 B 点”；"
                "如果还需要按顺序检查并返回，可回复“先检查 A 点，再检查 B 点，最后回来”。"}
    if result["status"] != "ready":
        return result
    # Keep spaces for ASCII word boundaries; A 点 and point_a both match.
    mentioned = {key for key, pattern in LOCATION_PATTERNS.items() if re.search(pattern, source, re.I)}
    if not mentioned or any(p not in mentioned for p in result["task"]["targets"]):
        return {"status": "needs_clarification", "task": None,
                "clarification_question": "请明确目标地点：A 点、B 点或基地；不能根据缺失信息猜测地点。"}
    # Negated destinations and relative references need a fresh explicit statement.
    if re.search(r"(?:不要|别|不用|不去).{0,5}(?:a|b|point_a|point_b|基地)", normalized):
        return {"status": "needs_clarification", "task": None,
                "clarification_question": "指令包含地点否定，请直接说明本次要前往或检查的地点。"}
    returns = False
    for match in re.finditer(RETURN, normalized):
        returns = not bool(re.search(NEGATION, normalized[max(0, match.start() - 10):match.start()]))
    ordered = False
    for match in re.finditer(r"顺序不限|任意顺序|不分先后|依次|先|再|然后|最后|接着|接下来|按.{0,12}顺序", normalized):
        ordered = match.group() not in ("顺序不限", "任意顺序", "不分先后")
    # Completion rules describe requirements, not model-estimated progress.
    data = {key: value for key, value in result["task"].items() if key != "completion_rules"}
    data["constraints"] = {"visit_in_order": ordered, "return_to_start": returns}
    return dict(result, task=validate_task(data))
