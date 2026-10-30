"""A discrete simulator. No robot SDK is imported or called here."""

from copy import deepcopy
from datetime import datetime, timezone
from threading import Event, RLock
from time import monotonic
from typing import Protocol
from uuid import uuid4
import math

from .schemas import LOCATIONS, SchemaError, location, validate_action


def now():
    return datetime.now(timezone.utc).isoformat()


class ActionBackend(Protocol):
    def get_status(self) -> dict: ...

    def execute(self, action: dict, *, task_id: str, cancel: Event, timeout: float) -> dict: ...

    def stop(self) -> dict: ...


class SimulatedBackend:
    def __init__(self, start="home", delay=0.25, inspection_results=None):
        location(start, "start")
        if not math.isfinite(delay) or delay < 0:
            raise ValueError("delay 必须是非负有限数")
        self.delay = delay
        self._location = start
        self._lock = RLock()
        self._active = None
        self._cancel = Event()
        self._stopped = Event()
        self._stopped.set()
        self._records = []
        self._failures = {}
        self._inspection_results = dict(inspection_results or {})
        if any(p not in LOCATIONS or v not in ("normal", "abnormal") for p, v in self._inspection_results.items()):
            raise ValueError("检查结果只能为已知地点的 normal / abnormal")

    def get_status(self):
        with self._lock:
            return deepcopy({"current_location": self._location,
                             "motion_state": "moving" if self._active and self._active["name"] == "navigate_to" else "idle",
                             "active_action": self._active, "inspection_records": self._records})

    def fail_next(self, name="navigate_to", count=1):
        if name not in ("navigate_to", "inspect") or type(count) is not int or count < 1:
            raise ValueError("只能向 navigate_to / inspect 注入正整数次失败")
        with self._lock:
            self._failures[name] = count

    def _result(self, action_id, name, status, started, error=None, extra=None):
        data = self.get_status()
        data.update(extra or {})
        return {"action_id": action_id, "action": name, "status": status, "data": data,
                "error": error, "started_at": started, "ended_at": now()}

    def stop(self):
        started = now()
        with self._lock:
            self._cancel.set()
        # execute polls at most every 10ms. paused is only published after this acknowledgement.
        stopped = self._stopped.wait(1)
        return self._result(str(uuid4()), "stop", "succeeded" if stopped else "failed", started,
                            None if stopped else {"code": "stop_timeout", "message": "后端未确认停止"})

    def execute(self, action, *, task_id, cancel, timeout=30):
        action_id, started = str(uuid4()), now()
        try:
            action = validate_action(action)
        except SchemaError as exc:
            return self._result(action_id, "invalid", "failed", started,
                                {"code": "invalid_action", "message": str(exc)})
        name, arguments = action["name"], action["arguments"]
        if name == "get_status":
            return self._result(action_id, name, "succeeded", started)
        if name == "stop":
            return self.stop()
        with self._lock:
            if cancel.is_set():
                return self._result(action_id, name, "cancelled", started, {"code": "cancelled", "message": "已暂停"})
            if self._active:
                return self._result(action_id, name, "failed", started, {"code": "busy", "message": "已有动作正在执行"})
            if name == "inspect" and self._location != arguments["location"]:
                return self._result(action_id, name, "failed", started, {"code": "wrong_location", "message": "尚未到达检查地点"})
            self._active = dict(action, action_id=action_id)
            self._cancel = Event()
            self._stopped.clear()
        status, error, extra = "succeeded", None, {}
        deadline = monotonic() + timeout
        end = monotonic() + self.delay
        try:
            while monotonic() < end and not cancel.is_set() and not self._cancel.is_set():
                if monotonic() >= deadline:
                    break
                cancel.wait(min(0.01, max(0, end - monotonic())))
            with self._lock:
                if cancel.is_set() or self._cancel.is_set():
                    status, error = "cancelled", {"code": "cancelled", "message": "动作已停止，未提交位置或检查记录"}
                elif monotonic() >= deadline:
                    status, error = "failed", {"code": "timeout", "message": "模拟动作超时，需重新读取状态"}
                elif self._failures.get(name, 0):
                    self._failures[name] -= 1
                    status, error = "failed", {"code": "injected_failure", "message": "模拟注入的临时失败"}
                elif name == "navigate_to":
                    self._location = arguments["target"]
                else:
                    record = {"record_id": str(uuid4()), "task_id": task_id, "action_id": action_id,
                              "location": arguments["location"], "result": self._inspection_results.get(self._location, "normal"),
                              "created_at": now()}
                    self._records.append(record)
                    extra["inspection"] = deepcopy(record)
        finally:
            with self._lock:
                self._active = None
                self._stopped.set()
        return self._result(action_id, name, status, started, error, extra)
