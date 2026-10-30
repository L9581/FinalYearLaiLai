"""Task lifecycle, cancellation, bounded execution and independent completion checks."""

from copy import deepcopy
import json
import math
from pathlib import Path
from queue import Empty, Queue
from threading import Event, RLock, Thread
from time import monotonic
from uuid import uuid4

from .backend import ActionBackend, now
from .schemas import SchemaError, validate_task, validate_plan, validate_decision


class TaskController:
    def __init__(self, backend: ActionBackend, planner, *, output=None, max_rounds=20,
                 max_retries=2, max_no_progress=6, model_timeout=90, action_timeout=30):
        for name, value, minimum in (("max_rounds", max_rounds, 1), ("max_retries", max_retries, 0),
                                     ("max_no_progress", max_no_progress, 1)):
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} 必须是 >= {minimum} 的整数")
        if any(not math.isfinite(v) or v <= 0 for v in (model_timeout, action_timeout)):
            raise ValueError("超时必须是正的有限秒数")
        self.backend, self.planner = backend, planner
        self.max_rounds, self.max_retries, self.max_no_progress = max_rounds, max_retries, max_no_progress
        self.model_timeout, self.action_timeout = model_timeout, action_timeout
        self.output = Path(output) if output else None
        self._lock = RLock()
        self._cancel = Event()
        self._worker = None
        self.state = "idle"
        self.task_id = None
        self.task = None
        self.start_location = None
        self.reason = None
        self.round = 0
        self.initial_plan = []
        self.history = []
        self.events = []
        self._reached = set()

    def _emit(self, kind, **data):
        with self._lock:
            event = deepcopy({"sequence": len(self.events) + 1, "time": now(), "task_id": self.task_id,
                              "round": self.round, "kind": kind, **data})
            self.events.append(event)
            if self.output:
                self.output.mkdir(parents=True, exist_ok=True)
                with (self.output / "events.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
            return event

    def start(self, task, *, original_text="", parse_turns=None):
        task = validate_task(task)
        with self._lock:
            if self.state != "idle":
                raise RuntimeError("每个控制器只处理一个任务；请先完成或暂停当前任务")
            if self.output and (self.output / "events.jsonl").exists():
                raise ValueError("日志目录已包含任务，请使用新目录以免混入旧记录")
            observation = self.backend.get_status()
            if observation["active_action"] is not None or observation["motion_state"] != "idle":
                raise RuntimeError("后端尚未静止，不能开始新任务")
            self.task, self.task_id = task, str(uuid4())
            self.start_location = observation["current_location"]
            self._reached = {self.start_location}
            self.state = "running"
            self._emit("task_started", task=task, original_text=original_text, parse_turns=parse_turns or [],
                       start_location=self.start_location, observation=observation,
                       planner=type(self.planner).__name__, limits={"max_rounds": self.max_rounds,
                       "max_retries": self.max_retries, "max_no_progress": self.max_no_progress,
                       "model_timeout": self.model_timeout, "action_timeout": self.action_timeout})
            self._launch()
        return self.task_id

    def _launch(self):
        # Each run has a new cancellation token. Late model results only enter their private queue.
        self._cancel = Event()
        self._worker = Thread(target=self._run, args=(self._cancel,), daemon=True)
        self._worker.start()

    def pause(self, reason="人工暂停"):
        with self._lock:
            if self.state != "running":
                return False
            self.reason, self.state = reason, "pausing"
            self._cancel.set()
            self._emit("pausing", reason=reason)
            return True

    def resume(self):
        with self._lock:
            if self.state != "paused":
                raise RuntimeError("只有已确认停止的 paused 任务可以恢复")
            observation = self.backend.get_status()
            if observation["active_action"] is not None or observation["motion_state"] != "idle":
                raise RuntimeError("后端尚未确认静止，不能恢复")
            self.state, self.reason = "running", None
            self._emit("resumed", observation=observation, progress=self._progress(observation))
            self._launch()

    def wait(self, timeout=None):
        worker = self._worker
        if worker:
            worker.join(timeout)
        return self.snapshot()

    def _progress(self, observation):
        if not self.task:
            return {}
        records = [r for r in observation["inspection_records"] if r["task_id"] == self.task_id]
        inspected = [r["location"] for r in records]
        targets = self.task["targets"]
        done = [p for p in targets if p in (inspected if self.task["task_type"] == "inspect_locations" else self._reached)]
        order_ok = not self.task["constraints"]["visit_in_order"] or self.task["task_type"] == "navigate" or inspected == targets[:len(inspected)]
        at_start = observation["current_location"] == self.start_location
        final_location_ok = (at_start if self.task["constraints"]["return_to_start"] else
                             observation["current_location"] == targets[0] if self.task["task_type"] == "navigate" else True)
        complete = (done == targets and order_ok and final_location_ok and
                    observation["active_action"] is None and observation["motion_state"] == "idle")
        return {"done_targets": done, "inspection_records": records, "order_ok": order_ok,
                "at_start": at_start, "final_location_ok": final_location_ok, "completed": complete}

    def snapshot(self):
        with self._lock:
            observation = self.backend.get_status()
            return deepcopy({"task_id": self.task_id, "state": self.state, "reason": self.reason,
                             "task": self.task, "start_location": self.start_location, "round": self.round,
                             "initial_plan": self.initial_plan, "observation": observation,
                             "progress": self._progress(observation)})

    def events_since(self, index):
        with self._lock:
            return deepcopy(self.events[index:])

    def _context(self):
        context = self.snapshot()
        # Full history is on disk; this bounded window keeps model requests small.
        context["history"] = deepcopy(self.history[-8:])
        return context

    def _model_call(self, callback, cancel):
        answer = Queue(maxsize=1)

        def generate():
            try:
                answer.put((True, callback()))
            except Exception as exc:
                answer.put((False, exc))

        if cancel.is_set():
            raise InterruptedError()
        Thread(target=generate, daemon=True).start()
        deadline = monotonic() + self.model_timeout
        while True:
            if cancel.is_set():
                raise InterruptedError()
            if monotonic() >= deadline:
                raise TimeoutError("模型请求超时；已停止调度")
            try:
                ok, value = answer.get(timeout=min(0.02, max(0, deadline - monotonic())))
            except Empty:
                continue
            if cancel.is_set():
                raise InterruptedError()
            if not ok:
                raise value
            return value

    def _feedback_error(self, message):
        result = {"status": "failed", "error": {"code": "validation_error", "message": message},
                  "data": self.backend.get_status()}
        self.history.append(result)
        self._emit("validation_error", observation=result)

    def _check_scope(self, action, context):
        target = next(iter(action["arguments"].values()), None)
        allowed = set(self.task["targets"])
        if self.task["constraints"]["return_to_start"]:
            allowed.add(self.start_location)
        if target is not None and target not in allowed:
            raise SchemaError("动作目标不属于本任务或返回位置")
        remaining = [p for p in self.task["targets"] if p not in context["progress"]["done_targets"]]
        if (action["name"] == "navigate_to" and not remaining and self.task["constraints"]["return_to_start"]
                and target != self.start_location):
            raise SchemaError(f"所有目标已完成，剩余要求是返回出发点 {self.start_location}")
        if action["name"] == "navigate_to" and remaining and self.task["task_type"] == "inspect_locations":
            if self.task["constraints"]["visit_in_order"] and target != remaining[0]:
                raise SchemaError(f"顺序约束要求先到达并 inspect({remaining[0]})，不能跳过检查前往其他地点")
            if target == context["observation"]["current_location"] and target in remaining:
                raise SchemaError(f"已到达 {target} 但尚未检查，应该调用 inspect 而不是重复移动")
        if action["name"] == "inspect":
            if self.task["task_type"] != "inspect_locations" or target not in self.task["targets"]:
                raise SchemaError("本任务不要求检查该地点")
            done = context["progress"]["done_targets"]
            if target in done:
                raise SchemaError("本任务已经检查过该地点，不能重复检查")
            remaining = [p for p in self.task["targets"] if p not in done]
            if self.task["constraints"]["visit_in_order"] and target != remaining[0]:
                raise SchemaError(f"顺序约束要求先检查 {remaining[0]}")

    def _save_snapshot(self):
        if self.output:
            temporary = self.output / "snapshot.tmp"
            temporary.write_text(json.dumps(self.snapshot(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            temporary.replace(self.output / "snapshot.json")

    def _run(self, cancel):
        failures, invalid, stagnant, rounds_this_run = {}, 0, 0, 0
        try:
            if not self.initial_plan:
                for attempt in range(self.max_retries + 1):
                    context = self._context()
                    try:
                        plan = validate_plan(self._model_call(lambda: self.planner.plan(context), cancel))
                        with self._lock:
                            if cancel.is_set():
                                raise InterruptedError()
                            self.initial_plan = plan
                            self._emit("initial_plan", plan=plan)
                        break
                    except SchemaError as exc:
                        self._feedback_error(str(exc))
                        if attempt == self.max_retries:
                            raise RuntimeError("初步计划连续校验失败") from exc
            while not cancel.is_set():
                context = self._context()
                with self._lock:
                    if cancel.is_set():
                        break
                    if context["progress"]["completed"]:
                        self.state = "completed"
                        self._emit("completed", completion_check=context["progress"], observation=context["observation"])
                        return
                    if rounds_this_run >= self.max_rounds:
                        self.pause("达到本次运行的最大轮数；请检查日志后再恢复")
                        break
                    self.round += 1
                    rounds_this_run += 1
                context = self._context()
                try:
                    decision = validate_decision(self._model_call(lambda: self.planner.decide(context), cancel))
                    with self._lock:
                        if cancel.is_set() or self.state != "running":
                            break
                        self._emit("thinking", **decision)
                    if decision["request_pause"]:
                        self.pause(decision["request_pause"])
                        break
                    action = decision["next_action"]
                    if action is None:
                        raise SchemaError("模型声称完成，但环境尚未满足完成条件；需要继续执行或说明暂停原因")
                    self._check_scope(action, context)
                except SchemaError as exc:
                    if cancel.is_set():
                        break
                    invalid += 1
                    self._feedback_error(str(exc))
                    if invalid > self.max_retries:
                        self.pause("模型输出或动作参数连续校验失败")
                    continue
                with self._lock:
                    if cancel.is_set() or self.state != "running":
                        break
                    self._emit("action_started", action=action)
                # The backend must check this cancellation token again when admitting the action.
                result = self.backend.execute(action, task_id=self.task_id, cancel=cancel, timeout=self.action_timeout)
                observation = self.backend.get_status()
                if result["status"] == "succeeded" and action["name"] == "navigate_to":
                    self._reached.add(observation["current_location"])
                self.history.append(deepcopy(result))
                progress = self._progress(observation)
                self._emit("observation", result=result, observation=observation, progress=progress)
                self._save_snapshot()
                if cancel.is_set():
                    break
                if (result.get("error") or {}).get("code") == "timeout":
                    self._emit("timeout_state_checked", observation=observation)
                    self.pause("动作超时；已重新读取状态，需检查后恢复")
                    break
                if result["status"] == "cancelled" or action["name"] == "stop":
                    self.pause("动作已停止，等待人工恢复")
                    break
                key = json.dumps(action, sort_keys=True)
                if result["status"] == "failed":
                    failures[key] = failures.get(key, 0) + 1
                    if failures[key] > self.max_retries:
                        self.pause(f"动作连续失败，已用尽 {self.max_retries} 次重试：{result['error']}")
                else:
                    failures.pop(key, None)
                    invalid = 0
                advanced = (progress["done_targets"] != context["progress"]["done_targets"] or progress["completed"])
                stagnant = 0 if advanced else stagnant + 1
                if stagnant >= self.max_no_progress:
                    self.pause("多轮没有任务进展，已停止循环")
        except InterruptedError:
            pass
        except Exception as exc:
            self._emit("execution_error", error={"type": type(exc).__name__, "message": str(exc)})
            self.pause(str(exc))
        finally:
            with self._lock:
                completed = self.state == "completed"
            if not completed:
                stop_result = self.backend.stop()
                with self._lock:
                    observation = self.backend.get_status()
                    confirmed = (stop_result["status"] == "succeeded" and observation["active_action"] is None
                                 and observation["motion_state"] == "idle")
                    self.state = "paused" if confirmed else "pausing"
                    self.reason = self.reason or "执行已停止"
                    self._emit("paused" if confirmed else "stop_unconfirmed", reason=self.reason,
                               stop_result=stop_result, observation=observation, progress=self._progress(observation))
                    self._save_snapshot()
            else:
                self._save_snapshot()
