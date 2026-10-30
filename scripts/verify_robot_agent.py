"""Reproducible simulation acceptance; --live also exercises the local Ollama model."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from threading import Event
from time import monotonic, sleep
import tomllib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from voice_lab.robot_agent.backend import SimulatedBackend
from voice_lab.robot_agent.controller import TaskController
from voice_lab.robot_agent.model import DemoPlanner, LLMPlanner, OllamaJSON, TaskParser
from voice_lab.robot_agent.schemas import validate_task


def wait_for(predicate, seconds=5):
    deadline = monotonic() + seconds
    while monotonic() < deadline:
        if predicate():
            return
        sleep(0.01)
    raise RuntimeError("等待控制测试的同步点超时")


def make_task(targets=("point_a",), *, kind="inspect_locations", returns=False):
    return validate_task({"task_type": kind, "targets": list(targets),
                          "constraints": {"visit_in_order": len(targets) > 1, "return_to_start": returns}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions 必须 >= 1")
    output = args.output or Path("outputs/robot-agent") / ("acceptance-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
    output.mkdir(parents=True, exist_ok=False)
    results = []

    def record(name, controller, expected):
        snapshot = controller.wait(120 if args.live else 5)
        passed = snapshot["state"] == expected
        if expected == "completed":
            passed = passed and snapshot["progress"]["completed"]
        row = {"case": name, "expected": expected, "passed": passed, "snapshot": snapshot,
               "log": str((output / name / "events.jsonl").resolve())}
        results.append(row)
        print(f"{name}: {'PASS' if passed else 'FAIL'} ({snapshot['state']})", flush=True)
        (output / "report.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    cases = [
        ("01-navigate", make_task(kind="navigate"), "home", 0, "completed"),
        ("02-inspect", make_task(), "home", 0, "completed"),
        ("03-ordered", make_task(("point_a", "point_b")), "home", 0, "completed"),
        ("04-return", make_task(("point_a", "point_b"), returns=True), "home", 0, "completed"),
        ("05-return-to-b", make_task(returns=True), "point_b", 0, "completed"),
        ("08-single-failure", make_task(), "home", 1, "completed"),
        ("09-repeated-failure", make_task(), "home", 10, "paused"),
    ]
    for name, task, start, fail_count, expected in cases:
        backend = SimulatedBackend(start, delay=0.01)
        if fail_count:
            backend.fail_next(count=fail_count)
        controller = TaskController(backend, DemoPlanner(), output=output / name)
        controller.start(task, original_text="人工结构化任务（控制逻辑验收）")
        record(name, controller, expected)

    backend = SimulatedBackend("point_b", delay=1)
    controller = TaskController(backend, DemoPlanner(), output=output / "10-motion-pause-resume")
    controller.start(make_task(returns=True), original_text="移动中暂停并恢复")
    wait_for(lambda: backend.get_status()["motion_state"] == "moving")
    controller.pause()
    record("10-motion-pause-resume", controller, "paused")
    backend.delay = 0.01
    controller.resume()
    record("10-motion-pause-resume", controller, "completed")

    entered, release = Event(), Event()

    class SlowPlanner(DemoPlanner):
        def decide(self, context):
            if not entered.is_set():
                entered.set()
                release.wait(10)
            return super().decide(context)

    controller = TaskController(SimulatedBackend(delay=0), SlowPlanner(), output=output / "11-model-pause-resume")
    controller.start(make_task(), original_text="等待模型时暂停；迟到回复被丢弃")
    wait_for(entered.is_set)
    controller.pause()
    record("11-model-pause-resume", controller, "paused")
    controller.resume()
    record("11-model-pause-resume", controller, "completed")
    release.set()

    entered, release = Event(), Event()

    class BetweenPlanner(DemoPlanner):
        def decide(self, context):
            if context["progress"]["done_targets"] == ["point_a"] and not entered.is_set():
                entered.set()
                release.wait(10)
            return super().decide(context)

    controller = TaskController(SimulatedBackend(delay=0), BetweenPlanner(), output=output / "12-between-pause-resume")
    controller.start(make_task(("point_a", "point_b"), returns=True), original_text="已检查 A 后暂停；恢复不重复检查")
    wait_for(entered.is_set)
    controller.pause()
    record("12-between-pause-resume", controller, "paused")
    controller.resume()
    record("12-between-pause-resume", controller, "completed")
    release.set()

    if args.live:
        with args.config.open("rb") as file:
            client = OllamaJSON(tomllib.load(file)["llm"])
        live_cases = [
            ("normal", "先去 A 点检查，再去 B 点检查，最后回到出发点", "home", 0),
            ("return-to-b", "检查 A 点，完成后回来", "point_b", 0),
            ("single-failure", "检查 A 点", "home", 1),
        ]
        for i in range(args.repetitions):
            for suffix, text, start, failures in live_cases:
                name = f"live-{suffix}-{i + 1}"
                task_parser = TaskParser(client)
                parsed = task_parser.parse(text)
                if parsed["status"] != "ready":
                    raise RuntimeError(f"{name} 未被模型解析为可执行任务: {parsed}")
                backend = SimulatedBackend(start, delay=0.01)
                if failures:
                    backend.fail_next(count=failures)
                controller = TaskController(backend, LLMPlanner(client), output=output / name)
                controller.start(parsed["task"], original_text=text, parse_turns=task_parser.turns)
                record(name, controller, "completed")
    print(f"验收：{sum(r['passed'] for r in results)}/{len(results)}；{output.resolve()}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
