"""python -m voice_lab.robot_agent: parse, run, demo, validate, tools, evaluate."""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Thread
from time import sleep
import tomllib
from uuid import uuid4

from .backend import SimulatedBackend
from .controller import TaskController
from .model import DemoPlanner, LLMPlanner, OllamaJSON, TaskParser
from .schemas import ACTION_SCHEMAS, LOCATIONS, read_json, validate_task


def dump(value):
    return json.dumps(value, ensure_ascii=False, indent=2)


def transcribe_wav(path, config_path):
    # Import the existing audio stack only for voice input. Text/demo uses stdlib only.
    from voice_lab.audio import denoise, read_wav
    from voice_lab.backends import create_asr
    from voice_lab.config import load_config
    config = load_config(config_path)
    audio = read_wav(path.read_bytes(), config["audio"]["max_seconds"])
    processed = denoise(audio, config["audio"]["denoise"], config["audio"]["noise_reduction"])
    return create_asr(config["asr"]).transcribe(processed).strip()


def monitor(controller, *, interactive=True):
    commands = Queue()
    if interactive:
        print("运行控制：输入 p 暂停、r 恢复、s 查看状态、q 停止并退出，然后按回车。Ctrl+C 也会暂停。", flush=True)

        def read_commands():
            # input() holds stdin's buffered-reader lock while blocked. A daemon
            # using it can crash Python shutdown after the task finishes. Raw fd
            # reads leave no Python buffered-stream lock behind.
            pending = bytearray()
            while True:
                try:
                    char = os.read(sys.stdin.fileno(), 1)
                except (OSError, ValueError):
                    char = b""
                if not char:
                    commands.put("eof")
                    return
                if char == b"\n":
                    commands.put(pending.decode("ascii", errors="ignore").strip().lower())
                    pending.clear()
                else:
                    pending.extend(char)

        Thread(target=read_commands, daemon=True).start()
    cursor, quitting, eof = 0, False, False
    while True:
        try:
            for event in controller.events_since(cursor):
                cursor += 1
                kind = event["kind"]
                if kind == "initial_plan":
                    print("初步计划：" + " → ".join(event["plan"]), flush=True)
                elif kind == "thinking":
                    print(f"[{event['round']}] Thinking：{event['decision_summary']}", flush=True)
                elif kind == "action_started":
                    print("  Action：" + json.dumps(event["action"], ensure_ascii=False), flush=True)
                elif kind == "observation":
                    result = event["result"]
                    print(f"  Observation：{result['status']}，位置={event['observation']['current_location']}，"
                          f"已完成={event['progress']['done_targets']}，错误={result['error']}", flush=True)
                elif kind in ("pausing", "paused", "resumed", "completed", "validation_error", "execution_error", "stop_unconfirmed"):
                    print(json.dumps(event, ensure_ascii=False), flush=True)
            snapshot = controller.snapshot()
            if snapshot["state"] == "completed":
                controller.wait(2)
                print("完成条件已由程序核对通过。", flush=True)
                return 0
            if snapshot["state"] == "paused" and (not interactive or quitting or eof):
                controller.wait(2)
                return 1
            try:
                command = commands.get_nowait()
            except Empty:
                command = None
            if command == "p":
                controller.pause()
            elif command == "r":
                try:
                    controller.resume()
                except RuntimeError as exc:
                    print(str(exc), flush=True)
            elif command == "s":
                print(dump(snapshot), flush=True)
            elif command == "q":
                quitting = True
                controller.pause("人工退出；进度已保存")
            elif command == "eof":
                eof = True
            sleep(0.05)
        except KeyboardInterrupt:
            controller.pause()
            if not interactive:
                quitting = True


def evaluate(client, samples_path, output):
    samples = read_json(samples_path.read_text(encoding="utf-8-sig"))
    results = []
    for sample in samples:
        parser = TaskParser(client)
        try:
            for text in sample["inputs"]:
                result = parser.parse(text)
            actual = {"status": result["status"]}
            if result["status"] == "ready":
                task = result["task"]
                actual.update(task_type=task["task_type"], targets=task["targets"], **task["constraints"])
            passed = actual == sample["expected"]
            row = {**sample, "actual": actual, "passed": passed, "turns": parser.turns}
        except Exception as exc:
            row = {**sample, "passed": False, "error": str(exc), "turns": parser.turns}
        results.append(row)
        print(f"{sample['id']}: {'PASS' if row['passed'] else 'FAIL'}", flush=True)
        output.mkdir(parents=True, exist_ok=True)
        (output / "evaluation.json").write_text(dump({"model": client.config["model"], "results": results,
            "passed": sum(r["passed"] for r in results), "total": len(results)}) + "\n", encoding="utf-8")
    print(f"{sum(r['passed'] for r in results)}/{len(results)}；记录：{output.resolve()}")
    return 0 if all(r["passed"] for r in results) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="自然语言结构化 → 单 Agent → 模拟动作 → 反馈循环")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("tools", help="显示四个动作接口参数")
    validate = commands.add_parser("validate", help="校验结构化任务文件（task 或完整解析结果）")
    validate.add_argument("path", type=Path)
    for name in ("run", "parse", "demo", "evaluate"):
        sub = commands.add_parser(name)
        sub.add_argument("--output", type=Path)
        if name != "demo":
            sub.add_argument("--config", type=Path, default=Path("config.toml"))
            sub.add_argument("--model", help="覆盖配置中的 Ollama 模型名")
        if name in ("run", "parse"):
            sub.add_argument("text", nargs="?", help="自然语言任务")
            sub.add_argument("--wav", type=Path, help="使用已有 ASR 识别 WAV 后结构化")
            sub.add_argument("--non-interactive", action="store_true", help="需要澄清/暂停时退出，不等待输入")
        if name in ("run", "demo"):
            sub.add_argument("--start", choices=LOCATIONS, default="home")
            sub.add_argument("--delay", type=float, default=0.5)
            sub.add_argument("--fail-next", type=int, default=0, help="注入前 N 次移动失败")
            sub.add_argument("--abnormal", choices=LOCATIONS, help="指定一个地点的检查结果为异常")
            sub.add_argument("--max-rounds", type=int, default=20)
            sub.add_argument("--max-retries", type=int, default=2)
            sub.add_argument("--max-no-progress", type=int, default=6)
            sub.add_argument("--model-timeout", type=float, default=90)
            sub.add_argument("--action-timeout", type=float, default=30)
        if name == "demo":
            sub.add_argument("--non-interactive", action="store_true")
        if name == "evaluate":
            sub.add_argument("--samples", type=Path, default=Path("data/agent_parse_examples.json"))
    args = parser.parse_args(argv)
    output = getattr(args, "output", None) or Path("outputs/robot-agent") / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    try:
        if args.command == "tools":
            print(dump(ACTION_SCHEMAS))
            return 0
        if args.command == "validate":
            data = read_json(args.path.read_text(encoding="utf-8-sig"))
            if isinstance(data, dict) and "status" in data:
                from .schemas import validate_parse
                result = validate_parse(data)
                print(dump(result))
                return 0 if result["status"] == "ready" else 2
            print(dump(validate_task(data)))
            return 0
        turns = []
        if args.command == "demo":
            task = validate_task({"task_type": "inspect_locations", "targets": ["point_a", "point_b"],
                                  "constraints": {"visit_in_order": True, "return_to_start": True}})
            text = "人工定义的离线示例：依次检查 A、B 并返回出发点"
            planner = DemoPlanner()
            print("离线演示：任务为人工定义，决策使用规则替身，不代表 LLM 理解能力。", flush=True)
        else:
            with args.config.open("rb") as file:
                config = tomllib.load(file)["llm"]
            if args.model:
                config["model"] = args.model
            client = OllamaJSON(config)
            if args.command == "evaluate":
                return evaluate(client, args.samples, output)
            if bool(args.text) == bool(args.wav):
                raise ValueError("请提供文字任务或 --wav 文件，二者选一")
            text = transcribe_wav(args.wav, args.config) if args.wav else args.text
            task_parser = TaskParser(client)

            def parse_and_save(value, source="text"):
                try:
                    return task_parser.parse(value, source=source)
                finally:
                    output.mkdir(parents=True, exist_ok=True)
                    (output / "parse.json").write_text(dump(task_parser.turns) + "\n", encoding="utf-8")

            result = parse_and_save(text, source="asr" if args.wav else "text")
            while True:
                print(dump(result), flush=True)
                if result["status"] != "needs_clarification" or args.non_interactive:
                    break
                result = parse_and_save(input("请补充（输入完整的新任务可重新表述）："))
            if result["status"] != "ready":
                return 2
            if args.command == "parse":
                return 0
            task, turns = result["task"], task_parser.turns
            planner = LLMPlanner(client)
        backend = SimulatedBackend(args.start, args.delay, {args.abnormal: "abnormal"} if args.abnormal else None)
        if args.fail_next:
            backend.fail_next(count=args.fail_next)
        controller = TaskController(backend, planner, output=output, max_rounds=args.max_rounds,
                                    max_retries=args.max_retries, max_no_progress=args.max_no_progress,
                                    model_timeout=args.model_timeout, action_timeout=args.action_timeout)
        controller.start(task, original_text=text, parse_turns=turns)
        print(f"日志目录：{output.resolve()}", flush=True)
        return monitor(controller, interactive=not args.non_interactive)
    except (OSError, ValueError, RuntimeError, EOFError) as exc:
        print(f"未进入执行或运行失败：{exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("输入/解析阶段已取消，未开始动作。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
