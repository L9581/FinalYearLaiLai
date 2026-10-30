"""Usage: python -m voice_lab.simulation {run,replay,tools}."""

import argparse
import json
from pathlib import Path
import sys

from .runner import Scenario, read_json, render_trace, run_scenario, save_trace, verify_trace
from .scenarios import SCENARIO_NAMES, make_scenario
from .tools import ToolRegistry


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="阶段一：人工任务计划的确定性仿真，无需模型服务。")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="执行内置或 JSON 场景并保存日志")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--scenario", choices=SCENARIO_NAMES, default="move-cup")
    source.add_argument("--scenario-file", type=Path)
    run.add_argument("--output", type=Path, help="输出目录；同目录再次运行会覆盖上次日志")
    replay = commands.add_parser("replay", help="重新执行 trace.json 并逐状态核对")
    replay.add_argument("trace", type=Path)
    commands.add_parser("tools", help="显示允许调用的工具及严格参数 schema")
    args = parser.parse_args(argv)
    try:
        if args.command == "tools":
            print(json.dumps(ToolRegistry().describe(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "replay":
            trace = verify_trace(read_json(args.trace))
            print(render_trace(trace))
            print("REPLAY VERIFIED：每一步状态、路径、结果及最终目标均一致。")
            # Reproducing an expected failed task is a successful replay verification.
            return 0
        scenario = (Scenario.from_dict(read_json(args.scenario_file)) if args.scenario_file
                    else make_scenario(args.scenario))
        trace = run_scenario(scenario)
        output = args.output if args.output is not None else Path("outputs/simulation") / scenario.id
        save_trace(trace, output)
        print(render_trace(trace))
        print(f"日志目录: {output.resolve()}")
        return 0 if trace.status == "completed" else 1
    except (OSError, ValueError) as exc:
        print(f"仿真输入或回放错误: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    # Windows redirects Python output using the local code page by default.
    # Trace files and CLI pipes use UTF-8 consistently, including Chinese messages.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    raise SystemExit(main())
