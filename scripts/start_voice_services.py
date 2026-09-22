"""Start local voice services using the GPT-SoVITS bundle's CUDA runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "vitsModel/GPT-SoVITS-v2pro-20250604-nvidia50"
OUT = ROOT / "outputs/voice_upgrade"


def healthy(url, path):
    try:
        with urlopen(url + "/openapi.json", timeout=2) as r:
            return path in json.load(r).get("paths", {})
    except (OSError, ValueError):
        return False


def start(name, cwd, args, url, endpoint):
    if healthy(url, endpoint):
        print(f"Already running: {name} at {url}", flush=True)
        return
    env = dict(os.environ)
    env.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    startup = subprocess.STARTUPINFO()
    startup.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startup.wShowWindow = 0
    with (OUT / f"{name}.stdout.log").open("w") as stdout, (OUT / f"{name}.stderr.log").open("w") as stderr:
        p = subprocess.Popen([str(BUNDLE / "runtime/python.exe"), *args], cwd=cwd, env=env,
                             stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
                             startupinfo=startup, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
    (OUT / f"{name}.pid").write_text(str(p.pid))
    for _ in range(45):
        if healthy(url, endpoint):
            print(f"Ready: {name} at {url}; PID {p.pid}", flush=True)
            return
        if p.poll() is not None:
            raise RuntimeError(f"{name} exited ({p.returncode}); inspect {OUT / (name + '.stderr.log')}")
        time.sleep(1)
    print(f"{name} still loading; PID {p.pid}. Check logs before sending audio.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpt-sovits", action="store_true")
    parser.add_argument("--rvc", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    both = not args.gpt_sovits and not args.rvc
    if args.gpt_sovits or both:
        start("gpt_sovits", BUNDLE, ["api_v2.py", "-a", "127.0.0.1", "-p", "9880", "-c", str(ROOT / "config.gpt-sovits-v4.yaml")], "http://127.0.0.1:9880", "/tts")
    if args.rvc or both:
        start("rvc", ROOT, [str(ROOT / "scripts/rvc_service.py")], "http://127.0.0.1:9881", "/convert")


if __name__ == "__main__":
    main()
