import argparse
from importlib.util import find_spec
import json
from pathlib import Path
from urllib.request import urlopen

from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="检查当前配置所需的依赖、TTS 文件和 Ollama")
    parser.add_argument("--config", default="config.toml")
    parser.add_argument("--probe-ollama", action="store_true", help="请求配置中的 Ollama /api/tags")
    parser.add_argument("--probe-voice-services", action="store_true", help="检查 GPT-SoVITS / RVC 本机服务")
    args = parser.parse_args()
    config = load_config(args.config)
    okay = True
    modules = ["streamlit", "numpy", "scipy", "sherpa_onnx"]
    if config["asr"]["backend"] == "whisper":
        modules.append("faster_whisper")
    elif config["asr"].get("sensevoice_runtime", "funasr") != "sherpa-onnx":
        modules.append("funasr")
    if config["audio"]["denoise"] == "spectral":
        modules.append("noisereduce")
    for module in modules:
        found = find_spec(module) is not None
        print(f"{'OK' if found else 'MISSING'} dependency: {module}")
        okay &= found
    if config["asr"]["backend"] == "sensevoice" and config["asr"].get("sensevoice_runtime") == "sherpa-onnx":
        for key in ("sensevoice_onnx_model", "sensevoice_onnx_tokens"):
            value = config["asr"].get(key, "")
            found = bool(value) and Path(value).is_file()
            print(f"{'OK' if found else 'MISSING'} ASR {key}: {value}")
            okay &= found
    for key, value in config["tts"][config["tts"]["backend"]].items():
        if key not in {"model", "tokens", "voices", "lexicon", "data_dir", "dict_dir", "rule_fsts", "reference_audio"} or not value:
            continue
        for name in value.split(",") if key in {"rule_fsts", "lexicon"} else [value]:
            found = Path(name).exists()
            print(f"{'OK' if found else 'MISSING'} TTS {key}: {name}")
            okay &= found
    if args.probe_voice_services:
        services = []
        if config["tts"]["backend"] == "gpt_sovits":
            services.append(("GPT-SoVITS", config["tts"]["gpt_sovits"]["base_url"], "/tts"))
        if config["tts"].get("rvc", {}).get("enabled"):
            services.append(("RVC", config["tts"]["rvc"]["base_url"], "/convert"))
        for name, url, endpoint in services:
            try:
                with urlopen(url.rstrip("/") + "/openapi.json", timeout=3) as response:
                    found = endpoint in json.load(response).get("paths", {})
                print(f"{'OK' if found else 'MISSING'} {name} service: {url}")
                okay &= found
            except (OSError, ValueError) as exc:
                print(f"MISSING {name}: {exc}")
                okay = False
    if args.probe_ollama:
        try:
            with urlopen(config["llm"]["base_url"].rstrip("/") + "/api/tags", timeout=5) as response:
                names = [item["name"] for item in json.load(response).get("models", [])]
            found = config["llm"]["model"] in names
            print(f"{'OK' if found else 'MISSING'} Ollama model: {config['llm']['model']}; installed: {names}")
            okay &= found
        except (OSError, ValueError, KeyError) as exc:
            print(f"MISSING Ollama: {exc}")
            okay = False
    else:
        print("NOT CHECKED: Ollama service (use --probe-ollama)")
    print("NOT CHECKED: real model inference; named ASR models may download weights on first use.")
    raise SystemExit(0 if okay else 1)


if __name__ == "__main__":
    main()
