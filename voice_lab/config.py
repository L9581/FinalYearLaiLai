from pathlib import Path
import tomllib


def load_config(path: str | Path = "config.toml") -> dict:
    path = Path(path).resolve()
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    # All model paths are relative to the configuration file, not the shell cwd.
    for backend in ("vits", "kokoro"):
        for key, value in config["tts"][backend].items():
            if key in {"model", "voices", "tokens", "data_dir", "dict_dir"} and value:
                config["tts"][backend][key] = str((path.parent / value).resolve())
            elif key in {"rule_fsts", "lexicon"} and value:
                config["tts"][backend][key] = ",".join(
                    str((path.parent / item.strip()).resolve()) for item in value.split(",")
                )
    for key in ("whisper_model", "sensevoice_model", "sensevoice_onnx_model", "sensevoice_onnx_tokens"):
        if key not in config["asr"]:
            continue
        value = config["asr"][key]
        candidate = path.parent / value
        if candidate.exists() or value.startswith(("./", "../", ".\\", "..\\", "models/")):
            config["asr"][key] = str(candidate.resolve())
    if "gpt_sovits" in config["tts"]:
        settings = config["tts"]["gpt_sovits"]
        settings["reference_audio"] = str((path.parent / settings["reference_audio"]).resolve())
    return config
