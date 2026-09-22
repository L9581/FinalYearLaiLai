"""Run with python -m voice_lab.experiments --help."""
import argparse
import csv
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
from time import perf_counter

from .audio import denoise, mix_at_snr, read_wav, wav_bytes
from .backends import create_tts, create_asr
from .config import load_config
from .metrics import character_errors


def read_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("数据清单为空。")
    for row in rows:
        if not row.get("path") or not row.get("text"):
            raise ValueError("清单每行必须包含非空 path 和 text。")
        character_errors(row["text"], "")
        row["path"] = str((path.resolve().parent / row["path"]).resolve())
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
        row["condition"] = row.get("condition") or "unspecified"
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def metadata(config: dict) -> dict:
    packages = {}
    for name in ("faster-whisper", "funasr", "sherpa-onnx", "noisereduce", "numpy", "scipy"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    return {"config": config, "platform": platform.platform(), "python": platform.python_version(), "packages": packages}


def benchmark_asr(args):
    config = load_config(args.config)
    config["asr"]["backend"] = args.backend or config["asr"]["backend"]
    rows = read_manifest(args.manifest)
    asr = create_asr(config["asr"])
    started = perf_counter()
    asr.load()
    load_seconds = perf_counter() - started
    # Warm up the loaded model; download/load/warmup are excluded from inference RTF.
    asr.transcribe(read_wav(rows[0]["path"]))
    output = []
    for method in args.denoise:
        for row in rows:
            audio = read_wav(row["path"])
            started = perf_counter()
            processed = denoise(audio, method, config["audio"]["noise_reduction"])
            preprocess_seconds = perf_counter() - started
            started = perf_counter()
            hypothesis = asr.transcribe(processed)
            inference_seconds = perf_counter() - started
            errors, characters = character_errors(row["text"], hypothesis)
            output.append({
                "path": row["path"], "condition": row["condition"], "backend": config["asr"]["backend"],
                "denoise": method, "reference": row["text"], "hypothesis": hypothesis,
                "errors": errors, "reference_characters": characters, "cer": errors / characters,
                "audio_seconds": audio.seconds, "preprocess_seconds": preprocess_seconds,
                "inference_seconds": inference_seconds, "rtf": inference_seconds / audio.seconds,
            })
            print(f"{method:8s} CER={errors / characters:.3f} RTF={inference_seconds / audio.seconds:.3f} {Path(row['path']).name}")
    write_csv(args.output, output)
    groups = []
    for method, condition in sorted({(row["denoise"], row["condition"]) for row in output}):
        selected = [row for row in output if row["denoise"] == method and row["condition"] == condition]
        groups.append({
            "denoise": method, "condition": condition, "samples": len(selected),
            "cer": sum(row["errors"] for row in selected) / sum(row["reference_characters"] for row in selected),
            "rtf": sum(row["inference_seconds"] for row in selected) / sum(row["audio_seconds"] for row in selected),
        })
    details = metadata(config) | {"load_seconds": load_seconds, "warmup_utterances": 1, "groups": groups}
    args.output.with_suffix(".json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


def make_noisy(args):
    rows = read_manifest(args.manifest)
    noise = read_wav(args.noise, max_seconds=600)
    args.output.mkdir(parents=True, exist_ok=True)
    output = []
    for index, row in enumerate(rows, 1):
        clean = read_wav(row["path"])
        output.append({"path": os.path.relpath(row["path"], args.output.resolve()), "text": row["text"], "condition": "clean"})
        for snr in args.snr:
            filename = f"{index:04d}_{Path(row['path']).stem}_{snr:g}db.wav"
            (args.output / filename).write_bytes(wav_bytes(mix_at_snr(clean, noise, snr)))
            output.append({"path": filename, "text": row["text"], "condition": f"{args.noise.stem}_{snr:g}dB"})
    write_csv(args.output / "manifest.csv", output)
    (args.output / "recipe.json").write_text(json.dumps({
        "source_manifest": str(args.manifest.resolve()), "noise": str(args.noise.resolve()),
        "snr_db": args.snr, "method": "whole-clip RMS; noise repeated from start; common peak scaling",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.output / "manifest.csv")


def benchmark_tts(args):
    config = load_config(args.config)
    config["tts"]["backend"] = args.backend or config["tts"]["backend"]
    texts = [line.strip() for line in args.texts.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not texts:
        raise ValueError("测试文本文件为空。")
    tts = create_tts(config["tts"])
    started = perf_counter()
    tts.load()
    load_seconds = perf_counter() - started
    tts.synthesize(texts[0])
    args.output.mkdir(parents=True, exist_ok=True)
    output = []
    for index, text in enumerate(texts, 1):
        started = perf_counter()
        audio = tts.synthesize(text)
        seconds = perf_counter() - started
        filename = f"{index:03d}.wav"
        (args.output / filename).write_bytes(wav_bytes(audio))
        output.append({"file": filename, "text": text, "backend": config["tts"]["backend"],
                       "generation_seconds": seconds, "audio_seconds": audio.seconds, "rtf": seconds / audio.seconds})
        print(f"{filename} RTF={seconds / audio.seconds:.3f}")
    write_csv(args.output / "results.csv", output)
    details = metadata(config) | {"load_seconds": load_seconds, "warmup_utterances": 1}
    (args.output / "metadata.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="本地语音模型与噪声实验工具")
    sub = parser.add_subparsers(dest="command", required=True)
    asr = sub.add_parser("asr", help="比较 ASR 的字符错误率和实时率")
    asr.add_argument("--config", default="config.toml")
    asr.add_argument("--manifest", required=True, type=Path)
    asr.add_argument("--backend", choices=["whisper", "sensevoice"])
    asr.add_argument("--denoise", nargs="+", choices=["off", "highpass", "spectral"], default=["off"])
    asr.add_argument("--output", type=Path, default=Path("outputs/asr.csv"))
    asr.set_defaults(run=benchmark_asr)
    noisy = sub.add_parser("mix-noise", help="用真实环境噪声生成指定 SNR 的 WAV 数据集")
    noisy.add_argument("--manifest", required=True, type=Path)
    noisy.add_argument("--noise", required=True, type=Path)
    noisy.add_argument("--snr", nargs="+", type=float, default=[20, 10, 0])
    noisy.add_argument("--output", type=Path, default=Path("outputs/noisy"))
    noisy.set_defaults(run=make_noisy)
    tts = sub.add_parser("tts", help="保存 TTS 样音和生成耗时，供人工听评")
    tts.add_argument("--config", default="config.toml")
    tts.add_argument("--backend", choices=["vits", "kokoro", "gpt_sovits"])
    tts.add_argument("--texts", type=Path, default=Path("data/tts_texts.txt"))
    tts.add_argument("--output", type=Path, default=Path("outputs/tts"))
    tts.set_defaults(run=benchmark_tts)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"实验失败：{exc}\n")


if __name__ == "__main__":
    main()
