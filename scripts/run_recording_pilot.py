"""Run the prepared recordings through local ASR/noise/TTS experiments.

Run after prepare_recordings.py and model setup. Uses config.local.toml.
Only filename-derived, unconfirmed transcripts are available initially.
"""
from argparse import ArgumentParser, Namespace
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_lab.config import load_config
from voice_lab.experiments import benchmark_asr, benchmark_tts, make_noisy, write_csv
from voice_lab.pipeline import VoicePipeline


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("asr", "tts", "conversation"),
                        default=["asr", "tts", "conversation"])
    stages = parser.parse_args().stages
    os.chdir(ROOT)
    out = ROOT / "outputs/pilot"
    out.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/manifest.csv").open(encoding="utf-8-sig", newline="") as f:
        clean = list(csv.DictReader(f))
    languages = {Path(r["path"]).stem: r["language"] for r in clean}
    transcript_status = {Path(r["path"]).stem: r.get("transcript_status", "unconfirmed") for r in clean}
    noisy = []
    for name in ("keyboard", "noise1", "noise2"):
        directory = out / "mixed" / name
        make_noisy(Namespace(manifest=ROOT / "data/manifest.csv", noise=ROOT / f"data/noise/{name}.wav",
                             snr=[20, 10, 0], output=directory))
        with (directory / "manifest.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row["condition"] == "clean":
                    continue
                row["language"] = languages[Path(row["path"]).stem.split("_")[1]]
                row["transcript_status"] = transcript_status[Path(row["path"]).stem.split("_")[1]]
                row["path"] = os.path.relpath(directory / row["path"], out).replace(os.sep, "/")
                noisy.append(row)
    write_csv(out / "noise_manifest.csv", noisy)
    for backend in (("whisper", "sensevoice") if "asr" in stages else ()):
        for label, manifest in (("clean", ROOT / "data/manifest.csv"), ("noise", out / "noise_manifest.csv")):
            print(f"Starting {backend}: {label}", flush=True)
            benchmark_asr(Namespace(config=ROOT / "config.local.toml", manifest=manifest, backend=backend,
                                    denoise=["off", "highpass", "spectral"], output=out / f"{backend}_{label}.csv"))
    for backend in (("vits", "kokoro") if "tts" in stages else ()):
        print(f"Starting TTS: {backend}", flush=True)
        benchmark_tts(Namespace(config=ROOT / "config.local.toml", backend=backend,
                                texts=ROOT / "data/tts_texts.txt", output=out / f"tts_{backend}"))
    if "conversation" not in stages:
        return
    config = load_config(ROOT / "config.local.toml")
    config["asr"]["backend"] = "sensevoice"
    pipeline = VoicePipeline(config)
    history, turns = [], []
    demo = out / "conversation"
    demo.mkdir(exist_ok=True)
    (demo / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    # Four actual Mandarin recordings plus a fifth text follow-up test the real chain and history.
    inputs = [(ROOT / "data" / r["path"], "") for r in clean if r["language"] in {"zh", "zh+en"}]
    inputs.append((None, "请用一句话总结我们刚才讨论的内容。"))
    for i, (audio, text) in enumerate(inputs, 1):
        print(f"Conversation turn {i}", flush=True)
        result = pipeline.run(audio_wav=audio.read_bytes() if audio else None, text=text, history=history)
        record = dict(turn=i, input_file=str(audio.relative_to(ROOT)) if audio else None,
                      input_text=text, transcript=result.transcript, answer=result.answer,
                      timings=result.timings, tts_error=result.tts_error)
        if result.audio_wav:
            (demo / f"reply_{i:02d}.wav").write_bytes(result.audio_wav)
        turns.append(record)
        history.extend([dict(role="user", content=result.transcript), dict(role="assistant", content=result.answer)])
        (demo / "turns.json").write_text(json.dumps(turns, ensure_ascii=False, indent=2), encoding="utf-8")
    if any(r["tts_error"] for r in turns):
        raise RuntimeError("Conversation had TTS errors; inspect conversation/turns.json")
    print("Pilot complete. Transcripts and listening quality still require human review.", flush=True)


if __name__ == "__main__":
    main()
