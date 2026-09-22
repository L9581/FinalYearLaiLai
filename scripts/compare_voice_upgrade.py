"""Generate v4 speech and RVC variants of the same six evaluation texts."""
import argparse
import csv
import html
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_lab.audio import read_wav, wav_bytes
from voice_lab.backends import GPTSoVITSTTS, RvcTTS, create_asr
from voice_lab.config import load_config
from voice_lab.metrics import character_errors

OUT = ROOT / "outputs/voice_upgrade"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["tts", "rvc", "check", "page", "all"], default="all")
    stage = parser.parse_args().stage
    config = load_config(ROOT / "config.voice-upgrade.toml")
    source = GPTSoVITSTTS(config["tts"])
    converter = RvcTTS(source, config["tts"]["rvc"])
    texts = [s.strip() for s in (ROOT / "data/tts_texts.txt").read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    OUT.mkdir(parents=True, exist_ok=True)
    if stage in {"tts", "all"}:
        rows = []
        for i, text in enumerate(texts, 1):
            started = perf_counter()
            audio = source.synthesize(text)
            seconds = perf_counter() - started
            name = f"v4_{i:03d}.wav"
            (OUT / name).write_bytes(wav_bytes(audio))
            rows.append(dict(file=name, text=text, seconds=seconds, sample_rate=audio.sample_rate, audio_seconds=audio.seconds))
            (OUT / "v4_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"v4 {i}/{len(texts)}: {audio.sample_rate} Hz, {seconds:.2f}s", flush=True)
        (OUT / "v4_config.json").write_text(json.dumps(config["tts"]["gpt_sovits"], ensure_ascii=False, indent=2), encoding="utf-8")
    if stage in {"rvc", "all"}:
        rows = []
        for i, text in enumerate(texts, 1):
            original = read_wav(OUT / f"v4_{i:03d}.wav", max_seconds=180, resample=False)
            started = perf_counter()
            audio = converter.convert(original)
            seconds = perf_counter() - started
            name = f"v4_rvc_{i:03d}.wav"
            (OUT / name).write_bytes(wav_bytes(audio))
            rows.append(dict(file=name, text=text, seconds=seconds, sample_rate=audio.sample_rate, audio_seconds=audio.seconds))
            (OUT / "rvc_results.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"RVC {i}/{len(texts)}: {audio.sample_rate} Hz, {seconds:.2f}s", flush=True)
        (OUT / "rvc_config.json").write_text(json.dumps(config["tts"]["rvc"], ensure_ascii=False, indent=2), encoding="utf-8")
    if stage in {"check", "all"}:
        recognizer = create_asr(config["asr"])
        checks = []
        for backend, pattern in [("kokoro", "../pilot/tts_kokoro/{i:03d}.wav"), ("v4", "v4_{i:03d}.wav"), ("v4_rvc", "v4_rvc_{i:03d}.wav")]:
            for i, text in enumerate(texts, 1):
                hypothesis = recognizer.transcribe(read_wav(OUT / pattern.format(i=i)))
                errors, count = character_errors(text, hypothesis)
                checks.append(dict(backend=backend, text=text, hypothesis=hypothesis, errors=errors,
                                   characters=count, cer=errors/count))
        with (OUT / "pronunciation_check.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)
        print("Saved automatic transcription checks; human listening remains necessary.")
    if stage in {"page", "all"}:
        cards = []
        for i, text in enumerate(texts, 1):
            samples = [("原来的 Kokoro", f"../pilot/tts_kokoro/{i:03d}.wav"),
                       ("GPT-SoVITS v4（48 kHz）", f"v4_{i:03d}.wav"),
                       ("GPT-SoVITS v4 + 纳西妲 RVC（40 kHz）", f"v4_rvc_{i:03d}.wav")]
            for _, filename in samples:
                if not (OUT / filename).is_file():
                    raise FileNotFoundError(OUT / filename)
            players = ''.join(f'<label>{html.escape(label)}<audio controls preload="none" src="{filename}"></audio></label>' for label, filename in samples)
            cards.append(f'<article><h2>第 {i} 组</h2><p>{html.escape(text)}</p>{players}</article>')
        page = '<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>新声音试听</title><style>body{font:17px/1.6 system-ui;max-width:880px;margin:40px auto;padding:0 20px;background:#f4f6fa;color:#172033}article{padding:22px;background:white;border-radius:12px;margin:20px 0}label{display:block;margin-top:16px}audio{display:block;width:100%}h2{font-size:20px}</style><h1>新声音试听</h1><p>每组使用相同文字。先听发音、漏字和自然度，再比较是否喜欢音色。</p><p>v4 当前使用已有的合成女声作为参考；RVC 使用你提供的纳西妲模型。基础发音和最终音色都需要听评确认。</p>' + ''.join(cards) + '</html>'
        (OUT / "compare.html").write_text(page, encoding="utf-8")
        print(OUT / "compare.html")


if __name__ == "__main__":
    main()
