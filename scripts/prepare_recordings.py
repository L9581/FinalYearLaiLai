"""Prepare this project's filename-labelled recordings without changing originals.

Run from the project root with .venv/Scripts/python.exe scripts/prepare_recordings.py.
Filename transcripts remain provisional until checked by the speaker.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from voice_lab.audio import read_wav, wav_bytes


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    audit, review, manifest, noises = [], [], [], []
    seen = {}
    (ROOT / "data/recordings").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/noise").mkdir(parents=True, exist_ok=True)
    for path in sorted((ROOT / "wav_docs").rglob("*.wav")):
        source = path.relative_to(ROOT).as_posix()
        sr, raw = wavfile.read(path)
        if np.issubdtype(raw.dtype, np.signedinteger):
            samples = raw.astype(np.float64) / float(-np.iinfo(raw.dtype).min)
        elif raw.dtype == np.uint8:
            samples = (raw.astype(np.float64) - 128) / 128
        else:
            samples = raw.astype(np.float64)
        digest = hashlib.sha256(str(sr).encode() + str(raw.shape).encode() + raw.tobytes()).hexdigest()
        is_noise = path.parent.name.lower() == "noise"
        duplicate = seen.get(digest, "")
        seen.setdefault(digest, source)
        peak = float(np.max(np.abs(samples)))
        rms = float(np.sqrt(np.mean(samples ** 2)))
        audit.append(dict(source=source, kind="noise" if is_noise else "speech",
                          seconds=round(len(raw) / sr, 4), sample_rate=sr,
                          channels=1 if raw.ndim == 1 else raw.shape[1], dtype=str(raw.dtype),
                          peak_dbfs=round(20 * np.log10(max(peak, 1e-12)), 2),
                          rms_dbfs=round(20 * np.log10(max(rms, 1e-12)), 2),
                          near_full_scale_samples=int(np.sum(np.abs(samples) >= .999)),
                          duplicate_of=duplicate, pcm_sha256=digest))
        audio = read_wav(path, max_seconds=600 if is_noise else 60)
        if is_noise:
            target = ROOT / "data/noise" / path.name
            target.write_bytes(wav_bytes(audio))
            noises.append(dict(path=f"noise/{path.name}", source=source,
                               label="keyboard" if path.stem == "keyboard" else "unconfirmed",
                               seconds=audio.seconds))
            continue
        text = re.sub(r"^(?:CN\+EN|CN\+JP|EN\+JP)_?", "", path.stem)
        if path.parent.name == "EN":
            text = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text)
            text = text[0] + text[1:].lower()
            language = "en"
        elif path.stem.startswith("CN+JP"):
            language = "zh+ja"
        elif path.stem.startswith("EN+JP"):
            text = text.replace("StudyPlan", "Study Plan")
            language = "en+ja"
        elif "Python" in text:
            language = "zh+en"
        else:
            language = "zh"
        target_name = ""
        if not duplicate:
            target_name = f"recordings/{len(manifest) + 1:03d}.wav"
            (ROOT / "data" / target_name).write_bytes(wav_bytes(audio))
            manifest.append(dict(path=target_name, text=text, condition=f"clean_{language}",
                                 language=language, transcript_status="filename_unconfirmed"))
        review.append(dict(source=source, prepared_path=target_name, text=text, language=language,
                           transcript_status="filename_unconfirmed", duplicate_of=duplicate))
    write_csv(ROOT / "data/manifest.csv", manifest)
    write_csv(ROOT / "data/manifest_zh.csv", [r for r in manifest if r["language"] in {"zh", "zh+en"}])
    write_csv(ROOT / "data/transcript_review.csv", review)
    write_csv(ROOT / "data/noise_manifest.csv", noises)
    write_csv(ROOT / "outputs/recording_audit.csv", audit)
    summary = dict(source_speech_files=len(review), unique_speech_files=len(manifest),
                   noise_files=len(noises), unique_speech_seconds=sum(
                       r["seconds"] for r in audit if r["kind"] == "speech" and not r["duplicate_of"]),
                   transcript_status="filename_unconfirmed", conversion="16 kHz mono PCM16; no denoising or gain adjustment")
    (ROOT / "outputs/recording_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
