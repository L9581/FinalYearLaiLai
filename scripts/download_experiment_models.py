"""Download pinned ASR weights and official sherpa TTS release packages locally."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tarfile

import requests

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
WHISPER_REV = "536b0662742c02347bc0e980a01041f333bce120"
SENSEVOICE_REV = "2365baeacb507f821a0c8120fcee3d484dba7a07"


def download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        temp = dest.with_name(dest.name + ".part")
        with requests.get(url, stream=True, timeout=(20, 120)) as response:
            response.raise_for_status()
            with temp.open("wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
        temp.replace(dest)
    with dest.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    print(f"Ready: {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)", flush=True)
    return dict(path=dest.relative_to(ROOT).as_posix(), url=url, bytes=dest.stat().st_size, sha256=digest)


def hf_model(repo, revision, folder, filenames):
    return [download(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", MODELS / folder / name)
            for name in filenames]


def tts_package(name):
    archive = MODELS / (name + ".tar.bz2")
    entry = download(f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{archive.name}", archive)
    if not (MODELS / name / "model.onnx").is_file():
        with tarfile.open(archive) as f:
            f.extractall(MODELS, filter="data")
    print(f"Extracted: {name}", flush=True)
    return [entry]


def main():
    jobs = [
        lambda: hf_model("Systran/faster-whisper-small", WHISPER_REV, "faster-whisper-small",
                         ["config.json", "tokenizer.json", "vocabulary.txt", "README.md", "model.bin"]),
        lambda: hf_model("csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17", SENSEVOICE_REV,
                         "sensevoice-small-onnx", ["tokens.txt", "LICENSE", "README.md", "model.int8.onnx"]),
        lambda: tts_package("vits-icefall-zh-aishell3"),
        lambda: tts_package("kokoro-multi-lang-v1_1"),
    ]
    entries, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(job) for job in jobs]
        for future in futures:
            try:
                entries.extend(future.result())
            except Exception as e:
                errors.append(str(e))
                print(f"Download failed: {e}", flush=True)
    (MODELS / "sources.json").write_text(json.dumps(dict(files=entries, errors=errors), indent=2), encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
