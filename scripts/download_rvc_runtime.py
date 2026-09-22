"""Fetch the official RVC source and feature/pitch weights into this workspace."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import zipfile

import requests

ROOT = Path(__file__).resolve().parents[1]
REVISION = "81eed5e8f68b6bed1789f682fe78cdd324495afc"


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.is_file():
        temp = dest.with_suffix(dest.suffix + ".part")
        candidates = [url, url]
        if url.startswith("https://huggingface.co/"):
            candidates.append(url.replace("https://huggingface.co/", "https://hf-mirror.com/", 1))
        for index, candidate in enumerate(candidates):
            try:
                with requests.get(candidate, stream=True, timeout=(12, 90)) as response:
                    response.raise_for_status()
                    with temp.open("wb") as f:
                        for block in response.iter_content(1024 * 1024):
                            f.write(block)
                url = candidate
                break
            except requests.RequestException as exc:
                print(f"Retry {dest.name}: {type(exc).__name__}", flush=True)
                if index == len(candidates) - 1:
                    raise
        temp.replace(dest)
    with dest.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    print(f"Ready: {dest.relative_to(ROOT)} ({dest.stat().st_size:,} bytes)", flush=True)
    return dict(url=url, path=str(dest.relative_to(ROOT)), sha256=sha, bytes=dest.stat().st_size)


def main():
    archive = ROOT / "vendor/rvc-source.zip"
    jobs = [
        (f"https://codeload.github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/zip/{REVISION}", archive),
        ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base/config.json", ROOT / "vendor/rvc_webui/assets/hubert_base/config.json"),
        ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base/preprocessor_config.json", ROOT / "vendor/rvc_webui/assets/hubert_base/preprocessor_config.json"),
        ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base/pytorch_model.bin", ROOT / "vendor/rvc_webui/assets/hubert_base/pytorch_model.bin"),
        ("https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt", ROOT / "models/rvc/rmvpe.pt"),
    ]
    records = [fetch(*jobs[0])]
    target = ROOT / "vendor/rvc_webui"
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            relative = Path(*Path(info.filename).parts[1:])
            if not relative.parts or info.is_dir():
                continue
            dest = (target / relative).resolve()
            if not dest.is_relative_to(target.resolve()):
                raise ValueError(f"Invalid archive member: {info.filename}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                dest.write_bytes(z.read(info))
    print("RVC source extracted; review source before running.", flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        records.extend(pool.map(lambda args: fetch(*args), jobs[1:]))
    (ROOT / "models/rvc/sources.json").write_text(json.dumps(dict(revision=REVISION, files=records), indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
