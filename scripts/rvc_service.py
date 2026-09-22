"""Loopback-only RVC inference service; run with the GPT-SoVITS bundle runtime."""
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from threading import Lock
from time import perf_counter
from types import ModuleType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/rvc_webui"
sys.path.insert(0, str(VENDOR))
# The embedded GPT-SoVITS runtime has its own regular "tools" package.
# Bind this isolated process to RVC's tools namespace before importing RVC.
rvc_tools = ModuleType("tools")
rvc_tools.__path__ = [str(VENDOR / "tools")]
sys.modules["tools"] = rvc_tools
os.environ["rmvpe_root"] = str(ROOT / "models/rvc")
os.environ["RVC_CUDA_GRAPH"] = "0"

import faiss
from fastapi import FastAPI, HTTPException, Query, Request, Response
import librosa
import numpy as np
import soundfile as sf
from starlette.concurrency import run_in_threadpool
import torch
import uvicorn

from infer.hubert import load_hubert_model
from infer.module.models import SynthesizerTrnMs768NSFsid
from infer.rmvpe import RMVPE
from infer.vc.pipeline import Pipeline

settings = json.loads((ROOT / "config.rvc.json").read_text(encoding="utf-8"))
model_path = ROOT / settings["model"]
index_path = ROOT / settings["index"]
checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
if checkpoint.get("version") != "v2" or checkpoint.get("f0") != 1:
    raise ValueError("This service is configured for an RVC v2 pitch-guided voice model.")
model_config = list(checkpoint["config"])
model_config[-3] = checkpoint["weight"]["emb_g.weight"].shape[0]
sample_rate = int(model_config[-1])
device = settings["device"]
is_half = bool(settings["is_half"])
if device.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("RVC is configured for CUDA, but CUDA is unavailable.")
index = faiss.read_index(str(index_path))
if index.d != 768 or index.ntotal < 8:
    raise ValueError("The RVC v2 index must contain 768-dimensional feature vectors.")
index_vectors = int(index.ntotal)
del index
net = SynthesizerTrnMs768NSFsid(*model_config, is_half=is_half)
del net.enc_q
result = net.load_state_dict(checkpoint["weight"], strict=False)
if result.missing_keys:
    raise ValueError(f"RVC checkpoint is missing weights: {result.missing_keys}")
net.eval().to(device)
net = net.half() if is_half else net.float()
del checkpoint
hubert = load_hubert_model(device, is_half)
pipeline = Pipeline(sample_rate, SimpleNamespace(x_pad=1, x_query=6, x_center=38, x_max=41,
                                               is_half=is_half, device=device))
pipeline.model_rmvpe = RMVPE(str(ROOT / "models/rvc/rmvpe.pt"), is_half=is_half, device=device)
lock = Lock()
APP = FastAPI(title="Local RVC voice conversion")


@APP.get("/health")
def health():
    return dict(ready=True, voice=settings["voice_name"], version="v2", sample_rate=sample_rate,
                device=device, pitch_method="rmvpe", index_vectors=index_vectors)


def convert(data, pitch, index_rate, protect):
    try:
        samples, sr = sf.read(BytesIO(data), dtype="float32", always_2d=True)
    except Exception as exc:
        raise HTTPException(400, "Provide a valid WAV recording.") from exc
    if not 8000 <= sr <= 192000 or not 0.1 <= len(samples) / sr <= 180:
        raise HTTPException(400, "Audio must be 0.1–180 seconds at 8–192 kHz.")
    samples = samples.mean(axis=1)
    if not np.isfinite(samples).all() or np.max(np.abs(samples)) < 1e-7:
        raise HTTPException(400, "Audio is silent or contains invalid samples.")
    audio = librosa.resample(samples, orig_sr=sr, target_sr=16000) if sr != 16000 else samples
    audio = audio / max(float(np.max(np.abs(audio))) / .95, 1)
    started = perf_counter()
    with lock, torch.no_grad():
        output = pipeline.pipeline(hubert, net, int(settings["speaker_id"]), audio, [0, 0, 0],
                                   pitch, "rmvpe", str(index_path), index_rate, 1,
                                   sample_rate, 0, 1.0, "v2", protect)
    if not len(output) or not np.isfinite(output).all():
        raise RuntimeError("RVC returned empty or invalid audio.")
    buffer = BytesIO()
    sf.write(buffer, output, sample_rate, format="WAV", subtype="PCM_16")
    print(f"Converted {len(samples) / sr:.2f}s to {sample_rate} Hz in {perf_counter() - started:.2f}s; pitch={pitch}, index={index_rate}", flush=True)
    return buffer.getvalue()


@APP.post("/convert")
async def convert_endpoint(request: Request, pitch: int = Query(0, ge=-24, le=24),
                           index_rate: float = Query(.5, ge=0, le=1),
                           protect: float = Query(.33, ge=0, le=.5)):
    data = await request.body()
    if len(data) > 30 * 1024 * 1024:
        raise HTTPException(413, "WAV is limited to 30 MiB.")
    result = await run_in_threadpool(convert, data, pitch, index_rate, protect)
    return Response(result, media_type="audio/wav")


if __name__ == "__main__":
    print(f"Ready: {settings['voice_name']}, RVC v2, {sample_rate} Hz, {index_vectors} index vectors", flush=True)
    uvicorn.run(APP, host="127.0.0.1", port=9881)
