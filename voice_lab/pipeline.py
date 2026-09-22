from dataclasses import dataclass, field
from threading import RLock
from time import perf_counter
from typing import Callable

import numpy as np

from .audio import denoise, read_wav, wav_bytes
from .backends import OllamaLLM, create_tts, create_asr


@dataclass
class TurnResult:
    transcript: str
    answer: str
    audio_wav: bytes | None = None
    processed_wav: bytes | None = None
    timings: dict[str, float] = field(default_factory=dict)
    tts_error: str | None = None


class VoicePipeline:
    def __init__(self, config: dict, *, asr=None, llm=None, tts=None):
        self.config = config
        self.asr = asr if asr is not None else create_asr(config["asr"])
        self.llm = llm if llm is not None else OllamaLLM(config["llm"])
        self.tts = tts if tts is not None else create_tts(config["tts"])
        self.lock = RLock()

    def run(self, *, audio_wav: bytes | None = None, text: str = "",
            history: list[dict] | None = None, progress: Callable[[str], None] | None = None) -> TurnResult:
        # Cached native models may be shared across Streamlit sessions.
        with self.lock:
            return self._run(audio_wav, text, history or [], progress or (lambda _: None))

    def _run(self, audio_wav, text, history, progress):
        if audio_wav is not None and text.strip():
            raise ValueError("请只提供语音或文字中的一种输入。")
        started = perf_counter()
        timings = {}
        processed_wav = None
        if audio_wav is not None:
            progress("正在处理音频")
            stage = perf_counter()
            audio = read_wav(audio_wav, self.config["audio"]["max_seconds"])
            if np.sqrt(np.mean(audio.samples**2)) < 1e-5:
                raise ValueError("录音接近静音，请检查麦克风后重试。")
            processed = denoise(audio, self.config["audio"]["denoise"], self.config["audio"]["noise_reduction"])
            processed_wav = wav_bytes(processed)
            timings["preprocess"] = perf_counter() - stage
            progress("正在识别语音（首次使用可能需要加载或下载模型）")
            stage = perf_counter()
            text = self.asr.transcribe(processed).strip()
            timings["asr"] = perf_counter() - stage
        text = text.strip()
        if not text:
            raise ValueError("没有识别到有效文字，请靠近麦克风重试，或使用文字输入。")
        if len(text) > 4000:
            raise ValueError("输入过长，请限制在 4000 字以内。")
        progress("模型正在回答")
        stage = perf_counter()
        answer = self.llm.reply(text, history)
        timings["llm"] = perf_counter() - stage
        result = TurnResult(text, answer, processed_wav=processed_wav, timings=timings)
        progress("正在合成语音")
        stage = perf_counter()
        try:
            result.audio_wav = wav_bytes(self.tts.synthesize(answer))
        except Exception as exc:
            # Preserve the real text reply when the speech backend is unavailable.
            result.tts_error = str(exc)
        timings["tts"] = perf_counter() - stage
        timings["total"] = perf_counter() - started
        return result
