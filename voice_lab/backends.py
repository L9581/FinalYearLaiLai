import json
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlencode

import numpy as np

from .audio import Audio, read_wav, wav_bytes


class WhisperASR:
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def load(self):
        if self.model is None:
            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError('请先安装 ASR 依赖：python -m pip install -e ".[whisper]"') from exc
            self.model = WhisperModel(
                self.config["whisper_model"], device=self.config["device"],
                compute_type=self.config["compute_type"],
            )

    def transcribe(self, audio: Audio) -> str:
        self.load()
        segments, _ = self.model.transcribe(
            audio.samples, language=self.config["language"] or None,
            beam_size=5, vad_filter=True, condition_on_previous_text=False,
        )
        return "".join(segment.text for segment in segments).strip()


class SenseVoiceASR:
    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def load(self):
        if self.model is None:
            try:
                from funasr import AutoModel
            except ImportError as exc:
                raise RuntimeError('请先安装 ASR 依赖：python -m pip install -e ".[sensevoice]"') from exc
            self.model = AutoModel(
                model=self.config["sensevoice_model"], device=self.config["device"],
                disable_update=True, trust_remote_code=False,
            )

    def transcribe(self, audio: Audio) -> str:
        self.load()
        from funasr.utils.postprocess_utils import rich_transcription_postprocess

        result = self.model.generate(
            input=audio.samples, cache={}, language=self.config["language"] or "auto",
            use_itn=True, batch_size_s=60,
        )
        return rich_transcription_postprocess("".join(item.get("text", "") for item in result)).strip()


class SenseVoiceOnnxASR:
    """SenseVoice via sherpa-onnx, using an explicitly configured local export."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def load(self):
        if self.model is not None:
            return
        import sherpa_onnx

        for key in ("sensevoice_onnx_model", "sensevoice_onnx_tokens"):
            if not self.config.get(key) or not Path(self.config[key]).is_file():
                raise FileNotFoundError(f"SenseVoice ONNX missing {key}: {self.config.get(key, '')}")
        self.model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=self.config["sensevoice_onnx_model"], tokens=self.config["sensevoice_onnx_tokens"],
            num_threads=int(self.config.get("num_threads", 4)),
            provider=self.config["device"], language=self.config["language"] or "auto", use_itn=True,
        )

    def transcribe(self, audio: Audio) -> str:
        self.load()
        stream = self.model.create_stream()
        stream.accept_waveform(audio.sample_rate, audio.samples)
        self.model.decode_stream(stream)
        return stream.result.text.strip()


class OllamaLLM:
    def __init__(self, config: dict):
        self.config = config

    def reply(self, text: str, history: list[dict]) -> str:
        limit = max(0, int(self.config["max_history_turns"])) * 2
        messages = [{"role": "system", "content": self.config["system_prompt"]}]
        messages.extend(history[-limit:] if limit else [])
        messages.append({"role": "user", "content": text})
        body = json.dumps({
            "model": self.config["model"], "messages": messages, "stream": False,
            "options": {"num_predict": self.config["max_reply_tokens"]},
        }).encode("utf-8")
        request = Request(
            self.config["base_url"].rstrip("/") + "/api/chat", data=body,
            headers={"Content-Type": "application/json"}, method="POST",
        )
        try:
            with urlopen(request, timeout=self.config["timeout_seconds"]) as response:
                result = json.load(response)
        except HTTPError as exc:
            detail = exc.read(2048).decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama 返回 HTTP {exc.code}：{detail}。请确认已拉取配置中的模型。") from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise RuntimeError("无法连接 Ollama 或请求超时。请启动 Ollama，检查模型、地址与超时设置。") from exc
        except (ValueError, UnicodeError) as exc:
            raise RuntimeError("Ollama 返回的内容不是有效 JSON。") from exc
        message = result.get("message", {}) if isinstance(result, dict) else {}
        answer = message.get("content", "") if isinstance(message, dict) else ""
        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError("LLM 未返回有效文本。请检查所选模型是否支持 /api/chat。")
        return answer.strip()


class SherpaTTS:
    """VITS and Kokoro use different models through the same local ONNX runtime."""

    def __init__(self, config: dict):
        self.config = config
        self.model = None

    def load(self):
        if self.model is not None:
            return
        backend = self.config["backend"]
        if backend not in {"vits", "kokoro"}:
            raise ValueError(f"未知 TTS：{backend}")
        settings = dict(self.config[backend])
        for key in ("model", "tokens", "lexicon", "voices", "data_dir", "dict_dir"):
            value = settings.get(key)
            for path in (value.split(",") if key == "lexicon" and value else [value]):
                if path and not Path(path).exists():
                    raise FileNotFoundError(f"TTS 缺少 {key}：{path}。请按 README 下载模型并修改配置。")
        rules = settings.pop("rule_fsts", "")
        for path in filter(None, rules.split(",")):
            if not Path(path).is_file():
                raise FileNotFoundError(f"TTS 缺少文本规范化规则：{path}")
        try:
            import sherpa_onnx
        except ImportError as exc:
            raise RuntimeError('请先安装 TTS 依赖：python -m pip install -e ".[tts]"') from exc
        model_config = {
            "provider": self.config["provider"], "num_threads": self.config["num_threads"],
        }
        if backend == "vits":
            model_config["vits"] = sherpa_onnx.OfflineTtsVitsModelConfig(**settings)
        else:
            model_config["kokoro"] = sherpa_onnx.OfflineTtsKokoroModelConfig(**settings)
        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(**model_config),
            rule_fsts=rules, max_num_sentences=1,
        )
        if not config.validate():
            raise ValueError("TTS 配置校验失败，请检查模型路径及配套文件。")
        self.model = sherpa_onnx.OfflineTts(config)

    def synthesize(self, text: str) -> Audio:
        self.load()
        speaker_id = int(self.config["speaker_id"])
        if not 0 <= speaker_id < self.model.num_speakers:
            raise ValueError(f"speaker_id 需要在 0 到 {self.model.num_speakers - 1} 之间。")
        speed = float(self.config["speed"])
        if speed <= 0:
            raise ValueError("TTS 语速必须大于 0。")
        result = self.model.generate(text, sid=speaker_id, speed=speed)
        samples = np.asarray(result.samples, dtype=np.float32)
        if not samples.size or not np.isfinite(samples).all():
            raise RuntimeError("TTS 没有生成有效音频，请检查文本语言和模型词典。")
        return Audio(samples, int(result.sample_rate))


def request_generated_wav(request: Request, timeout: float, name: str) -> Audio:
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read(30 * 1024 * 1024 + 1)
    except HTTPError as exc:
        detail = exc.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"{name} 返回 HTTP {exc.code}：{detail}") from exc
    except (URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"无法连接 {name} 或合成超时。请运行 scripts/start_voice_services.py 并检查服务日志。") from exc
    try:
        # Preserve v4's 48 kHz and RVC's 40 kHz; 16 kHz resampling is for ASR only.
        return read_wav(payload, max_seconds=300, resample=False)
    except ValueError as exc:
        raise RuntimeError(f"{name} 未返回有效 WAV：{exc}") from exc


class GPTSoVITSTTS:
    def __init__(self, config: dict):
        self.config = config

    def load(self):
        settings = self.config["gpt_sovits"]
        if not Path(settings["reference_audio"]).is_file():
            raise FileNotFoundError(f"GPT-SoVITS 参考音频不存在：{settings['reference_audio']}")
        if not settings["reference_text"].strip():
            raise ValueError("GPT-SoVITS v4 需要填写参考音频的准确文本。")

    def synthesize(self, text: str) -> Audio:
        self.load()
        settings = self.config["gpt_sovits"]
        body = dict(text=text, text_lang=settings.get("text_lang", "zh"),
                    ref_audio_path=settings["reference_audio"], prompt_text=settings["reference_text"],
                    prompt_lang=settings.get("reference_language", "zh"), media_type="wav",
                    streaming_mode=False, parallel_infer=False, text_split_method="cut5",
                    speed_factor=float(self.config["speed"]), seed=42,
                    sample_steps=int(settings.get("sample_steps", 32)))
        request = Request(settings["base_url"].rstrip("/") + "/tts", data=json.dumps(body).encode("utf-8"),
                          headers={"Content-Type": "application/json"}, method="POST")
        return request_generated_wav(request, float(settings.get("timeout_seconds", 180)), "GPT-SoVITS v4")


class RvcTTS:
    def __init__(self, source, settings: dict):
        self.source = source
        self.settings = settings

    def load(self):
        self.source.load()

    def synthesize(self, text: str) -> Audio:
        original = self.source.synthesize(text)
        return self.convert(original)

    def convert(self, original: Audio) -> Audio:
        query = urlencode({"pitch": int(self.settings.get("pitch", 0)),
                           "index_rate": float(self.settings.get("index_rate", 0.5)),
                           "protect": float(self.settings.get("protect", 0.33))})
        request = Request(self.settings["base_url"].rstrip("/") + "/convert?" + query,
                          data=wav_bytes(original), headers={"Content-Type": "audio/wav"}, method="POST")
        return request_generated_wav(request, float(self.settings.get("timeout_seconds", 180)), "RVC 音色转换")


def create_tts(config: dict):
    if config["backend"] == "gpt_sovits":
        tts = GPTSoVITSTTS(config)
    elif config["backend"] in {"vits", "kokoro"}:
        tts = SherpaTTS(config)
    else:
        raise ValueError(f"未知 TTS：{config['backend']}")
    if config.get("rvc", {}).get("enabled", False):
        tts = RvcTTS(tts, config["rvc"])
    return tts


def create_asr(config: dict):
    if config["backend"] == "whisper":
        return WhisperASR(config)
    if config["backend"] == "sensevoice":
        if config.get("sensevoice_runtime", "funasr") == "sherpa-onnx":
            return SenseVoiceOnnxASR(config)
        return SenseVoiceASR(config)
    raise ValueError(f"未知 ASR：{config['backend']}")
