from dataclasses import dataclass
from io import BytesIO
from math import gcd
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, resample_poly, sosfiltfilt


@dataclass
class Audio:
    samples: np.ndarray
    sample_rate: int

    @property
    def seconds(self) -> float:
        return len(self.samples) / self.sample_rate


def read_wav(source: bytes | str | Path, max_seconds: float = 60, *, resample: bool = True) -> Audio:
    if isinstance(source, bytes) and len(source) > 30 * 1024 * 1024:
        raise ValueError("音频文件超过 30MB，请截取较短的 WAV。")
    try:
        sample_rate, raw = wavfile.read(BytesIO(source) if isinstance(source, bytes) else source)
    except (ValueError, EOFError) as exc:
        raise ValueError("无法读取音频，请使用未压缩的 PCM WAV 文件。") from exc
    if sample_rate < 8000 or sample_rate > 192000:
        raise ValueError("音频采样率必须在 8kHz 到 192kHz 之间。")
    if raw.ndim not in (1, 2) or raw.size == 0:
        raise ValueError("音频为空或声道格式不受支持。")
    if not 0.1 <= len(raw) / sample_rate <= max_seconds:
        raise ValueError(f"录音长度需要在 0.1 到 {max_seconds:g} 秒之间。")
    if raw.dtype == np.uint8:
        samples = (raw.astype(np.float32) - 128) / 128
    elif np.issubdtype(raw.dtype, np.signedinteger):
        samples = raw.astype(np.float32) / float(-np.iinfo(raw.dtype).min)
    elif np.issubdtype(raw.dtype, np.floating):
        samples = raw.astype(np.float32)
    else:
        raise ValueError("不支持这种 WAV 采样格式。")
    if raw.ndim == 2:
        samples = samples.mean(axis=1)
    if not np.isfinite(samples).all():
        raise ValueError("音频包含无效采样值。")
    samples = np.clip(samples, -1, 1)
    divisor = gcd(int(sample_rate), 16000)
    if resample and sample_rate != 16000:
        samples = resample_poly(samples, 16000 // divisor, sample_rate // divisor)
    return Audio(np.ascontiguousarray(samples, dtype=np.float32), 16000 if resample else int(sample_rate))


def wav_bytes(audio: Audio) -> bytes:
    samples = np.asarray(audio.samples)
    if audio.sample_rate <= 0 or samples.size == 0 or not np.isfinite(samples).all():
        raise ValueError("模型生成了空音频或无效音频。")
    buffer = BytesIO()
    wavfile.write(buffer, audio.sample_rate, (np.clip(samples, -1, 1) * 32767).astype(np.int16))
    return buffer.getvalue()


def denoise(audio: Audio, method: str, strength: float = 0.65) -> Audio:
    if method == "off":
        return audio
    if method == "highpass":
        # Suppresses low-frequency rumble; it does not remove competing voices.
        sos = butter(4, 80, btype="highpass", fs=audio.sample_rate, output="sos")
        samples = sosfiltfilt(sos, audio.samples)
    elif method == "spectral":
        if not 0 <= strength <= 1:
            raise ValueError("降噪强度必须在 0 到 1 之间。")
        try:
            import noisereduce as nr
        except ImportError as exc:
            raise RuntimeError('频谱降噪需要安装：python -m pip install -e ".[denoise]"') from exc
        samples = nr.reduce_noise(
            y=audio.samples, sr=audio.sample_rate, stationary=False, prop_decrease=strength,
        )
    else:
        raise ValueError(f"未知降噪方法：{method}")
    if not np.isfinite(samples).all():
        raise ValueError("降噪产生了无效采样，请关闭降噪后重试。")
    return Audio(np.ascontiguousarray(samples, dtype=np.float32), audio.sample_rate)


def mix_at_snr(clean: Audio, noise: Audio, snr_db: float) -> Audio:
    """Repeat noise if needed; scale both signals together to prevent clipping."""
    if clean.sample_rate != noise.sample_rate:
        raise ValueError("混音前需要统一采样率。")
    if not np.isfinite(snr_db) or not len(clean.samples) or not len(noise.samples):
        raise ValueError("信噪比必须是有限数值，且两段音频都不能为空。")
    background = np.resize(noise.samples.astype(np.float64), len(clean.samples))
    speech = clean.samples.astype(np.float64)
    speech_power = np.mean(speech**2)
    noise_power = np.mean(background**2)
    if speech_power < 1e-12 or noise_power < 1e-12:
        raise ValueError("不能用静音作为干净语音或噪声样本。")
    gain = np.sqrt(speech_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = speech + gain * background
    mixed /= max(1, float(np.max(np.abs(mixed))) / 0.99)
    return Audio(mixed.astype(np.float32), clean.sample_rate)
