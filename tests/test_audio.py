from io import BytesIO

import numpy as np
import pytest
from scipy.io import wavfile

from voice_lab.audio import Audio, denoise, mix_at_snr, read_wav, wav_bytes


def test_stereo_pcm_is_mixed_and_resampled():
    rate = 48000
    wave = 0.2 * np.sin(2 * np.pi * 440 * np.arange(rate) / rate)
    stereo = np.stack([wave, wave / 2], axis=1)
    buffer = BytesIO()
    wavfile.write(buffer, rate, (stereo * 32767).astype(np.int16))
    audio = read_wav(buffer.getvalue())
    assert audio.sample_rate == 16000
    assert audio.samples.shape == (16000,)
    assert np.max(np.abs(audio.samples)) == pytest.approx(0.15, abs=0.005)
    assert read_wav(wav_bytes(audio)).seconds == 1


@pytest.mark.parametrize("samples", [np.zeros(10), np.full(16000, np.nan), np.full(16000, np.inf)])
def test_invalid_audio_is_rejected(samples):
    buffer = BytesIO()
    wavfile.write(buffer, 16000, samples.astype(np.float32))
    with pytest.raises(ValueError):
        read_wav(buffer.getvalue())


def test_unsigned_pcm_silence_is_zero():
    buffer = BytesIO()
    wavfile.write(buffer, 16000, np.full(16000, 128, dtype=np.uint8))
    assert np.max(np.abs(read_wav(buffer.getvalue()).samples)) == 0


def test_highpass_suppresses_rumble_and_preserves_speech_frequency():
    time = np.arange(32000) / 16000
    rumble = 0.1 * np.sin(2 * np.pi * 25 * time)
    speech = 0.1 * np.sin(2 * np.pi * 1000 * time)
    filtered = denoise(Audio((rumble + speech).astype(np.float32), 16000), "highpass").samples
    middle = slice(1600, -1600)
    assert np.std(filtered[middle] - speech[middle]) < 0.005
    assert np.std(filtered[middle]) > 0.06


def test_noise_mix_has_requested_snr_without_clipping():
    time = np.arange(16000) / 16000
    clean = Audio((0.05 * np.sin(2 * np.pi * 440 * time)).astype(np.float32), 16000)
    noise = Audio((0.03 * np.sin(2 * np.pi * 1234 * time)).astype(np.float32), 16000)
    mixed = mix_at_snr(clean, noise, 10)
    actual_snr = 10 * np.log10(np.mean(clean.samples**2) / np.mean((mixed.samples - clean.samples)**2))
    assert actual_snr == pytest.approx(10, abs=0.01)
    assert np.max(np.abs(mixed.samples)) < 1


def test_zero_power_noise_is_rejected():
    with pytest.raises(ValueError, match="静音"):
        mix_at_snr(Audio(np.ones(16000), 16000), Audio(np.zeros(16000), 16000), 0)
