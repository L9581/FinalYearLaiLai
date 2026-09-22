from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import numpy as np
import pytest

from voice_lab.audio import Audio, read_wav, wav_bytes
from voice_lab.backends import GPTSoVITSTTS, RvcTTS
from voice_lab.config import load_config


def test_generated_wav_preserves_native_rate_while_asr_resamples():
    audio = Audio(np.sin(np.arange(48000, dtype=np.float32) * .1) * .1, 48000)
    wav = wav_bytes(audio)
    assert read_wav(wav).sample_rate == 16000
    assert read_wav(wav, resample=False).sample_rate == 48000
    assert read_wav(wav, resample=False).samples.shape == (48000,)


def test_v4_request_includes_reference_and_keeps_high_sample_rate(tmp_path, monkeypatch):
    reference = tmp_path / "reference.wav"
    output = wav_bytes(Audio(np.full(48000, .01, dtype=np.float32), 48000))
    reference.write_bytes(output)
    config = dict(speed=1.1, gpt_sovits=dict(base_url="http://127.0.0.1:9880", reference_audio=str(reference),
                                           reference_text="你好", reference_language="zh", text_lang="zh"))
    captured = []
    def respond(request, timeout):
        captured.append(json.loads(request.data))
        return BytesIO(output)
    monkeypatch.setattr("voice_lab.backends.urlopen", respond)
    audio = GPTSoVITSTTS(config).synthesize("新的回答")
    assert audio.sample_rate == 48000
    assert captured[0]["ref_audio_path"] == str(reference)
    assert captured[0]["prompt_text"] == "你好"
    assert captured[0]["text"] == "新的回答"
    assert captured[0]["speed_factor"] == 1.1
    assert captured[0]["streaming_mode"] is False


def test_rvc_converts_source_audio_and_does_not_hide_connection_failure(monkeypatch):
    original = Audio(np.full(48000, .02, dtype=np.float32), 48000)
    converted = Audio(np.full(40000, .03, dtype=np.float32), 40000)
    source = Mock(synthesize=Mock(return_value=original))
    def respond(request, timeout):
        assert read_wav(request.data, resample=False).sample_rate == 48000
        assert "pitch=2" in request.full_url
        return BytesIO(wav_bytes(converted))
    monkeypatch.setattr("voice_lab.backends.urlopen", respond)
    tts = RvcTTS(source, dict(base_url="http://127.0.0.1:9881", pitch=2))
    assert tts.synthesize("新的回答").sample_rate == 40000
    source.synthesize.assert_called_once_with("新的回答")
    monkeypatch.setattr("voice_lab.backends.urlopen", Mock(side_effect=URLError("refused")))
    with pytest.raises(RuntimeError, match="RVC"):
        tts.synthesize("新的回答")


def test_v4_reference_path_is_resolved_from_config_directory(tmp_path):
    original = (Path(__file__).parents[1] / "config.toml").read_text(encoding="utf-8")
    config_file = tmp_path / "config.toml"
    config_file.write_text(original + '\n[tts.gpt_sovits]\nreference_audio = "reference.wav"\n', encoding="utf-8")
    assert load_config(config_file)["tts"]["gpt_sovits"]["reference_audio"] == str(tmp_path / "reference.wav")
