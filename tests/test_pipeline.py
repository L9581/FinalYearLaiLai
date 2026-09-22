from io import BytesIO
import json
from pathlib import Path
from unittest.mock import Mock
from urllib.error import URLError

import numpy as np
import pytest

from voice_lab.audio import Audio, wav_bytes
from voice_lab.backends import OllamaLLM
from voice_lab.config import load_config
from voice_lab.pipeline import VoicePipeline


@pytest.fixture
def config():
    return load_config(Path(__file__).parents[1] / "config.toml")


@pytest.fixture
def speech():
    return Audio((np.sin(np.arange(16000) * 0.1) * 0.1).astype(np.float32), 16000)


def test_complete_turn_preserves_history_and_generates_wav(config, speech):
    history = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好呀"}]
    asr = Mock(transcribe=Mock(return_value="介绍一下自己"))
    llm = Mock(reply=Mock(return_value="我是本地语音助手。"))
    tts = Mock(synthesize=Mock(return_value=speech))
    result = VoicePipeline(config, asr=asr, llm=llm, tts=tts).run(audio_wav=wav_bytes(speech), history=history)
    assert result.transcript == "介绍一下自己"
    assert result.audio_wav.startswith(b"RIFF")
    assert set(result.timings) == {"preprocess", "asr", "llm", "tts", "total"}
    llm.reply.assert_called_once_with("介绍一下自己", history)
    tts.synthesize.assert_called_once_with("我是本地语音助手。")
    assert len(history) == 2


def test_empty_transcript_does_not_call_llm(config, speech):
    llm = Mock()
    pipeline = VoicePipeline(config, asr=Mock(transcribe=Mock(return_value=" ")), llm=llm, tts=Mock())
    with pytest.raises(ValueError, match="没有识别"):
        pipeline.run(audio_wav=wav_bytes(speech))
    llm.reply.assert_not_called()


def test_tts_failure_keeps_text_answer_and_text_input_skips_asr(config):
    asr = Mock()
    pipeline = VoicePipeline(config, asr=asr, llm=Mock(reply=Mock(return_value="有效回答")),
                             tts=Mock(synthesize=Mock(side_effect=FileNotFoundError("缺少权重"))))
    result = pipeline.run(text="你好")
    assert result.answer == "有效回答"
    assert result.audio_wav is None
    assert result.tts_error == "缺少权重"
    asr.transcribe.assert_not_called()


@pytest.mark.parametrize("turns,expected", [(0, 2), (1, 4)])
def test_ollama_request_bounds_history(config, monkeypatch, turns, expected):
    config["llm"]["max_history_turns"] = turns
    history = [{"role": role, "content": str(i)} for i in range(3) for role in ("user", "assistant")]
    captured = []

    def respond(request, timeout):
        captured.append(json.loads(request.data))
        return BytesIO(json.dumps({"message": {"content": "回答"}}).encode())

    monkeypatch.setattr("voice_lab.backends.urlopen", respond)
    assert OllamaLLM(config["llm"]).reply("问题", history) == "回答"
    assert len(captured[0]["messages"]) == expected
    assert captured[0]["messages"][-1] == {"role": "user", "content": "问题"}
    assert captured[0]["stream"] is False
    assert len(history) == 6


def test_ollama_connection_error_is_actionable(config, monkeypatch):
    monkeypatch.setattr("voice_lab.backends.urlopen", Mock(side_effect=URLError("refused")))
    with pytest.raises(RuntimeError, match="启动 Ollama"):
        OllamaLLM(config["llm"]).reply("你好", [])
