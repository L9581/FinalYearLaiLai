from pathlib import Path
from unittest.mock import Mock

import numpy as np
from streamlit.testing.v1 import AppTest

from voice_lab.audio import Audio, wav_bytes
from voice_lab.pipeline import TurnResult


def test_text_conversation_does_not_resubmit_on_rerun(monkeypatch):
    audio = wav_bytes(Audio(np.full(16000, 0.01, dtype=np.float32), 16000))
    pipeline = Mock(run=Mock(return_value=TurnResult("你好", "你好，我是语音助手。", audio_wav=audio)))
    monkeypatch.setattr("voice_lab.pipeline.VoicePipeline", lambda _: pipeline)
    monkeypatch.delenv("VOICE_LAB_CONFIG", raising=False)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20).run()
    assert not app.exception
    app.radio[0].set_value("文字").run()
    app.text_area[0].set_value("你好").run()
    next(button for button in app.button if button.label == "发送并生成语音").click().run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert pipeline.run.call_count == 1
    app.run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert pipeline.run.call_count == 1
    next(button for button in app.button if button.label == "清空对话").click().run()
    assert not app.exception
    assert len(app.chat_message) == 0
    assert app.text_area[0].value == ""
