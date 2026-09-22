from argparse import Namespace
import csv
import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

from voice_lab.audio import Audio, wav_bytes
from voice_lab.experiments import benchmark_asr
from voice_lab.metrics import character_errors


def test_chinese_cer_ignores_punctuation_but_counts_substitutions():
    assert character_errors("你好，世界！", "你好世介") == (1, 4)
    assert character_errors("ＡＰＩ 123", "api123") == (0, 6)
    assert character_errors("你", "你好吗") == (2, 1)
    with pytest.raises(ValueError):
        character_errors("！", "你好")


def test_asr_report_uses_corpus_cer_and_keeps_conditions_separate(tmp_path, monkeypatch):
    wav = wav_bytes(Audio((np.sin(np.arange(16000) * 0.1) * 0.1).astype(np.float32), 16000))
    (tmp_path / "test.wav").write_bytes(wav)
    (tmp_path / "manifest.csv").write_text(
        "path,text,condition\ntest.wav,你,clean\ntest.wav,一二三四五六七八九,clean\ntest.wav,你好,noise\n", encoding="utf-8",
    )
    asr = Mock(transcribe=Mock(side_effect=["预热", "他", "一二三四五六七八九", "你好"]))
    monkeypatch.setattr("voice_lab.experiments.create_asr", lambda _: asr)
    args = Namespace(config=Path(__file__).parents[1] / "config.toml", backend="whisper",
                     manifest=tmp_path / "manifest.csv", denoise=["off"], output=tmp_path / "results.csv")
    benchmark_asr(args)
    report = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert report["groups"][0]["condition"] == "clean"
    assert report["groups"][0]["cer"] == pytest.approx(0.1)
    assert report["groups"][1]["cer"] == 0
    with (tmp_path / "results.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 3
    assert asr.transcribe.call_count == 4
