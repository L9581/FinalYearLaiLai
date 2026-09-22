from pathlib import Path

import numpy as np
import pytest

from voice_lab.audio import Audio, denoise
from voice_lab.config import load_config


def test_spectral_denoise_handles_short_recording():
    pytest.importorskip("noisereduce")
    samples = np.random.default_rng(12).normal(0, 0.01, 1600).astype(np.float32)
    output = denoise(Audio(samples, 16000), "spectral")
    assert output.samples.shape == samples.shape
    assert np.isfinite(output.samples).all()


@pytest.mark.parametrize("backend", ["vits", "kokoro"])
def test_tts_settings_match_installed_sherpa_api(backend):
    sherpa = pytest.importorskip("sherpa_onnx")
    config = load_config(Path(__file__).parents[1] / "config.toml")
    settings = dict(config["tts"][backend])
    rules = settings.pop("rule_fsts")
    constructor = sherpa.OfflineTtsVitsModelConfig if backend == "vits" else sherpa.OfflineTtsKokoroModelConfig
    tts_config = sherpa.OfflineTtsConfig(
        model=sherpa.OfflineTtsModelConfig(**{backend: constructor(**settings)}, provider="cpu", num_threads=4),
        rule_fsts=rules, max_num_sentences=1,
    )
    assert tts_config.model.provider == "cpu"
