"""normalize_command / WER tests (fast) and a faster-whisper model test that skips if the model is not cached."""

import os

import numpy as np
import pytest

from langgrasp.language.stt import SpeechToText, TextCommandSource, normalize_command, word_error_rate


def test_normalize_command_basic():
    assert normalize_command("Pick the RED cube, please!") == "pick the red cube please"
    assert normalize_command("  Grab   the   blue  block.  ") == "grab the blue block"
    assert normalize_command("PICK THE 'GREEN' ONE") == "pick the green one"
    assert normalize_command("don't pick the screw-driver") == "don't pick the screw-driver"
    assert normalize_command("") == ""


def test_normalize_command_curly_quotes_and_dashes():
    assert normalize_command("pick the red cube - now") == "pick the red cube now"
    assert normalize_command("it’s the can") == "it's the can"


def test_word_error_rate():
    assert word_error_rate("pick the red cube", "pick the red cube") == 0.0
    assert word_error_rate("pick the red cube", "Pick the red cube.") == 0.0
    assert word_error_rate("pick the red cube", "pick a red cube please") == pytest.approx(0.5)
    assert word_error_rate("pick the red cube", "") == 1.0
    assert word_error_rate("", "") == 0.0
    assert word_error_rate("", "x") == 1.0


def test_text_command_source():
    src = TextCommandSource(["Pick the RED cube!", "grab the can"])
    assert list(src) == ["pick the red cube", "grab the can"]
    assert TextCommandSource("Pick the can.").next() == "pick the can"


def test_stt_lazy_load_does_not_touch_model():
    stt = SpeechToText(model_size="tiny")
    assert not stt.loaded and stt.load_ms is None


def _model_cached(size: str) -> bool:
    hub = os.path.expanduser(os.environ.get("HF_HUB_CACHE", os.path.join(os.environ.get("HF_HOME", "~/.cache/huggingface"), "hub")))
    return os.path.isdir(os.path.join(hub, f"models--Systran--faster-whisper-{size}"))


@pytest.mark.skipif(not _model_cached("tiny"), reason="faster-whisper tiny model not cached")
def test_stt_transcribe_synthetic_silence():
    pytest.importorskip("faster_whisper")
    stt = SpeechToText(model_size="tiny", device="cpu")
    audio = np.zeros(16000, dtype=np.float32)  # 1 s of silence, must not crash and must return the schema
    out = stt.transcribe(audio, sample_rate=16000)
    assert set(out) == {"text", "language", "latency_ms", "segments"}
    assert out["latency_ms"] > 0 and stt.loaded and stt.load_ms is not None
    assert isinstance(out["text"], str)


@pytest.mark.skipif(not _model_cached("tiny"), reason="faster-whisper tiny model not cached")
def test_stt_real_speech_sample_if_available():
    """Uses one LibriSpeech dummy clip if the dataset is cached (no network); otherwise skips."""
    pytest.importorskip("faster_whisper")
    from langgrasp.language.stt import load_librispeech_samples

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        samples = load_librispeech_samples(1)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"librispeech dummy sample not cached: {type(e).__name__}")
    audio, ref = samples[0]
    stt = SpeechToText(model_size="tiny", device="cpu")
    out = stt.transcribe(audio)
    assert word_error_rate(ref, out["text"]) < 0.5
    assert stt.transcribe_command(audio) == normalize_command(out["text"])
