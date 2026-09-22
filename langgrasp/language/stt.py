"""Speech-to-text front end: faster-whisper wrapper with lazy loading, plus a text fallback source.

Measured numbers live in results/stt_latency_l4.json (NVIDIA L4, x86 EC2 host, not Jetson). Command-domain
accuracy is not measured: there is no recorded corpus of robot commands, only LibriSpeech read speech.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_PUNCT_RE = re.compile(r"[^\w\s'-]")
_WS_RE = re.compile(r"\s+")


def normalize_command(text: str) -> str:
    """Lowercase, strip punctuation (keeps apostrophes and hyphens inside words), collapse whitespace."""
    t = text.lower().replace("’", "'")
    t = _PUNCT_RE.sub(" ", t)
    t = re.sub(r"(?<!\w)['-]|['-](?!\w)", " ", t)  # dangling quotes / dashes
    return _WS_RE.sub(" ", t).strip()


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance divided by the reference length (both normalised first)."""
    ref = normalize_command(reference).split()
    hyp = normalize_command(hypothesis).split()
    if not ref:
        return 0.0 if not hyp else 1.0
    d = np.arange(len(hyp) + 1, dtype=np.int32)
    for i, r in enumerate(ref, start=1):
        prev = d.copy()
        d[0] = i
        for j, h in enumerate(hyp, start=1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (0 if r == h else 1))
    return float(d[len(hyp)]) / len(ref)


@dataclass
class Transcript:
    text: str
    language: str
    latency_ms: float
    segments: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"text": self.text, "language": self.language, "latency_ms": self.latency_ms, "segments": self.segments}


class SpeechToText:
    """faster-whisper wrapper. The model is loaded on first use (or with ``load()``)."""

    def __init__(self, model_size: str = "base", device: str = "auto", compute_type: str = "float16", beam_size: int = 1, language: str | None = "en"):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size
        self.language = language
        self._model: Any = None
        self.load_ms: float | None = None

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device != "auto":
            return device
        try:
            import ctranslate2

            return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
        except Exception:
            return "cpu"

    def load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            dev = self._resolve_device(self.device)
            compute = self.compute_type if dev == "cuda" else "int8"
            t0 = time.perf_counter()
            self._model = WhisperModel(self.model_size, device=dev, compute_type=compute)
            self.load_ms = (time.perf_counter() - t0) * 1000.0
            self.device = dev
        return self._model

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @staticmethod
    def _to_float32(audio, sample_rate: int) -> np.ndarray:
        a = np.asarray(audio)
        if a.ndim == 2:
            a = a.mean(axis=1)
        if a.dtype.kind in "iu":
            a = a.astype(np.float32) / float(np.iinfo(a.dtype).max)
        a = a.astype(np.float32)
        if sample_rate != 16000:
            n = int(round(len(a) * 16000 / sample_rate))
            a = np.interp(np.linspace(0, len(a) - 1, n), np.arange(len(a)), a).astype(np.float32)
        return a

    def transcribe(self, path_or_array, sample_rate: int = 16000) -> dict:
        """Transcribe a file path or a 1-D float/int array. Returns dict(text, language, latency_ms, segments)."""
        model = self.load()
        audio = path_or_array if isinstance(path_or_array, str) else self._to_float32(path_or_array, sample_rate)
        t0 = time.perf_counter()
        segs, info = model.transcribe(audio, beam_size=self.beam_size, language=self.language, vad_filter=False, without_timestamps=True)
        segments = [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segs]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        text = " ".join(s["text"] for s in segments).strip()
        return Transcript(text=text, language=str(info.language), latency_ms=latency_ms, segments=segments).as_dict()

    def transcribe_command(self, path_or_array, sample_rate: int = 16000) -> str:
        return normalize_command(self.transcribe(path_or_array, sample_rate)["text"])


class TextCommandSource:
    """Fallback command source: a fixed string, a list of strings, or stdin (used by the ROS command node)."""

    def __init__(self, commands: str | list[str] | None = None):
        if commands is None:
            self._queue: list[str] | None = None
        elif isinstance(commands, str):
            self._queue = [commands]
        else:
            self._queue = list(commands)

    def next(self) -> str | None:
        if self._queue is not None:
            return normalize_command(self._queue.pop(0)) if self._queue else None
        try:
            line = input("command> ")
        except EOFError:
            return None
        return normalize_command(line)

    def __iter__(self):
        while True:
            c = self.next()
            if c is None:
                return
            yield c


# ---------------------------------------------------------------------- benchmark (python -m langgrasp.language.stt)
def load_librispeech_samples(n: int = 8) -> list[tuple[np.ndarray, str]]:
    """First n items of hf-internal-testing/librispeech_asr_dummy (clean/validation), decoded to 16 kHz float32."""
    import os
    import tempfile

    from datasets import Audio, load_dataset
    from faster_whisper.audio import decode_audio

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(decode=False))
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(min(n, len(ds))):
            item = ds[i]
            path = os.path.join(tmp, f"{i}.flac")
            with open(path, "wb") as f:
                f.write(item["audio"]["bytes"])
            out.append((decode_audio(path, sampling_rate=16000), item["text"]))
    return out


def benchmark(model_sizes=("tiny", "base", "small"), n_samples: int = 8, n_runs: int = 5, out_path: str | None = None) -> dict:
    import json
    import platform
    import subprocess

    samples = load_librispeech_samples(n_samples)
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except Exception:
        gpu = "unknown"
    results: dict = {}
    for size in model_sizes:
        stt = SpeechToText(model_size=size)
        stt.load()
        audio0 = samples[0][0]
        for _ in range(2):  # warmup
            stt.transcribe(audio0)
        lat = []
        for _ in range(n_runs):
            lat.append(stt.transcribe(audio0)["latency_ms"])
        wers, hyps = [], []
        for audio, ref in samples:
            hyp = stt.transcribe(audio)["text"]
            hyps.append(hyp)
            wers.append(word_error_rate(ref, hyp))
        results[size] = {
            "median_ms": float(np.median(lat)),
            "p90_ms": float(np.percentile(lat, 90)),
            "min_ms": float(np.min(lat)),
            "load_ms": stt.load_ms,
            "wer_on_samples": float(np.mean(wers)),
            "n_samples": len(samples),
            "n_runs": n_runs,
            "warmup_runs": 2,
            "audio_seconds_timed_clip": float(len(audio0) / 16000.0),
            "device": stt.device,
            "compute_type": stt.compute_type if stt.device == "cuda" else "int8",
            "example": {"reference": samples[0][1], "hypothesis": hyps[0]},
        }
        print(f"{size:6s} median {results[size]['median_ms']:.1f} ms p90 {results[size]['p90_ms']:.1f} ms WER {results[size]['wer_on_samples']:.3f}")
        del stt
    report = {
        "hardware": f"{gpu} (x86 EC2 host, {platform.machine()}), not Jetson",
        "backend": "faster-whisper (CTranslate2), beam_size=1, language=en, vad_filter=False",
        "dataset": "hf-internal-testing/librispeech_asr_dummy clean/validation, first n_samples items (read speech, not robot commands)",
        "latency_definition": "wall-clock of model.transcribe + segment iteration on one clip, median over n_runs after warmup",
        "note": "command-domain accuracy not measured, no recorded robot commands",
        "results": results,
    }
    if out_path:
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
    return report


if __name__ == "__main__":
    import sys

    benchmark(out_path=sys.argv[1] if len(sys.argv) > 1 else "results/stt_latency_l4.json")
