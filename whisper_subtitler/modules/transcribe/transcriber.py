"""
Audio transcription module using faster-whisper.

This module handles the transcription of audio files using
Whisper models via CTranslate2 (faster-whisper), with CPU-first
defaults and optional CUDA acceleration.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel

from ..logger import get_logger

# Models supported by faster-whisper. Plain "large" maps to large-v3.
SUPPORTED_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
    "distil-large-v3",
)


def resolve_model_name(model_size: str) -> str:
    """Map CLI/config model aliases to faster-whisper model ids."""
    if model_size == "large":
        return "large-v3"
    return model_size


def resolve_device(config) -> str:
    """Resolve inference device from config (cpu | cuda | auto / use_cuda)."""
    device = getattr(config, "device", None)
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    # Legacy use_cuda / auto: prefer CUDA when available and not forced off
    use_cuda = getattr(config, "use_cuda", True)
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_compute_type(config, device: str) -> str:
    """Pick a compute type: explicit config, else best supported default for device.

    CUDA prefers float16 when supported. When it is not (e.g. Pascal GTX 1080),
    prefer int8 over float32 so large models fit in limited VRAM.
    """
    explicit = getattr(config, "compute_type", None)
    preferred = {
        "cuda": ["float16", "int8_float16", "int8", "int8_float32", "float32"],
        "cpu": ["int8", "int8_float32", "float32", "int16"],
    }.get(device, ["int8", "float32"])

    supported: set[str] | None = None
    try:
        import ctranslate2 as ct2

        supported = set(ct2.get_supported_compute_types(device))
    except Exception:
        supported = None

    def _first_supported() -> str:
        if supported is None:
            return preferred[0]
        for candidate in preferred:
            if candidate in supported:
                return candidate
        return next(iter(supported)) if supported else preferred[0]

    if explicit:
        if supported is not None and explicit not in supported:
            fallback = _first_supported()
            get_logger("transcribe").warning(
                f"compute_type={explicit} is not supported on {device} "
                f"(supported: {sorted(supported)}); using {fallback}"
            )
            return fallback
        return explicit

    chosen = _first_supported()
    if supported is not None and preferred[0] not in supported and chosen != preferred[0]:
        get_logger("transcribe").warning(
            f"Default {preferred[0]} is not supported on {device}; using {chosen} (supported: {sorted(supported)})"
        )
    return chosen


def _is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def should_log_progress(config: Any) -> bool:
    """Show faster-whisper progress on interactive terminals or when verbose."""
    if getattr(config, "verbose", False):
        return True
    return sys.stderr.isatty()


def format_hms(seconds: float) -> str:
    """Format a duration as M:SS, or H:MM:SS when it reaches one hour."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def collect_segments(
    segments_gen: Iterable[Any],
    info: Any,
    on_progress: Callable[[float, float], None] | None,
) -> list[Any]:
    """Consume the faster-whisper generator, optionally reporting audio-time progress.

    ``on_progress(current_seconds, total_seconds)`` uses original ``info.duration``.
    Emits 0 immediately, then throttles (0.25s or 1% of duration), and always
    emits the last segment. Skips intra-callbacks when duration is missing.
    """
    total = float(getattr(info, "duration", 0) or 0)
    if on_progress is None or total <= 0:
        return list(segments_gen)

    on_progress(0.0, total)
    collected: list[Any] = []
    last_emit = time.perf_counter()
    last_frac = 0.0
    for segment in segments_gen:
        collected.append(segment)
        current = min(float(getattr(segment, "end", 0) or 0), total)
        frac = current / total
        now = time.perf_counter()
        if frac - last_frac >= 0.01 or (now - last_emit) >= 0.25:
            on_progress(current, total)
            last_emit = now
            last_frac = frac
    if collected:
        final = min(float(getattr(collected[-1], "end", 0) or 0), total)
        on_progress(final, total)
    return collected


class Transcriber:
    """Audio transcription using faster-whisper.

    Loads Whisper models via CTranslate2 and returns the legacy result shape
    so downstream diarization/formatters stay stable.
    """

    def __init__(self, config: Any):
        """Initialize the transcriber with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_size = config.model_size
        self.language = config.language
        self.logger = get_logger("transcribe")

        requested_device = getattr(config, "device", None)
        self.device = resolve_device(config)
        if requested_device == "cuda" and self.device == "cpu":
            self.logger.warning("CUDA was requested but is not available; falling back to CPU")
        self.compute_type = resolve_compute_type(config, self.device)
        self.model: WhisperModel | None = None
        self._cuda_oom_fell_back = False

        self.transcription_options: dict[str, Any] = {
            "language": self.language,
            "beam_size": getattr(config, "beam_size", 5),
            "best_of": getattr(config, "best_of", 5),
            "temperature": getattr(config, "temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
            "initial_prompt": getattr(config, "initial_prompt", None),
            "condition_on_previous_text": getattr(config, "condition_on_previous_text", False),
            "vad_filter": getattr(config, "vad_filter", True),
            "compression_ratio_threshold": getattr(config, "compression_ratio_threshold", 2.4),
            "log_prob_threshold": getattr(config, "log_prob_threshold", -1.0),
            "no_speech_threshold": getattr(config, "no_speech_threshold", 0.6),
            "repetition_penalty": getattr(config, "repetition_penalty", 1.0),
            "no_repeat_ngram_size": getattr(config, "no_repeat_ngram_size", 0),
            "hallucination_silence_threshold": getattr(config, "hallucination_silence_threshold", None),
        }

        vad_parameters: dict[str, Any] = {}
        vad_min_silence = getattr(config, "vad_min_silence_duration_ms", None)
        vad_speech_pad = getattr(config, "vad_speech_pad_ms", None)
        if vad_min_silence is not None:
            vad_parameters["min_silence_duration_ms"] = vad_min_silence
        if vad_speech_pad is not None:
            vad_parameters["speech_pad_ms"] = vad_speech_pad
        if self.transcription_options.get("vad_filter") and vad_parameters:
            self.transcription_options["vad_parameters"] = vad_parameters

        self.transcription_options = {k: v for k, v in self.transcription_options.items() if v is not None}

    def _fallback_to_cpu_after_oom(self) -> None:
        """Drop the CUDA model and reload on CPU after an out-of-memory error."""
        self.logger.warning(f"CUDA out of memory with compute_type={self.compute_type}; falling back to CPU (int8)")
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        self._cuda_oom_fell_back = True
        try:
            torch.cuda.empty_cache()
        except Exception:
            self.logger.debug("torch.cuda.empty_cache() failed", exc_info=True)

    def load_model(self) -> WhisperModel:
        """Load the faster-whisper model.

        Returns:
            Loaded WhisperModel
        """
        if self.model is None:
            model_name = resolve_model_name(self.model_size)
            self.logger.info(
                f"Loading faster-whisper model: {model_name} (device={self.device}, compute_type={self.compute_type})"
            )
            started = time.perf_counter()
            self.model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            elapsed = time.perf_counter() - started
            self.logger.info(f"Model loaded ({elapsed:.1f}s)")
        return self.model

    def _segments_to_result(self, segments: list[Any], info: Any) -> dict[str, Any]:
        """Convert faster-whisper segments/info into the legacy result shape."""
        result_segments: list[dict[str, Any]] = []
        texts: list[str] = []

        for i, segment in enumerate(segments):
            text = segment.text.strip()
            texts.append(text)
            result_segments.append({
                "id": getattr(segment, "id", i),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "speaker": None,
            })

        return {
            "text": " ".join(t for t in texts if t).strip(),
            "segments": result_segments,
            "language": getattr(info, "language", self.language),
        }

    def transcribe(
        self,
        audio_path: str,
        reference_text: str | None = None,
        *,
        on_progress: Callable[[float, float], None] | None = None,
    ) -> dict[str, Any]:
        """Transcribe the given audio file.

        Args:
            audio_path: Path to the audio file
            reference_text: Unused; kept for call-site compatibility
            on_progress: Optional ``(current_seconds, total_seconds)`` callback
                while segments are produced. CLI leaves this unset.

        Returns:
            Dictionary containing transcription results in the legacy-compatible shape
        """
        del reference_text  # unused; retained for API compatibility
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self.load_model()
        self.logger.info(f"Transcribing: {audio_path}")

        try:
            options = self.transcription_options.copy()
            options["log_progress"] = should_log_progress(self.config)
            self.logger.debug(f"Transcription options: {options}")
            segments_gen, info = model.transcribe(str(audio_path), **options)
            segments = collect_segments(segments_gen, info, on_progress)
            result = self._segments_to_result(segments, info)
            self.logger.info(
                f"Detected language '{result.get('language')}' "
                f"({getattr(info, 'language_probability', 0):.2f}), "
                f"{len(result['segments'])} segments"
            )
            return result
        except Exception as e:
            if self.device == "cuda" and not self._cuda_oom_fell_back and _is_cuda_oom_error(e):
                self.logger.error(f"Transcription error: {e!s}")
                self._fallback_to_cpu_after_oom()
                return self.transcribe(str(audio_path), on_progress=on_progress)
            self.logger.error(f"Transcription error: {e!s}")
            raise
