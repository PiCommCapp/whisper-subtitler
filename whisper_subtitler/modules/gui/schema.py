"""Env-key schema for the Options grid. No Qt dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Literal

from whisper_subtitler.modules.config import Config, parse_temperature

Kind = Literal[
    "str",
    "optional_str",
    "language",
    "bool",
    "int",
    "optional_int",
    "float",
    "optional_float",
    "temperature",
    "csv",
    "environ",
]


@dataclass(frozen=True)
class Setting:
    key: str
    kind: Kind
    comment: str
    attr: str | None = None
    default: str = ""


ENV_SETTINGS: tuple[Setting, ...] = (
    Setting("HUGGINGFACE_TOKEN", "optional_str", "HuggingFace token for diarization", "huggingface_token", ""),
    Setting("WHISPER_MODEL_SIZE", "str", "Whisper model size", "model_size", "large-v3"),
    Setting("WHISPER_LANGUAGE", "language", "Language code, or empty/auto", "language", "en"),
    Setting("WHISPER_BEAM_SIZE", "int", "Beam search width", "beam_size", "5"),
    Setting("WHISPER_BEST_OF", "int", "Beam-search sample count", "best_of", "5"),
    Setting(
        "WHISPER_TEMPERATURE",
        "temperature",
        "Temperature or comma-separated fallback list",
        "temperature",
        "0.0,0.2,0.4,0.6,0.8,1.0",
    ),
    Setting("WHISPER_INITIAL_PROMPT", "optional_str", "Optional initial prompt", "initial_prompt", ""),
    Setting("WHISPER_DEVICE", "str", "auto | cpu | cuda", "device", "auto"),
    Setting("WHISPER_COMPUTE_TYPE", "optional_str", "Optional faster-whisper compute type", "compute_type", ""),
    Setting(
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT",
        "bool",
        "Condition on previous text (false reduces loops)",
        "condition_on_previous_text",
        "false",
    ),
    Setting("WHISPER_VAD_FILTER", "bool", "Drop non-speech before decoding", "vad_filter", "true"),
    Setting(
        "WHISPER_COMPRESSION_RATIO_THRESHOLD",
        "float",
        "Retry if output is too repetitive",
        "compression_ratio_threshold",
        "2.4",
    ),
    Setting(
        "WHISPER_LOG_PROB_THRESHOLD", "float", "Retry if average log-prob is below this", "log_prob_threshold", "-1.0"
    ),
    Setting(
        "WHISPER_NO_SPEECH_THRESHOLD", "float", "Skip segment when no-speech prob is high", "no_speech_threshold", "0.6"
    ),
    Setting("WHISPER_REPETITION_PENALTY", "float", ">1.0 penalizes token repeats", "repetition_penalty", "1.0"),
    Setting("WHISPER_NO_REPEAT_NGRAM_SIZE", "int", "Block repeated n-grams (0=off)", "no_repeat_ngram_size", "0"),
    Setting(
        "WHISPER_HALLUCINATION_SILENCE_THRESHOLD",
        "optional_float",
        "Optional seconds; empty disables",
        "hallucination_silence_threshold",
        "",
    ),
    Setting(
        "WHISPER_VAD_MIN_SILENCE_DURATION_MS",
        "optional_int",
        "Optional VAD override (ms)",
        "vad_min_silence_duration_ms",
        "",
    ),
    Setting("WHISPER_VAD_SPEECH_PAD_MS", "optional_int", "Optional VAD speech padding (ms)", "vad_speech_pad_ms", ""),
    Setting("SKIP_DIARIZATION", "bool", "Skip speaker diarization", "skip_diarization", "false"),
    Setting("NUM_SPEAKERS", "optional_int", "Exact speaker count if known", "num_speakers", ""),
    Setting("MIN_SPEAKERS", "optional_int", "Minimum speakers", "min_speakers", "2"),
    Setting("MAX_SPEAKERS", "optional_int", "Maximum speakers", "max_speakers", ""),
    Setting("AUDIO_SAMPLE_RATE", "str", "Extracted audio sample rate", "audio_sample_rate", "16000"),
    Setting("AUDIO_CHANNELS", "str", "Audio channels (1=mono)", "audio_channels", "1"),
    Setting("AUDIO_CONVERSION", "bool", "Convert audio to optimal format", "audio_conversion", "true"),
    Setting("AUDIO_NORMALIZATION", "bool", "Normalize volume", "audio_normalization", "true"),
    Setting("NOISE_REDUCTION", "bool", "Apply noise reduction", "noise_reduction", "true"),
    Setting("HIGH_PASS_FILTER", "bool", "Apply high-pass filter", "high_pass_filter", "true"),
    Setting("HIGH_PASS_CUTOFF", "int", "High-pass cutoff (Hz)", "high_pass_cutoff", "80"),
    Setting("OUTPUT_FORMATS", "csv", "Comma-separated output formats", "output_formats", "json"),
    Setting("TTML_TITLE", "str", "TTML title", "ttml_title", "Transcription"),
    Setting("TTML_LANGUAGE", "str", "TTML language code", "ttml_language", "en-GB"),
    Setting("USE_CUDA", "bool", "Legacy CUDA flag (prefer WHISPER_DEVICE)", "use_cuda", "true"),
    Setting("OMP_NUM_THREADS", "environ", "OpenMP thread count", None, "4"),
    Setting("CUDA_VISIBLE_DEVICES", "environ", "GPU device id", None, "0"),
    Setting("LOG_LEVEL", "str", "DEBUG, INFO, WARNING, ERROR", "log_level", "INFO"),
    Setting("SHOW_SPEAKER_DEBUG", "bool", "Verbose speaker debug (maps to verbose)", "verbose", "false"),
    Setting("LOG_FILE", "optional_str", "Optional log file path", "log_file", ""),
)

ENV_KEYS: tuple[str, ...] = tuple(s.key for s in ENV_SETTINGS)


def _is_empty(raw: str) -> bool:
    return raw.strip() == "" or raw.strip().lower() in ("none", "null")


def parse_setting(setting: Setting, raw: str) -> Any:
    """Parse a grid cell into a Config value or environ string."""
    text = raw.strip()
    kind = setting.kind

    if kind == "str":
        return text
    if kind == "optional_str":
        return None if _is_empty(text) else text
    if kind == "language":
        if _is_empty(text) or text.lower() == "auto":
            return None
        return text
    if kind == "bool":
        lowered = text.lower()
        if lowered in ("1", "true", "yes"):
            return True
        if lowered in ("0", "false", "no", ""):
            return False
        raise ValueError(f"expected true/false, got {raw!r}")
    if kind == "int":
        return int(text)
    if kind == "optional_int":
        return None if _is_empty(text) else int(text)
    if kind == "float":
        return float(text)
    if kind == "optional_float":
        return None if _is_empty(text) else float(text)
    if kind == "temperature":
        if _is_empty(text):
            raise ValueError("temperature cannot be empty")
        return parse_temperature(text)
    if kind == "csv":
        return [part.strip() for part in text.split(",") if part.strip()] or ["json"]
    if kind == "environ":
        return text
    raise ValueError(f"unknown setting kind: {kind}")


def format_setting(setting: Setting, config: Config) -> str:
    """Display value for a setting from Config / environ / default."""
    if setting.kind == "environ":
        return os.environ.get(setting.key, setting.default)

    if setting.attr is None:
        return setting.default

    value = getattr(config, setting.attr)
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, list):
        return ",".join(str(item) for item in value)
    return str(value)


def apply_settings(config: Config, values: dict[str, str]) -> list[str]:
    """Apply Options-grid strings to Config (and process env). Returns error messages."""
    errors: list[str] = []
    for setting in ENV_SETTINGS:
        if setting.key not in values:
            continue
        raw = values[setting.key]
        try:
            parsed = parse_setting(setting, raw)
        except ValueError as exc:
            errors.append(f"{setting.key}: {exc}")
            continue
        if setting.kind == "environ":
            if parsed == "":
                os.environ.pop(setting.key, None)
            else:
                os.environ[setting.key] = str(parsed)
        elif setting.attr is not None:
            setattr(config, setting.attr, parsed)
    return errors
