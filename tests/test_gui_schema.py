"""Tests for GUI env-key schema and run-config assembly (no Qt)."""

from pathlib import Path

from whisper_subtitler.modules.config import Config
from whisper_subtitler.modules.gui.log_handler import redact_secrets
from whisper_subtitler.modules.gui.run_config import build_run_config
from whisper_subtitler.modules.gui.schema import ENV_KEYS, ENV_SETTINGS, apply_settings, parse_setting


def _sample_env_keys() -> list[str]:
    keys = []
    sample = Path(__file__).resolve().parents[1] / ".env.sample"
    for line in sample.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        keys.append(stripped.split("=", 1)[0])
    return keys


def test_schema_covers_env_sample_keys():
    assert list(ENV_KEYS) == _sample_env_keys()
    assert len(ENV_SETTINGS) == len(ENV_KEYS)


def test_parse_bool_and_optional_int():
    skip = next(s for s in ENV_SETTINGS if s.key == "SKIP_DIARIZATION")
    num = next(s for s in ENV_SETTINGS if s.key == "NUM_SPEAKERS")
    assert parse_setting(skip, "true") is True
    assert parse_setting(skip, "false") is False
    assert parse_setting(num, "") is None
    assert parse_setting(num, "3") == 3


def test_parse_temperature_list():
    setting = next(s for s in ENV_SETTINGS if s.key == "WHISPER_TEMPERATURE")
    assert parse_setting(setting, "0.0,0.2") == [0.0, 0.2]


def test_apply_settings_sets_config_and_environ(monkeypatch):
    import os

    monkeypatch.setenv("OMP_NUM_THREADS", "1")
    config = Config()
    errors = apply_settings(
        config,
        {
            "WHISPER_MODEL_SIZE": "tiny",
            "SKIP_DIARIZATION": "true",
            "OMP_NUM_THREADS": "8",
        },
    )
    assert errors == []
    assert config.model_size == "tiny"
    assert config.skip_diarization is True
    assert os.environ["OMP_NUM_THREADS"] == "8"


def test_build_run_config_empty_token_skips_diarization(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    config, errors = build_run_config(
        input_file=str(media),
        output_dir=str(tmp_path),
        token="",
        formats=["json", "srt"],
        grid={"WHISPER_MODEL_SIZE": "tiny", "SKIP_DIARIZATION": "false"},
        env_file=str(tmp_path / "missing.env"),
    )
    assert errors == []
    assert config.skip_diarization is True
    assert config.output_formats == ["json", "srt"]
    assert config.model_size == "tiny"
    assert config.input_file == str(media)


def test_build_run_config_token_enables_diarization(tmp_path):
    media = tmp_path / "talk.mp3"
    media.write_bytes(b"x")
    config, errors = build_run_config(
        input_file=str(media),
        output_dir=str(tmp_path),
        token="hf_notarealtoken",
        formats=["json"],
        grid={},
        env_file=str(tmp_path / "missing.env"),
    )
    assert errors == []
    assert config.skip_diarization is False
    assert config.huggingface_token == "hf_notarealtoken"


def test_build_run_config_requires_input():
    config, errors = build_run_config(
        input_file="  ",
        output_dir="",
        token="",
        formats=["json"],
        grid={},
    )
    assert config.input_file is None or config.input_file == ""
    assert any("Input file" in err for err in errors)


def test_redact_secrets_strips_hf_tokens():
    assert "hf_abc123xyz" not in redact_secrets("token=hf_abc123xyz")
    assert redact_secrets("hello", token="secret") == "hello"
    assert "***" in redact_secrets("using secret now", token="secret")
