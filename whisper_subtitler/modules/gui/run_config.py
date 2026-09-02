"""Assemble a Config from Operation widgets + Options grid. No Qt dependency."""

from __future__ import annotations

from typing import Any

from whisper_subtitler.modules.config import Config

from .schema import apply_settings


def build_run_config(
    *,
    input_file: str,
    output_dir: str,
    token: str,
    formats: list[str],
    grid: dict[str, str],
    env_file: str | None = None,
) -> tuple[Config, list[str]]:
    """Build Config: defaults → env → grid → Operation overrides.

    Empty Operation token forces skip_diarization. A non-empty token enables
    diarization and overrides HUGGINGFACE_TOKEN from the grid.
    """
    errors: list[str] = []
    input_path = input_file.strip()
    if not input_path:
        errors.append("Input file is required")

    selected_formats = [fmt for fmt in formats if fmt] or ["json"]

    config = Config()
    config.load_from_env(env_file)
    errors.extend(apply_settings(config, grid))

    overrides: dict[str, Any] = {
        "input_file": input_path or None,
        "output_formats": selected_formats,
    }
    output_path = output_dir.strip()
    if output_path:
        overrides["output_dir"] = output_path

    token_value = token.strip()
    if token_value:
        overrides["huggingface_token"] = token_value
        overrides["skip_diarization"] = False
    else:
        overrides["skip_diarization"] = True

    config.load_from_args(overrides)

    if not errors:
        try:
            config.validate()
        except ValueError as exc:
            errors.append(str(exc))

    return config, errors
