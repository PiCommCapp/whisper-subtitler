"""Frozen-app helpers (PyInstaller) and host-prerequisite messages."""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

FFMPEG_MISSING_MESSAGE = (
    "FFmpeg was not found on PATH. Install FFmpeg and ensure the `ffmpeg` "
    "command is available. Standalone binaries do not bundle FFmpeg; it is the "
    "only host prerequisite. See the FAQ: FFmpeg not found."
)


def is_frozen() -> bool:
    """True when running from a PyInstaller (or similar) bundle."""
    return bool(getattr(sys, "frozen", False))


def executable_dir() -> Path:
    """Directory containing the frozen executable, or cwd when unfrozen."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def dotenv_candidates() -> list[Path]:
    """Ordered `.env` paths for frozen runs: exe dir, then cwd (cwd overrides)."""
    exe_env = executable_dir() / ".env"
    cwd_env = Path.cwd() / ".env"
    if exe_env.resolve() == cwd_env.resolve():
        return [exe_env]
    return [exe_env, cwd_env]


def load_default_dotenv() -> None:
    """Load `.env` the same way for unfrozen installs; frozen also checks the exe dir."""
    if not is_frozen():
        load_dotenv()
        return
    paths = dotenv_candidates()
    load_dotenv(paths[0], override=False)
    if len(paths) > 1:
        load_dotenv(paths[1], override=True)
