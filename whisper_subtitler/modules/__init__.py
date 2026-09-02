"""
Whisper-Subtitler modules package.

This package contains all the modules for the whisper-subtitler application.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .application import Application
    from .config import Config
    from .logger import get_logger, setup_logging

__all__ = ["Application", "Config", "get_logger", "setup_logging"]


def __getattr__(name: str) -> Any:
    if name == "Application":
        from .application import Application

        return Application
    if name == "Config":
        from .config import Config

        return Config
    if name in {"get_logger", "setup_logging"}:
        from .logger import get_logger, setup_logging

        return get_logger if name == "get_logger" else setup_logging
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
