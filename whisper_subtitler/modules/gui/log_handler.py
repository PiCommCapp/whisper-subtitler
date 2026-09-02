"""Logging helpers for the GUI console. No Qt dependency."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable

_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]+")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def redact_secrets(message: str, token: str | None = None) -> str:
    """Strip ANSI and hide HuggingFace tokens in log lines."""
    text = _ANSI_RE.sub("", message)
    if token:
        text = text.replace(token, "***")
    return _HF_TOKEN_RE.sub("hf_***", text)


class CallbackLogHandler(logging.Handler):
    """Forward formatted log records to a callable (e.g. a Qt Signal.emit)."""

    def __init__(self, callback: Callable[[str], None], token: str | None = None):
        super().__init__()
        self.callback = callback
        self.token = token
        self.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = redact_secrets(self.format(record), self.token)
        except Exception:
            message = redact_secrets(record.getMessage(), self.token)
        self.callback(message)
