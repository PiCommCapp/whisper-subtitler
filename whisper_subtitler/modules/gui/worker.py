"""QThread worker that runs Application.process() off the UI thread."""

from __future__ import annotations

import logging
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from whisper_subtitler.modules.application import Application
from whisper_subtitler.modules.config import Config

from .log_handler import CallbackLogHandler


class TranscribeWorker(QObject):
    """Runs the pipeline; emits log/progress/finished/failed Signals only."""

    log = Signal(str)
    progress = Signal(str, float)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    @Slot()
    def run(self) -> None:
        handler = CallbackLogHandler(self.log.emit, token=self.config.huggingface_token)
        logger = logging.getLogger("whisper_subtitler")
        try:
            app = Application(self.config)
            logger.addHandler(handler)
            app.on_progress = self._on_progress
            batch: dict[str, Any] = app.process()
            self.finished.emit(batch)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            logger.removeHandler(handler)
            handler.close()

    def _on_progress(self, label: str, fraction: float) -> None:
        self.progress.emit(label, fraction)
