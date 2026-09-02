"""Main window: Operation + Options tabs."""

from __future__ import annotations

import time
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from whisper_subtitler.modules.config import Config
from whisper_subtitler.modules.gui.run_config import build_run_config
from whisper_subtitler.modules.gui.schema import ENV_SETTINGS, format_setting
from whisper_subtitler.modules.gui.worker import TranscribeWorker

OUTPUT_FORMATS = ("json", "txt", "srt", "vtt", "ttml")
MEDIA_FILTER = "Media files (*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.aac *.mp4 *.mkv *.webm *.mov *.avi);;All files (*)"


def _format_hms(seconds: float) -> str:
    """Format elapsed wall-clock time without importing the transcriber stack."""
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


class MainWindow(QMainWindow):
    """Stock two-tab operator window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("whisper-subtitler")
        self.resize(720, 620)
        self._running = False
        self._thread: QThread | None = None
        self._worker: TranscribeWorker | None = None
        self._progress_label = ""
        self._run_started_at = 0.0
        self._progress_timer = QTimer(self)
        self._progress_timer.setInterval(1000)
        self._progress_timer.timeout.connect(self._tick_progress_elapsed)

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._build_operation_tab(), "Operation")
        self.tabs.addTab(self._build_options_tab(), "Options")

    def _build_operation_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        form = QFormLayout()

        input_row, self.input_edit, self.input_browse = self._path_row("Input media file or directory")
        self.input_browse.clicked.connect(self._browse_input)
        form.addRow("Input file", input_row)

        output_row, self.output_edit, self.output_browse = self._path_row("Output folder")
        self.output_browse.clicked.connect(self._browse_output)
        form.addRow("Output folder", output_row)

        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText("empty = skip diarization")
        form.addRow("HF token", self.token_edit)

        formats = QWidget()
        formats_layout = QHBoxLayout(formats)
        formats_layout.setContentsMargins(0, 0, 0, 0)
        self.format_checks: dict[str, QCheckBox] = {}
        for name in OUTPUT_FORMATS:
            box = QCheckBox(name)
            box.setChecked(name == "json")
            self.format_checks[name] = box
            formats_layout.addWidget(box)
        formats_layout.addStretch()
        form.addRow("Output", formats)

        layout.addLayout(form)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        layout.addWidget(self.progress_bar)

        self.run_button = QPushButton("Run")
        self.run_button.clicked.connect(self._on_run)
        layout.addWidget(self.run_button, alignment=Qt.AlignmentFlag.AlignHCenter)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(1000)
        line_height = self.console.fontMetrics().lineSpacing()
        self.console.setMinimumHeight(line_height * 10 + 8)
        layout.addWidget(self.console, stretch=1)
        return tab

    def _path_row(self, placeholder: str) -> tuple[QWidget, QLineEdit, QPushButton]:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        edit = QLineEdit()
        edit.setPlaceholderText(placeholder)
        browse = QPushButton("…")
        layout.addWidget(edit)
        layout.addWidget(browse)
        return row, edit, browse

    def _build_options_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.options_table = QTableWidget(len(ENV_SETTINGS), 2)
        self.options_table.setHorizontalHeaderLabels(["Setting", "Value"])
        self.options_table.verticalHeader().setVisible(False)
        self.options_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.options_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.SelectedClicked
        )
        header = self.options_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

        seed = Config()
        seed.load_from_env()
        for row, setting in enumerate(ENV_SETTINGS):
            key_item = QTableWidgetItem(setting.key)
            key_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            key_item.setToolTip(setting.comment)
            value_item = QTableWidgetItem(format_setting(setting, seed))
            self.options_table.setItem(row, 0, key_item)
            self.options_table.setItem(row, 1, value_item)

        layout.addWidget(self.options_table)
        return tab

    def grid_values(self) -> dict[str, str]:
        values: dict[str, str] = {}
        for row, setting in enumerate(ENV_SETTINGS):
            item = self.options_table.item(row, 1)
            values[setting.key] = item.text() if item is not None else ""
        return values

    def selected_formats(self) -> list[str]:
        return [name for name, box in self.format_checks.items() if box.isChecked()]

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Input file", str(Path.home()), MEDIA_FILTER)
        if path:
            self.input_edit.setText(path)

    def _browse_output(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output folder", str(Path.home()))
        if path:
            self.output_edit.setText(path)

    def _append_console(self, line: str) -> None:
        self.console.appendPlainText(line)

    def _on_run(self) -> None:
        if self._running:
            return
        config, errors = build_run_config(
            input_file=self.input_edit.text(),
            output_dir=self.output_edit.text(),
            token=self.token_edit.text(),
            formats=self.selected_formats(),
            grid=self.grid_values(),
        )
        if errors:
            for error in errors:
                self._append_console(error)
            return

        self._running = True
        self.run_button.setEnabled(False)
        self._progress_label = ""
        self._run_started_at = time.monotonic()
        self.progress_bar.setRange(0, 0)
        self._apply_progress_format()
        self._progress_timer.start()

        self._thread = QThread(self)
        self._worker = TranscribeWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_console)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.failed.connect(self._on_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_worker)
        self._thread.start()

    def _apply_progress_format(self) -> None:
        elapsed = _format_hms(time.monotonic() - self._run_started_at) if self._run_started_at else "0:00"
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setFormat(f"· {elapsed}")
            return
        if self._progress_label:
            self.progress_bar.setFormat(f"{self._progress_label}  %p%  · {elapsed}")
        else:
            self.progress_bar.setFormat(f"%p%  · {elapsed}")

    def _tick_progress_elapsed(self) -> None:
        self._apply_progress_format()

    def _on_progress(self, label: str, fraction: float) -> None:
        self._progress_label = label
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(int(fraction * 100))
        self._apply_progress_format()

    def _on_finished(self, batch: object) -> None:
        results = {}
        failures = {}
        if isinstance(batch, dict):
            results = batch.get("results") or {}
            failures = batch.get("failures") or {}
        if results:
            self._append_console("Output files:")
            for input_path, outputs in results.items():
                self._append_console(f"  {input_path}")
                if isinstance(outputs, dict):
                    for fmt, path in outputs.items():
                        self._append_console(f"    {fmt.upper()}: {path}")
        if failures:
            self._append_console("Failures:")
            for input_path, error in failures.items():
                self._append_console(f"  {input_path}: {error}")
        elif results:
            self._append_console("Processing complete.")
        self._finish_run()

    def _on_failed(self, message: str) -> None:
        self._append_console(f"Error: {message}")
        self._finish_run()

    def _finish_run(self) -> None:
        self._running = False
        self._progress_timer.stop()
        self.run_button.setEnabled(True)
        if self.progress_bar.maximum() == 0:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")

    def _cleanup_worker(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
            self._worker = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._running:
            event.ignore()
            return
        event.accept()
