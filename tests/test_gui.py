"""Widget smoke tests for the desktop GUI. Skipped without PySide6."""

from __future__ import annotations

import os
import sys
import time

import pytest

pytest.importorskip("PySide6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication

from whisper_subtitler.modules.gui.schema import ENV_KEYS
from whisper_subtitler.modules.gui.window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication(sys.argv)
    yield app


@pytest.fixture
def window(qapp):
    win = MainWindow()
    yield win
    win.close()


def test_two_tabs_named_operation_and_options(window):
    assert window.tabs.count() == 2
    assert window.tabs.tabText(0) == "Operation"
    assert window.tabs.tabText(1) == "Options"


def test_json_checkbox_on_by_default(window):
    assert window.format_checks["json"].isChecked()
    assert not window.format_checks["srt"].isChecked()
    assert window.selected_formats() == ["json"]


def test_options_grid_lists_all_env_keys(window):
    assert window.options_table.columnCount() == 2
    assert window.options_table.rowCount() == len(ENV_KEYS)
    keys = [window.options_table.item(row, 0).text() for row in range(window.options_table.rowCount())]
    assert keys == list(ENV_KEYS)
    flags = window.options_table.item(0, 0).flags()
    assert not (flags & Qt.ItemFlag.ItemIsEditable)


def test_console_is_scrollable_ten_line_viewport(window):
    assert window.console.isReadOnly()
    assert window.console.maximumBlockCount() == 1000
    assert window.console.minimumHeight() >= window.console.fontMetrics().lineSpacing() * 10


def test_close_ignored_while_running(window):
    window._running = True
    event = QCloseEvent()
    window.closeEvent(event)
    assert not event.isAccepted()
    window._running = False
    event2 = QCloseEvent()
    window.closeEvent(event2)
    assert event2.isAccepted()


def test_run_without_input_logs_error_and_does_not_start_worker(window):
    window.input_edit.setText("")
    window._on_run()
    assert not window._running
    assert window.run_button.isEnabled()
    assert "Input file" in window.console.toPlainText()


def test_progress_format_includes_label_and_elapsed(window):
    window._run_started_at = time.monotonic()
    window._on_progress("Transcription 0:00 / 1:00", 0.15)
    fmt = window.progress_bar.format()
    assert "Transcription 0:00 / 1:00" in fmt
    assert "%p%" in fmt
    assert "·" in fmt
    assert window.progress_bar.value() == 15


def test_elapsed_timer_stops_on_finish(window):
    window._running = True
    window._progress_timer.start()
    assert window._progress_timer.isActive()
    window._finish_run()
    assert not window._progress_timer.isActive()
