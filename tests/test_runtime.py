"""Tests for frozen-runtime helpers."""

from whisper_subtitler.modules.runtime import (
    FFMPEG_MISSING_MESSAGE,
    dotenv_candidates,
    executable_dir,
    is_frozen,
    load_default_dotenv,
)


def test_is_frozen_false_in_tests():
    assert is_frozen() is False


def test_executable_dir_unfrozen_is_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert executable_dir() == tmp_path


def test_executable_dir_frozen_is_exe_parent(tmp_path, monkeypatch):
    exe = tmp_path / "app" / "whisper-subtitler"
    exe.parent.mkdir()
    monkeypatch.setattr("whisper_subtitler.modules.runtime.is_frozen", lambda: True)
    monkeypatch.setattr("whisper_subtitler.modules.runtime.sys.executable", str(exe))
    assert executable_dir() == exe.parent.resolve()


def test_dotenv_candidates_dedupes_when_cwd_is_exe_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("whisper_subtitler.modules.runtime.is_frozen", lambda: True)
    monkeypatch.setattr("whisper_subtitler.modules.runtime.executable_dir", lambda: tmp_path)
    assert dotenv_candidates() == [tmp_path / ".env"]


def test_dotenv_candidates_exe_then_cwd(tmp_path, monkeypatch):
    app = tmp_path / "app"
    work = tmp_path / "work"
    app.mkdir()
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr("whisper_subtitler.modules.runtime.is_frozen", lambda: True)
    monkeypatch.setattr("whisper_subtitler.modules.runtime.executable_dir", lambda: app)
    assert dotenv_candidates() == [app / ".env", work / ".env"]


def test_load_default_dotenv_frozen_cwd_overrides_exe(tmp_path, monkeypatch):
    app = tmp_path / "app"
    work = tmp_path / "work"
    app.mkdir()
    work.mkdir()
    (app / ".env").write_text("WHISPER_MODEL_SIZE=tiny\n")
    (work / ".env").write_text("WHISPER_MODEL_SIZE=base\n")
    monkeypatch.chdir(work)
    monkeypatch.delenv("WHISPER_MODEL_SIZE", raising=False)
    monkeypatch.setattr("whisper_subtitler.modules.runtime.is_frozen", lambda: True)
    monkeypatch.setattr("whisper_subtitler.modules.runtime.executable_dir", lambda: app)

    load_default_dotenv()

    import os

    assert os.environ["WHISPER_MODEL_SIZE"] == "base"


def test_ffmpeg_missing_message_mentions_path():
    assert "FFmpeg" in FFMPEG_MISSING_MESSAGE
    assert "PATH" in FFMPEG_MISSING_MESSAGE
