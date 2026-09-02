"""Tests for packaging/build.py helpers (no PyInstaller run)."""

import importlib.util
import platform
import zipfile
from pathlib import Path

_BUILD_PATH = Path(__file__).resolve().parents[1] / "packaging" / "build.py"
_SPEC = importlib.util.spec_from_file_location("ws_dist_build", _BUILD_PATH)
assert _SPEC is not None and _SPEC.loader is not None
dist_build = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dist_build)


def test_artifact_os_arch_known():
    os_name, arch = dist_build.artifact_os_arch()
    assert os_name in {"linux", "macos", "windows"}
    assert arch in {"x86_64", "arm64"}


def test_frozen_executable_name_matches_platform():
    exe = dist_build.frozen_executable()
    if platform.system() == "Windows":
        assert exe.name == "whisper-subtitler.exe"
    else:
        assert exe.name == "whisper-subtitler"
    assert exe.parent == dist_build.COLLECT_DIR


def test_zip_collect_nests_onedir(tmp_path, monkeypatch):
    collect = tmp_path / "whisper-subtitler"
    collect.mkdir()
    (collect / "whisper-subtitler").write_text("exe")
    internal = collect / "_internal"
    internal.mkdir()
    (internal / "lib.so").write_text("so")
    monkeypatch.setattr(dist_build, "COLLECT_DIR", collect)
    monkeypatch.setattr(dist_build, "ROOT", tmp_path)
    monkeypatch.setattr(dist_build, "VERSION", "0.0-test")

    zip_path = dist_build.zip_collect("linux", "x86_64")

    assert zip_path == tmp_path / "dist" / "whisper-subtitler-0.0-test-linux-x86_64.zip"
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    assert "whisper-subtitler/whisper-subtitler" in names
    assert "whisper-subtitler/_internal/lib.so" in names
