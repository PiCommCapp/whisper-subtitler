# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onedir spec for whisper-subtitler (CLI + GUI)."""

from pathlib import Path

from PyInstaller.utils.hooks import collect_all

ROOT = Path(SPECPATH).resolve().parent

COLLECT_PACKAGES = (
    "faster_whisper",
    "ctranslate2",
    "pyannote",
    "pyannote.audio",
    "torch",
    "torchaudio",
    "torchvision",
    "librosa",
    "sklearn",
    "scipy",
    "soundfile",
    "PySide6",
    "lightning_fabric",
    "lightning",
    "pytorch_lightning",
    "speechbrain",
)

datas: list = []
binaries: list = []
hiddenimports: list = []

for package in COLLECT_PACKAGES:
    try:
        pkg_datas, pkg_binaries, pkg_hidden = collect_all(package)
    except Exception:
        continue
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

hiddenimports = sorted(set(hiddenimports))

a = Analysis(
    [str(ROOT / "whisper_subtitler" / "main.py")],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["pytest", "IPython", "tkinter"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="whisper-subtitler",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="whisper-subtitler",
)
