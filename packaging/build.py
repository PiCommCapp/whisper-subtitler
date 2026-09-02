"""Freeze whisper-subtitler with PyInstaller and zip the onedir output."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from whisper_subtitler.version import VERSION

ROOT = Path(__file__).resolve().parent.parent
SPEC = ROOT / "packaging" / "whisper-subtitler.spec"
COLLECT_DIR = ROOT / "dist" / "whisper-subtitler"
CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"


def artifact_os_arch() -> tuple[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    os_name = {"Windows": "windows", "Darwin": "macos", "Linux": "linux"}[system]
    arch = {
        "x86_64": "x86_64",
        "amd64": "x86_64",
        "arm64": "arm64",
        "aarch64": "arm64",
    }[machine]
    return os_name, arch


def frozen_executable() -> Path:
    name = "whisper-subtitler.exe" if platform.system() == "Windows" else "whisper-subtitler"
    return COLLECT_DIR / name


def _bare_version(dist_name: str) -> str:
    from importlib.metadata import version

    return version(dist_name).split("+", 1)[0]


def install_cpu_torch() -> None:
    if platform.system() == "Darwin":
        print("macOS: keeping lockfile torch wheels (MPS allowed)", flush=True)
        return
    torch_v = _bare_version("torch")
    torchvision_v = _bare_version("torchvision")
    torchaudio_v = _bare_version("torchaudio")
    print(
        f"Installing CPU torch=={torch_v} torchvision=={torchvision_v} torchaudio=={torchaudio_v}",
        flush=True,
    )
    cmd = [
        "uv",
        "pip",
        "install",
        f"torch=={torch_v}+cpu",
        f"torchvision=={torchvision_v}+cpu",
        f"torchaudio=={torchaudio_v}+cpu",
        "--index-url",
        CPU_TORCH_INDEX,
        "--python",
        sys.executable,
    ]
    subprocess.check_call(cmd, cwd=ROOT)


def run_pyinstaller() -> None:
    if COLLECT_DIR.exists():
        shutil.rmtree(COLLECT_DIR)
    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
        "--noconfirm",
        "--clean",
        "--distpath",
        str(ROOT / "dist"),
        "--workpath",
        str(ROOT / "build" / "pyinstaller"),
    ]
    subprocess.check_call(cmd, cwd=ROOT)


def zip_collect(os_name: str, arch: str) -> Path:
    zip_path = ROOT / "dist" / f"whisper-subtitler-{VERSION}-{os_name}-{arch}.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in COLLECT_DIR.rglob("*"):
            if path.is_file():
                arcname = Path("whisper-subtitler") / path.relative_to(COLLECT_DIR)
                zf.write(path, arcname.as_posix())
    return zip_path


def smoke_version() -> None:
    exe = frozen_executable()
    if not exe.exists():
        raise SystemExit(f"frozen executable missing: {exe}")
    env = os.environ.copy()
    env.setdefault("QT_QPA_PLATFORM", "offscreen")
    subprocess.check_call([str(exe), "version"], cwd=COLLECT_DIR, env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cpu-torch", action="store_true", help="Do not reinstall CPU torch wheels")
    parser.add_argument("--skip-smoke", action="store_true", help="Do not run frozen `version`")
    args = parser.parse_args(argv)

    os_name, arch = artifact_os_arch()
    print(f"Freezing whisper-subtitler {VERSION} for {os_name}-{arch}", flush=True)
    if not args.skip_cpu_torch:
        install_cpu_torch()
    run_pyinstaller()
    if not args.skip_smoke:
        smoke_version()
    zip_path = zip_collect(os_name, arch)
    print(f"Wrote {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
