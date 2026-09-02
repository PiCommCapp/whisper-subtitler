set shell := ["bash", "-uc"]

# Fresh uv installs land in ~/.local/bin (or ~/.cargo/bin on older installers).
export PATH := env("HOME") + "/.local/bin:" + env("HOME") + "/.cargo/bin:" + env("PATH")

help:
    @just --list

# Install uv if it is not already on PATH
install-uv:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v uv >/dev/null 2>&1; then
        echo "uv already installed: $(uv --version)"
        exit 0
    fi
    echo "Installing uv…"
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "error: need curl or wget to install uv" >&2
        exit 1
    fi
    if ! command -v uv >/dev/null 2>&1; then
        echo "error: uv installed but not on PATH; add ~/.local/bin to PATH and retry" >&2
        exit 1
    fi
    echo "uv installed: $(uv --version)"

# Install FFmpeg if it is not already on PATH
install-ffmpeg:
    #!/usr/bin/env bash
    set -euo pipefail
    if command -v ffmpeg >/dev/null 2>&1; then
        echo "ffmpeg already installed: $(ffmpeg -version | head -n 1)"
        exit 0
    fi
    echo "Installing FFmpeg…"
    _root() {
        if [[ "$(id -u)" -eq 0 ]]; then
            "$@"
        elif command -v sudo >/dev/null 2>&1; then
            sudo "$@"
        else
            echo "error: installing FFmpeg via the system package manager requires root or sudo" >&2
            exit 1
        fi
    }
    os="$(uname -s)"
    case "${os}" in
        Darwin)
            if ! command -v brew >/dev/null 2>&1; then
                echo "error: Homebrew is required to install FFmpeg on macOS" >&2
                exit 1
            fi
            brew install ffmpeg
            ;;
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                _root apt-get update -y
                _root apt-get install -y ffmpeg
            elif command -v dnf >/dev/null 2>&1; then
                _root dnf install -y ffmpeg
            elif command -v yum >/dev/null 2>&1; then
                _root yum install -y ffmpeg
            elif command -v pacman >/dev/null 2>&1; then
                _root pacman -Sy --noconfirm ffmpeg
            elif command -v apk >/dev/null 2>&1; then
                _root apk add --no-cache ffmpeg
            elif command -v zypper >/dev/null 2>&1; then
                _root zypper install -y ffmpeg
            elif command -v brew >/dev/null 2>&1; then
                brew install ffmpeg
            else
                echo "error: no supported package manager found (apt, dnf, yum, pacman, apk, zypper, brew)" >&2
                exit 1
            fi
            ;;
        MINGW*|MSYS*|CYGWIN*)
            if command -v winget >/dev/null 2>&1; then
                winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
            elif command -v choco >/dev/null 2>&1; then
                choco install ffmpeg -y
            elif command -v scoop >/dev/null 2>&1; then
                scoop install ffmpeg
            else
                echo "error: install FFmpeg with winget, chocolatey, or scoop, or add ffmpeg to PATH" >&2
                exit 1
            fi
            ;;
        *)
            echo "error: unsupported OS '${os}'; install FFmpeg manually and add it to PATH" >&2
            exit 1
            ;;
    esac
    if ! command -v ffmpeg >/dev/null 2>&1; then
        echo "error: FFmpeg installed but not on PATH" >&2
        exit 1
    fi
    echo "ffmpeg installed: $(ffmpeg -version | head -n 1)"

# Remove caches, build artifacts, coverage output, and the project virtualenv
clean:
    #!/usr/bin/env bash
    set -euo pipefail
    shopt -s globstar nullglob
    rm -rf \
        .venv \
        .tox \
        .nox \
        .pytest_cache \
        .ruff_cache \
        .mypy_cache \
        .pyright \
        .basedpyright \
        .coverage \
        htmlcov \
        dist \
        build \
        site \
        pytest-cache-files-* \
        coverage.xml \
        .coverage.* \
        *.egg-info \
        **/*.egg-info \
        **/__pycache__

# Clean, install uv + FFmpeg, then sync project dependencies (CLI; no GUI extra)
install: clean install-uv install-ffmpeg sync

# Same as install, plus the PySide6 GUI extra
install-gui: clean install-uv install-ffmpeg sync-gui

sync:
    uv sync --all-groups

# CLI + dev groups + optional GUI extra (PySide6)
sync-gui:
    uv sync --all-groups --extra gui

# Freeze a standalone onedir zip for this OS/arch (Linux/Windows swap in CPU torch)
dist:
    #!/usr/bin/env bash
    set -euo pipefail
    export PYTHONUNBUFFERED=1
    uv sync --all-groups --extra gui --extra packaging
    uv run --extra gui --extra packaging python packaging/build.py

lock:
    uv lock

test:
    uv run python -m pytest tests

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff check --fix .
    uv run ruff format .

typecheck:
    uv run basedpyright whisper_subtitler tests

check: lint typecheck test

run *args:
    uv run whisper-subtitler {{args}}

# Open the desktop GUI (pulls the gui extra if it is not already in the venv)
gui:
    uv run --extra gui whisper-subtitler gui

docs:
    uv run mkdocs build -s
