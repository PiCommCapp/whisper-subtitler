# Installation

## Standalone binaries

Download a zip from [GitHub Releases](https://github.com/picommcapp/whisper-subtitler/releases). You need **[FFmpeg](https://ffmpeg.org/)** on `PATH`. You do not need Python, uv, or just.

| Zip | Platform |
|-----|----------|
| `whisper-subtitler-{version}-windows-x86_64.zip` | Windows x86_64 |
| `whisper-subtitler-{version}-macos-arm64.zip` | macOS Apple Silicon |
| `whisper-subtitler-{version}-macos-x86_64.zip` | macOS Intel |
| `whisper-subtitler-{version}-linux-x86_64.zip` | Linux x86_64 |

Unzip and run `whisper-subtitler` (Windows: `whisper-subtitler.exe`) from a terminal, or launch it with no arguments to open the GUI. Place a `.env` next to the executable or in the working directory (working directory wins).

Windows and Linux zips are **CPU-only**. macOS may use Metal Performance Shaders when available. The first `transcribe` run downloads Whisper (and pyannote, if diarization is on) weights into the local HuggingFace cache — that needs network and disk, not extra host packages.

macOS Intel zips are published when the GitHub org enables larger macOS runners and sets the repository variable `ENABLE_MACOS_INTEL=true`. Until then, build on an Intel Mac with `just dist`.

Binaries are unsigned. macOS Gatekeeper and Windows SmartScreen may warn; see the [FAQ](faq.md).

## Install from source

### Prerequisites

- **[just](https://github.com/casey/just)** (recommended bootstrap)
- **Python 3.11 or 3.12** — uv will download one if needed
- **[uv](https://docs.astral.sh/uv/)** and **[FFmpeg](https://ffmpeg.org/)** — installed for you by `just install` if they are missing
- A **HuggingFace access token** if you use speaker diarization

Optional: a CUDA-capable GPU and a matching PyTorch/CUDA stack for faster inference.

## Install the project

```bash
git clone https://github.com/picommcapp/whisper-subtitler.git
cd whisper-subtitler
just install
```

`just install` removes caches and `.venv`, installs uv and FFmpeg when they are not on `PATH`, then runs `uv sync --all-groups`.

If uv and FFmpeg are already installed:

```bash
uv sync --all-groups
# or
just sync
```

Desktop GUI (optional):

```bash
just install-gui    # fresh clone: uv + FFmpeg + GUI extra
just sync-gui       # already installed: add PySide6 to the venv
just gui            # run
```

## Configure environment

```bash
cp .env.sample .env
```

Edit `.env`:

1. Set `HUGGINGFACE_TOKEN` to a token from [HuggingFace settings](https://hf.co/settings/tokens).
2. Accept the user conditions for [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) (and any gated model cards it requires).

If you only need transcription, skip the token and pass `--no-diarization`.

## Verify

```bash
uv run whisper-subtitler version
# or
just run version
```

## Device notes

- Default device is `auto`: CUDA when available, otherwise CPU.
- Force CPU with `--device cpu` (CPU default compute type is `int8`).
- CUDA default compute type is `float16` when a GPU is selected.
