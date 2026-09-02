# Installation

## Prerequisites

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
