# whisper-subtitler

[![Release](https://img.shields.io/github/v/release/picommcapp/whisper-subtitler)](https://github.com/picommcapp/whisper-subtitler/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/picommcapp/whisper-subtitler/main.yml?branch=main)](https://github.com/picommcapp/whisper-subtitler/actions/workflows/main.yml?query=branch%3Amain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Transcribe audio or video and optionally label speakers. Whisper (via [faster-whisper](https://github.com/SYSTRAN/faster-whisper)) does the transcription; [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) does the diarization. FFmpeg decodes the input (mp3, wav, m4a, flac, mp4, mkv, and anything else it can read).

JSON is the default output. TXT, SRT, VTT, and TTML are available on request. CPU is the baseline; CUDA is used when it is available.

**Documentation:** <https://picommcapp.github.io/whisper-subtitler/>

## Install

You need [just](https://github.com/casey/just). Clone the repo, then:

```bash
git clone https://github.com/picommcapp/whisper-subtitler.git
cd whisper-subtitler
just install
cp .env.sample .env
```

`just install` cleans build artifacts, installs [uv](https://docs.astral.sh/uv/) and FFmpeg if they are missing, then syncs the project (Python **3.11 or 3.12**). uv will download a compatible Python if the system one is not usable.

For speaker labels, add a [HuggingFace token](https://hf.co/settings/tokens) to `.env` as `HUGGINGFACE_TOKEN` and accept the [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) terms. Transcription-only runs can skip that and pass `--no-diarization`.

Already have uv and FFmpeg?

```bash
uv sync --all-groups
```

Verify with `uv run whisper-subtitler version` or `just run version`.

## Usage

```bash
# JSON next to the input (default model: large-v3)
uv run whisper-subtitler transcribe path/to/video.mp4
uv run whisper-subtitler transcribe path/to/talk.mp3

# Top-level media in a directory, sequentially, sorted by name
uv run whisper-subtitler transcribe path/to/meetings/

# Subtitle formats from the same segments
uv run whisper-subtitler transcribe path/to/video.mp4 -f srt
uv run whisper-subtitler transcribe path/to/video.mp4 -f json srt
uv run whisper-subtitler transcribe path/to/video.mp4 -f all

# Transcription only
uv run whisper-subtitler transcribe path/to/video.mp4 --no-diarization

# Speaker count, device, smaller model
uv run whisper-subtitler transcribe path/to/video.mp4 -s 3
uv run whisper-subtitler transcribe path/to/video.mp4 --device cpu
uv run whisper-subtitler transcribe path/to/video.mp4 -m medium

# Same commands via Just
just run transcribe path/to/video.mp4 --device cpu
```

CLI flags win over `.env`. Full reference: [Usage](docs/usage.md). Troubleshooting: [FAQ](docs/faq.md).

A no-frills desktop GUI is available via `just install-gui` (or `just sync-gui` if the project is already installed), then `just gui`.

## Configuration

Settings come from CLI flags, a `.env` file, or `--config`. Copy `.env.sample` and edit; the keys that matter first are:

| Variable | Purpose |
|---|---|
| `HUGGINGFACE_TOKEN` | Required for diarization unless `--no-diarization` |
| `WHISPER_MODEL_SIZE` | Default `large-v3` |
| `WHISPER_DEVICE` | `auto`, `cpu`, or `cuda` |
| `SKIP_DIARIZATION` | `true` to skip speaker labels |
| `OUTPUT_FORMATS` | Comma-separated; default `json` |

Device `auto` uses CUDA when PyTorch can see a GPU, otherwise CPU (`int8` on CPU, `float16` on CUDA). Override with `--device` / `--compute-type`.

## Development

```bash
just install    # clean + uv + FFmpeg + uv sync
just check      # lint, typecheck, tests
just test
just docs
```

| Recipe | Purpose |
|---|---|
| `just install` | Clean, install uv and FFmpeg, sync dependencies |
| `just install-gui` | Same as `just install`, plus the GUI extra |
| `just install-uv` | Install uv if missing |
| `just install-ffmpeg` | Install FFmpeg if missing |
| `just sync` | `uv sync --all-groups` |
| `just sync-gui` | `uv sync --all-groups --extra gui` |
| `just clean` | Caches, build artifacts, `.venv` |
| `just check` | lint + typecheck + tests |
| `just run *args` | `uv run whisper-subtitler …` |
| `just gui` | Open the desktop GUI |

See [Development](docs/development.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT. See [LICENSE](LICENSE).
